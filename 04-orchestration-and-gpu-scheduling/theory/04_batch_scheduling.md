# 04 - AI 批处理调度

## 引言：为什么默认调度器不够用

K8s 默认调度器是为微服务设计的：一个 Pod 一个 Pod 调度，彼此独立。
但 AI 训练有根本不同的调度需求：

```
微服务调度：                    AI 训练调度：
  Pod 独立，一个失败不影响其他      多个 Worker 必须同时运行（Gang）
  随时可以调度，无时序依赖          Task 有 DAG 依赖（数据→训练→评估）
  资源量小（<1 CPU）              资源量大（8 GPU/Pod）
  无公平性需求（先到先服务）        多团队需要公平共享
  不需要抢占回收                  低优先级任务应让路给高优先级
```

这就是为什么需要专门的批处理调度器：**Volcano** 和 **Kueue**。

## 1. Volcano

### 1.1 架构

```
┌──────────────────────────────────────────────────────────────┐
│                        Volcano                                │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Scheduler   │  │  Controller  │  │   Admission  │      │
│  │              │  │   Manager    │  │   Webhook    │      │
│  │  - Actions   │  │              │  │              │      │
│  │  - Plugins   │  │  - Job       │  │  - Validate  │      │
│  │              │  │  - Queue     │  │  - Mutate    │      │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘      │
│         │                  │                                 │
│         ▼                  ▼                                 │
│  ┌─────────────────────────────────────┐                    │
│  │           K8s API Server            │                    │
│  │                                     │                    │
│  │  CRDs: Job | Queue | PodGroup       │                    │
│  └─────────────────────────────────────┘                    │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 核心概念

**Job（vcjob）**— Volcano 的工作负载单位

```yaml
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: distributed-training
spec:
  minAvailable: 4    # 最少需要 4 个 Pod 才能运行（Gang）
  schedulerName: volcano
  queue: training-queue
  policies:
  - event: PodEvicted
    action: RestartJob
  - event: PodFailed
    action: RestartJob
  tasks:
  - replicas: 1
    name: master
    template:
      spec:
        containers:
        - name: trainer
          image: pytorch-train:v1
          resources:
            limits:
              nvidia.com/gpu: 1
  - replicas: 3
    name: worker
    template:
      spec:
        containers:
        - name: trainer
          image: pytorch-train:v1
          resources:
            limits:
              nvidia.com/gpu: 8
```

**Queue** — 资源隔离和公平共享

```yaml
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: training-queue
spec:
  weight: 4           # 公平共享权重
  capability:         # 队列资源上限
    cpu: "128"
    memory: "512Gi"
    nvidia.com/gpu: "32"
  reclaimable: true   # 空闲资源可被其他队列借用
  guarantee:          # 保底资源（即使空闲也不被抢占）
    resource:
      cpu: "32"
      memory: "128Gi"
      nvidia.com/gpu: "8"
---
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: inference-queue
spec:
  weight: 2
  capability:
    nvidia.com/gpu: "16"
  reclaimable: true
```

**PodGroup** — Gang Scheduling 的核心

```yaml
apiVersion: scheduling.volcano.sh/v1beta1
kind: PodGroup
metadata:
  name: training-pg
spec:
  minMember: 4           # 最少 4 个 Pod 才能启动
  minResources:
    nvidia.com/gpu: "32" # 总共需要 32 GPU
  queue: training-queue
  priorityClassName: training-normal
```

### 1.3 调度动作 (Actions)

Volcano 调度器通过一系列 **Action** 实现调度逻辑：

```yaml
# volcano-scheduler-configmap
apiVersion: v1
kind: ConfigMap
metadata:
  name: volcano-scheduler-configmap
  namespace: volcano-system
data:
  volcano-scheduler.conf: |
    actions: "enqueue, allocate, preempt, reclaim, backfill"
    tiers:
    - plugins:
      - name: priority
      - name: gang
        enablePreemptable: false
      - name: conformance
    - plugins:
      - name: overcommit
      - name: drf
      - name: predicates
      - name: proportion
      - name: nodeorder
      - name: binpack
        arguments:
          binpack.weight: 5
          binpack.cpu: 2
          binpack.memory: 1
          binpack.resources: nvidia.com/gpu
          binpack.resources.nvidia.com/gpu: 10
```

Actions 执行顺序：

```
1. Enqueue  ── 检查 PodGroup 的 minMember 资源是否可能满足
               如果集群总资源都不够，不进入调度队列
               
2. Allocate ── 核心分配动作
               按优先级遍历 Job
               对每个 Job 的 Task 分配节点
               Gang 语义：要么全部分配，要么全不分配
               
3. Preempt  ── 高优先级 Job 抢占低优先级 Job
               回收被抢占 Pod 的资源
               
4. Reclaim  ── 队列间资源回收
               超出 guarantee 的资源可被其他队列回收
               
5. Backfill ── 填充空闲资源
               不需要 Gang 的小任务可以填充碎片资源
```

### 1.4 核心插件

**Gang Scheduling 插件**

```
问题：分布式训练需要 4 个 Pod 同时运行
     如果调度器只调度了 3 个，第 4 个无资源
     3 个 Pod 空等第 4 个 → 资源浪费（死锁风险）

Gang Scheduling 解决方案：
  1. 检查所有 Pod 是否都能找到节点
  2. 如果是：同时绑定所有 Pod
  3. 如果否：所有 Pod 都不调度，释放已预留的资源

实现机制：在 Permit 阶段 Hold 所有 Pod
         直到 minMember 数量的 Pod 都通过调度
         然后一起 Approve
```

**DRF (Dominant Resource Fairness) 插件**

```
问题：Team A 主要用 GPU，Team B 主要用 CPU
     如何公平分配？

DRF 算法：
  1. 计算每个队列的 "dominant resource share"
     Team A: GPU 用了集群 60% → dominant share = 60%
     Team B: CPU 用了集群 40% → dominant share = 40%
  2. 优先调度 dominant share 小的队列
     → Team B 优先（40% < 60%）
  3. 结果：各队列的 dominant share 趋于相等

类比 Slurm：Fair-share scheduling
区别：DRF 考虑多维资源（CPU+GPU），Slurm fair-share 通常按单维度
```

**Proportion 插件**

```
按 Queue 的 weight 比例分配资源：
  training-queue (weight=4): 获得 4/(4+2) = 67% 资源
  inference-queue (weight=2): 获得 2/(4+2) = 33% 资源

当某队列空闲时，资源可被其他队列借用（reclaimable=true）
当队列有新 Job 时，回收借出的资源（Reclaim action）
```

**Binpack 插件**

```
GPU 装箱策略：
  优先将 Pod 调度到已有 GPU 被使用的节点
  目标：减少 GPU 碎片，留出整机给大任务

分数计算：
  score = (requested_gpu / allocatable_gpu) * weight
  节点已用 GPU 越多，分数越高 → 优先调度到该节点
```

### 1.5 与默认调度器的共存

```yaml
# Volcano 不替换默认调度器，而是作为第二调度器运行
# GPU 训练任务 → Volcano
# 普通微服务 → 默认调度器

# 训练 Pod 指定使用 Volcano
spec:
  schedulerName: volcano

# 普通 Pod 使用默认调度器（不需要指定）
spec:
  # schedulerName 默认是 default-scheduler
```

## 2. Kueue

### 2.1 设计理念差异

```
Volcano：自己是调度器，替代 kube-scheduler 做调度决策
Kueue：不是调度器！是 "调度准入控制器"
       决定 Job 何时可以开始，但调度决策仍由 kube-scheduler 做

工作流程：
  Volcano: Pod → Volcano Scheduler → 分配节点
  Kueue:   Job → Kueue 决定是否准入 → kube-scheduler 调度 Pod
```

**为什么这个设计更优？**

```
1. 不和默认调度器冲突（资源视图一致）
2. 利用默认调度器的成熟能力（Topology, Affinity...）
3. Kueue 只需做 "队列管理" 和 "准入控制"
4. 是 K8s SIG-Scheduling 官方项目（未来可能成标准）
```

### 2.2 核心概念

```
┌──────────────────────────────────────────────────────────────┐
│                         Kueue                                 │
│                                                              │
│  Workload (自动创建)                                          │
│    ↕                                                         │
│  LocalQueue (Namespace 级)                                   │
│    ↕                                                         │
│  ClusterQueue (集群级，绑定 ResourceFlavor)                    │
│    ↕                                                         │
│  ResourceFlavor (描述硬件特性)                                 │
│    ↕                                                         │
│  Cohort (队列组，允许资源借用)                                  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**ResourceFlavor** — 描述硬件类型

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: h20-gpu
spec:
  nodeLabels:
    nvidia.com/gpu.product: "NVIDIA-H20"
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: cpu-only
spec:
  nodeLabels:
    node-type: cpu
```

**ClusterQueue** — 集群级资源池

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: gpu-cluster-queue
spec:
  cohort: "all-teams"    # 属于 "all-teams" 队列组
  queueingStrategy: BestEffortFIFO
  
  resourceGroups:
  - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
    flavors:
    - name: h20-gpu
      resources:
      - name: "nvidia.com/gpu"
        nominalQuota: 32        # 正常配额：32 GPU
        borrowingLimit: 8       # 最多可从同 cohort 借 8 GPU
        lendingLimit: 16        # 最多可借出 16 GPU
      - name: "cpu"
        nominalQuota: 256
      - name: "memory"
        nominalQuota: "1024Gi"

  preemption:
    reclaimWithinCohort: Any    # 可以回收借出的资源
    withinClusterQueue: LowerPriority  # 队列内低优先级可被抢占
```

**LocalQueue** — Namespace 级队列

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: team-a-queue
  namespace: team-a
spec:
  clusterQueue: gpu-cluster-queue  # 关联到 ClusterQueue
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: team-b-queue
  namespace: team-b
spec:
  clusterQueue: gpu-cluster-queue
```

**Workload** — Kueue 的调度单位

```yaml
# Workload 通常由 Kueue 自动从 Job/RayJob 等创建
# 这里展示它的结构：
apiVersion: kueue.x-k8s.io/v1beta1
kind: Workload
metadata:
  name: training-job-xxxxx
  namespace: team-a
spec:
  queueName: team-a-queue
  priorityClassName: training-normal
  podSets:
  - name: master
    count: 1
    template:
      spec:
        containers:
        - resources:
            requests:
              nvidia.com/gpu: "1"
  - name: workers
    count: 3
    template:
      spec:
        containers:
        - resources:
            requests:
              nvidia.com/gpu: "8"
```

### 2.3 调度流程

```
1. 用户提交 Job（带 kueue.x-k8s.io/queue-name label）
   │
2. Kueue Controller 创建 Workload 对象
   │
3. Workload 进入 LocalQueue
   │
4. Kueue 检查 ClusterQueue 配额
   │
   ├── 配额充足 → Admit（设置 Workload.status.admission）
   │                → Job 的 Pod 被 unsuspend，开始调度
   │
   └── 配额不足 → 排队等待
       │
       ├── 有低优先级 Workload → 考虑抢占
       │
       └── 等待其他 Workload 完成释放配额
```

### 2.4 Kueue 支持的 Job 类型

```yaml
# 原生 K8s Job
apiVersion: batch/v1
kind: Job
metadata:
  labels:
    kueue.x-k8s.io/queue-name: team-a-queue  # 关键标签
spec:
  suspend: true  # 必须！Kueue 控制何时 unsuspend
  template:
    spec:
      containers:
      - name: train
        resources:
          requests:
            nvidia.com/gpu: "8"
---
# RayJob（KubeRay）
apiVersion: ray.io/v1
kind: RayJob
metadata:
  labels:
    kueue.x-k8s.io/queue-name: team-a-queue
spec:
  suspend: true
  # ...
---
# PyTorchJob (Kubeflow Training Operator)
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  labels:
    kueue.x-k8s.io/queue-name: team-a-queue
spec:
  # Kueue 自动管理 suspend
```

### 2.5 Cohort（队列组）与资源借用

```
Cohort: "all-teams"
  ├── ClusterQueue: "team-a-cq" (nominalQuota: 16 GPU)
  │     borrowingLimit: 8 GPU
  │     lendingLimit: 8 GPU
  │
  └── ClusterQueue: "team-b-cq" (nominalQuota: 16 GPU)
        borrowingLimit: 8 GPU
        lendingLimit: 8 GPU

场景：Team A 只用了 8 GPU，Team B 需要 24 GPU
  1. Team B 使用自己的 16 GPU
  2. Team A 借出 8 GPU（lendingLimit=8）
  3. Team B 借入 8 GPU（borrowingLimit=8）
  4. 当 Team A 提交新 Job 需要资源时：
     Kueue 通过 preemption 回收借出的资源
```

## 3. Volcano vs Kueue 对比

| 维度 | Volcano | Kueue |
|------|---------|-------|
| **定位** | 完整调度器 | 准入控制器 |
| **Gang Scheduling** | 原生支持 | 通过 PodGroup 支持 |
| **与 kube-scheduler** | 替代/并行 | 协作 |
| **Queue 管理** | Queue CR | LocalQueue + ClusterQueue |
| **公平共享** | DRF 插件 | Cohort + borrowing |
| **抢占** | Preempt Action | 内置 preemption 策略 |
| **成熟度** | CNCF Incubating | K8s SIG 官方项目 |
| **社区活跃度** | 高 | 非常高（增长快）|
| **适用规模** | 中大型集群 | 任意规模 |
| **学习曲线** | 中等 | 较低 |
| **Job DAG** | 支持 | 不支持（需配合 Argo）|
| **多集群** | 不支持 | MultiKueue 支持 |

### 选择建议

```
选择 Volcano：
  ✅ 需要复杂的 Job DAG（task dependencies）
  ✅ 需要 Gang Scheduling（分布式训练核心需求）
  ✅ 已有 Volcano 生态（MPI Operator 等）
  ✅ 需要更细粒度的调度插件控制

选择 Kueue：
  ✅ 想保持默认调度器的所有能力
  ✅ 多团队资源共享和配额管理
  ✅ 与 KubeRay、Kubeflow 集成
  ✅ 追求简单性和标准化
  ✅ 需要多集群调度（MultiKueue）
  ✅ 社区长期支持（K8s SIG 项目）

对于你的 8xH20 集群：
  如果主要跑分布式训练 → Volcano（Gang 更成熟）
  如果混合训练+推理+多团队 → Kueue（资源管理更灵活）
  也可以两者结合：Kueue 管配额 + Volcano 管 Gang
```

## 4. Coscheduling（Scheduler Plugins）

除了 Volcano 和 Kueue，还有一个轻量级 Gang Scheduling 方案：
K8s Scheduler Plugins 项目中的 **Coscheduling 插件**。

```yaml
# 以 kube-scheduler 插件形式运行
# 不需要额外部署完整的 Volcano
apiVersion: kubescheduler.config.k8s.io/v1
kind: KubeSchedulerConfiguration
profiles:
- schedulerName: default-scheduler
  plugins:
    queueSort:
      enabled:
      - name: Coscheduling
    preFilter:
      enabled:
      - name: Coscheduling
    permit:
      enabled:
      - name: Coscheduling
    postBind:
      enabled:
      - name: Coscheduling
  pluginConfig:
  - name: Coscheduling
    args:
      permitWaitingTimeSeconds: 60  # 等待 Gang 成员的超时时间
```

```yaml
# 使用 PodGroup 定义 Gang
apiVersion: scheduling.sigs.k8s.io/v1alpha1
kind: PodGroup
metadata:
  name: training-gang
spec:
  scheduleTimeoutSeconds: 300
  minMember: 4
---
# Pod 关联 PodGroup
apiVersion: v1
kind: Pod
metadata:
  labels:
    scheduling.sigs.k8s.io/pod-group: training-gang
spec:
  schedulerName: default-scheduler
  containers:
  - name: worker
    resources:
      limits:
        nvidia.com/gpu: 8
```

## 5. 实际部署建议

### 你的 8xH20 集群调度策略

```yaml
# 推荐方案：Kueue + Coscheduling

# 1. Kueue 管理配额和队列
#    - 多团队公平共享
#    - 自动准入控制
#    - 与 KubeRay 集成

# 2. Coscheduling 提供 Gang Scheduling
#    - 轻量级，作为调度器插件
#    - 不需要部署完整的 Volcano

# 3. 资源规划
ClusterQueue:
  total: 8 nodes × 8 GPU = 64 GPU
  training-cq: nominalQuota 48 GPU (75%)
  inference-cq: nominalQuota 12 GPU (19%)
  dev-cq: nominalQuota 4 GPU (6%)
  
# Cohort 允许弹性借用
# 夜间训练可以使用推理队列的空闲 GPU
# 白天推理可以回收
```

## 小结

| Slurm 概念 | K8s 批调度对应 | 备注 |
|------------|--------------|------|
| Partition | Kueue ClusterQueue / Volcano Queue | 资源分区 |
| QOS | PriorityClass + preemption policy | 服务质量 |
| Fair-share | Kueue Cohort / Volcano DRF | 公平共享 |
| Job Array | K8s Job (parallelism) | 并行任务 |
| Job Dependency | Volcano Job DAG / Argo Workflows | 任务依赖 |
| Backfill | Volcano backfill action | 回填调度 |
| Reservation | Kueue ResourceFlavor + quota | 资源预留 |
| `squeue` | `kubectl get workloads` / `kubectl get vcjob` | 查看队列 |
