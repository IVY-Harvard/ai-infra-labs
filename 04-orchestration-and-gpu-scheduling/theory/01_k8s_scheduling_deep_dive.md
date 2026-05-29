# 01 - K8s 调度器全解析

## 引言：从 Slurm 到 K8s 调度

你在 Slurm 中提交 `sbatch --gres=gpu:8 train.sh` 时，Slurm 调度器会检查所有节点的可用
GPU、内存、CPU，按优先级和公平份额策略选择节点。K8s 调度器做的事情类似，但架构完全不同。

Slurm 是面向 HPC 设计的集中式调度器，而 K8s 调度器 (kube-scheduler) 是面向微服务设计的、
以 Pod 为粒度的通用调度器。理解它的内部机制，是做好 GPU 调度优化的基础。

## 1. 调度器架构总览

```
                    ┌─────────────────────────────────────┐
                    │           kube-scheduler             │
                    │                                     │
  Informer Cache ──►│  Scheduling Queue                   │
  (Pod/Node/PV...) │  ┌─────────┐  ┌─────────┐          │
                    │  │ Active  │→│ Backoff │          │
                    │  │ Queue   │  │ Queue   │          │
                    │  └────┬────┘  └────┬────┘          │
                    │       │            │                │
                    │       ▼            │                │
                    │  ┌─────────────────┘                │
                    │  │ Scheduling Cycle                  │
                    │  │ (Filter → Score → Reserve)       │
                    │  └──────────┬──────────────────────│
                    │             ▼                        │
                    │  Binding Cycle (Permit → Bind)      │
                    └─────────────────────────────────────┘
```

kube-scheduler 本质上是一个 **控制循环**：
1. 从调度队列取出一个待调度的 Pod
2. 执行调度周期，找到最合适的节点
3. 执行绑定周期，将 Pod 绑定到节点
4. 重复

### 调度队列的三级结构

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  ActiveQueue │    │ BackoffQueue │    │Unschedulable │
│              │    │              │    │   Queue      │
│ 待调度 Pod   │    │ 调度失败暂退  │    │ 无法调度 Pod │
│ 按优先级排序  │    │ 指数退避等待  │    │ 等待集群变化  │
└──────────────┘    └──────────────┘    └──────────────┘
```

- **ActiveQueue**：按 PriorityClass 优先级排序，高优先级 Pod 先被调度
- **BackoffQueue**：调度失败的 Pod 进入指数退避（1s, 2s, 4s...最大 10s）
- **UnschedulableQueue**：确定无法调度的 Pod，等待集群事件（新节点加入等）触发重试

## 2. 调度周期详解

### 2.1 预选阶段 (Filter)

Filter 的目的是 **排除不满足条件的节点**，类似 Slurm 的 constraint 检查。

核心 Filter 插件：

| 插件 | 功能 | GPU 场景示例 |
|------|------|-------------|
| NodeResourcesFit | 检查资源是否充足 | 节点剩余 GPU 数 >= 请求数 |
| NodeAffinity | 节点亲和性 | 只调度到有 H20 GPU 的节点 |
| TaintToleration | 污点容忍 | GPU 节点设置 nvidia.com/gpu taint |
| PodTopologySpread | 拓扑分布 | 训练 Pod 均匀分布到不同机架 |
| InterPodAffinity | Pod 间亲和 | 训练 Pod 调度到同一节点/机架 |

Filter 阶段的关键优化 — **百分比筛选**：

```go
// 默认配置：当集群有 100+ 节点时，不需要检查所有节点
// 找到足够多的可行节点就停止
percentageOfNodesToScore: 50  // 检查 50% 的节点
// 但最少检查 100 个节点（hardcoded minFeasibleNodesToFind）
```

对于 8 节点的 GPU 集群，所有节点都会被检查。但在大规模集群中，这个优化很重要。

### 2.2 优选阶段 (Score)

Score 的目的是对通过 Filter 的节点 **打分排序**，选出最佳节点。

核心 Score 插件：

| 插件 | 策略 | 说明 |
|------|------|------|
| NodeResourcesBalancedAllocation | 均衡分配 | 倾向于 CPU/Memory/GPU 利用率均衡的节点 |
| NodeResourcesFit (LeastAllocated) | 最少使用 | 倾向于资源空闲多的节点（**默认策略**）|
| NodeResourcesFit (MostAllocated) | 最多使用 | 倾向于把节点填满（装箱策略）|
| InterPodAffinity | 亲和打分 | Pod 亲和性满足程度越高分越高 |
| ImageLocality | 镜像本地性 | 已有所需镜像的节点得分更高 |

**GPU 场景关键决策：LeastAllocated vs MostAllocated**

```yaml
# 默认 LeastAllocated（分散策略）：
# 结果：每个节点用 1-2 张 GPU，8 个节点都在用
# 优点：单任务性能好，散热均匀
# 缺点：无法运行需要 8 GPU 的大任务

# MostAllocated（装箱策略）：
# 结果：先填满一个节点的 8 张 GPU，再用下一个
# 优点：腾出整机给大任务，省电
# 缺点：热节点负载高

# 对于 AI 训练集群，通常选择 MostAllocated（装箱）
apiVersion: kubescheduler.config.k8s.io/v1
kind: KubeSchedulerConfiguration
profiles:
- schedulerName: default-scheduler
  pluginConfig:
  - name: NodeResourcesFit
    args:
      scoringStrategy:
        type: MostAllocated
        resources:
        - name: nvidia.com/gpu
          weight: 10    # GPU 权重最高
        - name: cpu
          weight: 1
        - name: memory
          weight: 1
```

### 2.3 Reserve → Permit → Bind

```
Score 完成后：
  │
  ▼
Reserve（预留）── 在调度缓存中标记资源已分配（乐观绑定）
  │                避免并发调度分配同一资源
  ▼
Permit（准入）── 三种结果：
  │              - Approve：立即绑定
  │              - Deny：拒绝调度
  │              - Wait：等待（Gang Scheduling 用）
  ▼
Bind（绑定）── 调用 API Server 创建 Binding 对象
               将 Pod 的 spec.nodeName 设为目标节点
```

**Permit 阶段对 GPU 训练至关重要**：Gang Scheduling（Volcano/Coscheduling）就是在
Permit 阶段让所有 worker Pod 等待，直到全部 Pod 都找到节点，才一起放行。

## 3. Scheduling Framework（调度框架）

K8s 1.19+ 引入的 Scheduling Framework 定义了 **15 个扩展点**：

```
调度周期（同步，串行处理一个 Pod）：
  PreEnqueue → QueueSort → PreFilter → Filter → PostFilter →
  PreScore → Score → NormalizeScore → Reserve → Permit

绑定周期（异步，可并行处理多个 Pod）：
  PreBind → Bind → PostBind
```

每个扩展点的作用：

```
PreEnqueue    ── Pod 进入队列前的检查
QueueSort     ── 决定队列中 Pod 的排序
PreFilter     ── 预处理（计算一些共享状态供 Filter 用）
Filter        ── 过滤不满足条件的节点
PostFilter    ── 所有节点都被过滤掉时触发（抢占逻辑在这里）
PreScore      ── 打分前预处理
Score         ── 给节点打分
NormalizeScore── 归一化分数到 [0, 100]
Reserve       ── 预留资源
Permit        ── 准入控制（支持 Wait）
PreBind       ── 绑定前操作（如创建 PV）
Bind          ── 执行绑定
PostBind      ── 绑定后清理
```

## 4. 自定义调度器的三种方式

### 方式一：Scheduler Extender（Webhook 扩展）

```
kube-scheduler ──HTTP──► Extender Server
                          │
                          ├── /filter   ── 额外过滤
                          ├── /prioritize── 额外打分
                          └── /bind     ── 自定义绑定
```

```yaml
# scheduler-config.yaml
apiVersion: kubescheduler.config.k8s.io/v1
kind: KubeSchedulerConfiguration
extenders:
- urlPrefix: "http://gpu-topology-extender:8888"
  filterVerb: "filter"
  prioritizeVerb: "prioritize"
  weight: 5
  enableHTTPS: false
  nodeCacheCapable: false
  managedResources:
  - name: "nvidia.com/gpu"
    ignoredByScheduler: false
```

**优点**：不需要修改调度器代码，独立部署
**缺点**：HTTP 调用延迟高，只能扩展 Filter 和 Score，无法访问调度框架完整状态
**适用**：快速原型验证，不想 fork 调度器

### 方式二：Scheduling Plugin（框架内插件）

```go
// 实现 framework.FilterPlugin 和 framework.ScorePlugin 接口
type GPUTopologyPlugin struct {
    handle framework.Handle
}

func (p *GPUTopologyPlugin) Name() string {
    return "GPUTopology"
}

func (p *GPUTopologyPlugin) Filter(ctx context.Context,
    state *framework.CycleState, pod *v1.Pod,
    nodeInfo *framework.NodeInfo) *framework.Status {
    // 检查 GPU 拓扑是否满足要求
    // 比如：要求 4 GPU 必须在同一 NVLink 域
    return framework.NewStatus(framework.Success, "")
}

func (p *GPUTopologyPlugin) Score(ctx context.Context,
    state *framework.CycleState, pod *v1.Pod,
    nodeName string) (int64, *framework.Status) {
    // 按 GPU 拓扑优劣打分
    // NVSwitch 全连接 > NVLink 部分连接 > PCIe
    return score, framework.NewStatus(framework.Success, "")
}
```

**优点**：性能好（进程内调用），可以访问调度框架所有状态，支持所有 15 个扩展点
**缺点**：需要编译到调度器中，升级调度器时需要重新编译
**适用**：生产环境，需要复杂调度逻辑

### 方式三：独立调度器（Secondary Scheduler）

```yaml
# 部署第二个调度器
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpu-scheduler
spec:
  template:
    spec:
      containers:
      - name: gpu-scheduler
        image: my-gpu-scheduler:v1
        command:
        - kube-scheduler
        - --config=/etc/scheduler/config.yaml
        - --leader-elect=false  # 或使用不同的 lock name
---
# Pod 指定使用自定义调度器
apiVersion: v1
kind: Pod
metadata:
  name: gpu-training
spec:
  schedulerName: gpu-scheduler  # 指定调度器
  containers:
  - name: train
    resources:
      limits:
        nvidia.com/gpu: 4
```

**优点**：完全独立，不影响默认调度器，可以用任意语言实现
**缺点**：资源视图可能不一致（两个调度器并发操作），需要处理冲突
**适用**：GPU 工作负载与普通工作负载调度策略差异极大时

## 5. 调度性能与调优

### 调度吞吐量

默认调度器处理能力：~100 Pods/sec（取决于集群规模和插件复杂度）

```yaml
# 关键调优参数
apiVersion: kubescheduler.config.k8s.io/v1
kind: KubeSchedulerConfiguration
# 并行度：同时评估多少节点
parallelism: 16
# 百分比筛选：大集群中不需要评估所有节点
percentageOfNodesToScore: 50
```

### 调度延迟分析

```
典型调度延迟分布：
  Queue Wait:    10-100ms（取决于队列深度）
  Filter:        1-5ms（8 节点集群）
  Score:         1-3ms
  Bind:          5-20ms（API Server 写入）
  Total:         ~20-130ms per Pod

GPU 训练任务的特殊性：
  Pod 数少但每个 Pod 资源量大 → 调度延迟不是瓶颈
  关键瓶颈在于：能否找到满足拓扑要求的节点组合
```

### 监控调度器

```bash
# 关键 Metrics
scheduler_scheduling_algorithm_duration_seconds  # 调度算法耗时
scheduler_binding_duration_seconds               # 绑定耗时
scheduler_pending_pods                           # 各队列待调度 Pod 数
scheduler_schedule_attempts_total                # 调度尝试总数（按结果分）
scheduler_plugin_execution_duration_seconds      # 各插件执行耗时
```

## 6. 对你的 8xH20 集群的实际建议

```yaml
# 推荐调度配置
apiVersion: kubescheduler.config.k8s.io/v1
kind: KubeSchedulerConfiguration
profiles:
- schedulerName: default-scheduler
  pluginConfig:
  # 1. 装箱策略：优先填满一个节点
  - name: NodeResourcesFit
    args:
      scoringStrategy:
        type: MostAllocated
        resources:
        - name: nvidia.com/gpu
          weight: 10
        - name: cpu
          weight: 1
        - name: memory
          weight: 1
  plugins:
    # 2. 如果需要 Gang Scheduling，启用 Coscheduling
    permit:
      enabled:
      - name: Coscheduling
    # 3. 队列排序按优先级
    queueSort:
      enabled:
      - name: Coscheduling
```

```yaml
# PriorityClass 建议
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: training-critical
value: 1000000
globalDefault: false
description: "分布式训练任务 - 不可中断"
---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: training-normal
value: 100000
description: "普通训练任务"
---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: inference-serving
value: 500000
description: "推理服务 - 高优先级但可被 critical 训练抢占"
---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: dev-experiment
value: 10000
preemptionPolicy: Never  # 实验任务不抢占
description: "开发实验任务"
```

## 小结

| Slurm 概念 | K8s 对应 | 备注 |
|------------|---------|------|
| `squeue` | `kubectl get pods --field-selector=status.phase=Pending` | 查看待调度任务 |
| `scontrol show job` | `kubectl describe pod` (Events 部分) | 查看调度详情 |
| `sinfo` | `kubectl describe node` | 查看节点资源 |
| Partition | Namespace + NodeSelector/Affinity | 资源分区 |
| QOS | PriorityClass | 优先级 |
| GRES | Extended Resource (nvidia.com/gpu) | GPU 资源声明 |
| Backfill | 默认就是按优先级+FIFO | K8s 无原生 backfill |
| Job Array | Job (parallelism/completions) | 批量任务 |
