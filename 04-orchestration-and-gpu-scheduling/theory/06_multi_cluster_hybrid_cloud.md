# 06 - 多集群与混合云 GPU 调度

## 引言：为什么需要多集群

单一 K8s 集群有物理和逻辑上限：

```
物理限制：
  - 单集群推荐 ≤5000 节点（API Server 性能瓶颈）
  - 网络分区风险（跨机房/跨区域）
  
逻辑限制：
  - 不同团队要求不同的 K8s 版本/安全策略
  - 合规要求（数据不能出特定区域）

GPU 特有的多集群需求：
  - 不同区域 GPU 型号不同（私有集群 H20，云上 A100/H100）
  - Spot/Preemptible GPU 实例（便宜但随时可能被回收）
  - 训练需要大量 GPU（私有集群不够时 burst 到云上）
  - 推理部署在靠近用户的区域（边缘/多区域）
```

## 1. 多集群联邦

### 1.1 KubeFed (Kubernetes Federation v2)

```
┌─────────────────────────────────────────────────────┐
│                 Federation Control Plane              │
│                                                      │
│  ┌──────────────┐  ┌─────────────────────────┐     │
│  │ KubeFed      │  │  FederatedDeployment    │     │
│  │ Controller   │  │  FederatedService       │     │
│  │              │  │  FederatedConfigMap      │     │
│  └──────┬───────┘  │  ReplicaSchedulingPref. │     │
│         │          └─────────────────────────┘     │
│         │                                           │
│    ┌────┼──────────────────────┐                   │
│    │    │                      │                   │
│    ▼    ▼                      ▼                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │Cluster A │  │Cluster B │  │Cluster C │        │
│  │(Private) │  │(AWS)     │  │(GCP)     │        │
│  │8xH20     │  │A100 Spot │  │H100      │        │
│  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────────────────────────────────────┘
```

```yaml
# FederatedDeployment：跨集群分发部署
apiVersion: types.kubefed.io/v1beta1
kind: FederatedDeployment
metadata:
  name: inference-service
  namespace: production
spec:
  template:
    spec:
      replicas: 6
      template:
        spec:
          containers:
          - name: model-server
            image: my-inference:v1
            resources:
              limits:
                nvidia.com/gpu: 1
  placement:
    clusters:
    - name: cluster-private   # 私有集群
    - name: cluster-aws       # AWS
    - name: cluster-gcp       # GCP
  overrides:
  - clusterName: cluster-private
    clusterOverrides:
    - path: "/spec/replicas"
      value: 3                # 私有集群 3 副本
  - clusterName: cluster-aws
    clusterOverrides:
    - path: "/spec/replicas"
      value: 2                # AWS 2 副本
  - clusterName: cluster-gcp
    clusterOverrides:
    - path: "/spec/replicas"
      value: 1                # GCP 1 副本
```

### 1.2 Liqo — 虚拟节点方式

```
Liqo 的独特方法：将远端集群映射为本地的 "虚拟节点"

本地集群视角：
┌────────────────────────────────────────────┐
│  kubectl get nodes                          │
│                                            │
│  NAME           STATUS   ROLES    GPU      │
│  node-1         Ready    worker   8xH20    │
│  node-2         Ready    worker   8xH20    │
│  liqo-aws       Ready    agent    16xA100  │ ← 虚拟节点（实际是 AWS 集群）
│  liqo-gcp       Ready    agent    8xH100   │ ← 虚拟节点（实际是 GCP 集群）
└────────────────────────────────────────────┘

好处：
  - 使用标准 K8s API，不需要学习新 API
  - NodeSelector/Affinity 天然支持
  - 调度器直接感知远端资源
```

```yaml
# 使用 Liqo 将训练 burst 到云上
apiVersion: batch/v1
kind: Job
metadata:
  name: large-training
spec:
  template:
    spec:
      # 优先使用本地 GPU
      affinity:
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            preference:
              matchExpressions:
              - key: liqo.io/type
                operator: NotIn
                values: ["virtual-node"]
          # 如果本地不够，允许调度到虚拟节点（云上）
          - weight: 50
            preference:
              matchExpressions:
              - key: liqo.io/type
                operator: In
                values: ["virtual-node"]
      containers:
      - name: trainer
        resources:
          limits:
            nvidia.com/gpu: 8
```

### 1.3 MultiKueue — Kueue 多集群扩展

```yaml
# MultiKueue：Kueue 原生的多集群方案
# 在管理集群上提交 Workload，自动分发到执行集群

# 管理集群配置
apiVersion: kueue.x-k8s.io/v1beta1
kind: AdmissionCheck
metadata:
  name: multi-cluster
spec:
  controllerName: kueue.x-k8s.io/multikueue
  parameters:
    apiGroup: kueue.x-k8s.io
    kind: MultiKueueConfig
    name: multi-cluster-config
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: MultiKueueConfig
metadata:
  name: multi-cluster-config
spec:
  clusters:
  - name: private-cluster
    kubeConfig:
      locationType: Secret
      location: private-cluster-kubeconfig
  - name: cloud-cluster
    kubeConfig:
      locationType: Secret
      location: cloud-cluster-kubeconfig
---
# ClusterQueue 关联 AdmissionCheck
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: multi-cluster-queue
spec:
  admissionChecks:
  - multi-cluster
  resourceGroups:
  - coveredResources: ["nvidia.com/gpu"]
    flavors:
    - name: any-gpu
      resources:
      - name: "nvidia.com/gpu"
        nominalQuota: 64
```

## 2. 混合云 GPU 调度

### 2.1 架构设计

```
┌───────────────────────────────────────────────────────┐
│                  Hybrid Cloud GPU Scheduler             │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │              Scheduling Decision Engine            │ │
│  │                                                   │ │
│  │  输入：                                            │ │
│  │  - Job 资源需求（GPU 类型/数量/时长）              │ │
│  │  - 成本模型（Spot 价格/On-Demand 价格/私有成本）  │ │
│  │  - SLA 要求（延迟/可用性/数据合规）               │ │
│  │  - 集群实时容量                                   │ │
│  │                                                   │ │
│  │  输出：                                            │ │
│  │  - 目标集群                                       │ │
│  │  - 实例类型（Spot/On-Demand/Reserved）            │ │
│  │  - Checkpoint 策略                                │ │
│  └──────────────────────────────────────────────────┘ │
│                        │                               │
│          ┌─────────────┼─────────────┐                │
│          ▼             ▼             ▼                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ Private  │  │ AWS Spot │  │ GCP      │           │
│  │ Cluster  │  │ Cluster  │  │ On-Demand│           │
│  │ 8xH20   │  │ p4d.24xl │  │ a3-mega  │           │
│  │          │  │ (A100x8) │  │ (H100x8) │           │
│  └──────────┘  └──────────┘  └──────────┘           │
└───────────────────────────────────────────────────────┘
```

### 2.2 成本优化策略

```python
# 混合云成本模型
class CostModel:
    """GPU 使用成本计算"""

    # 成本对比（每 GPU 小时，示例价格）
    PRICING = {
        "private_h20": {
            "hourly": 0.0,            # 固定成本已摊销
            "amortized_hourly": 2.50, # 硬件摊销+电费+运维
        },
        "aws_a100_spot": {
            "hourly": 3.50,  # Spot 价格（波动大）
            "spot_discount": 0.6,  # 比 On-Demand 便宜 60%
            "interruption_rate": 0.05,  # 每小时 5% 中断概率
        },
        "aws_a100_ondemand": {
            "hourly": 8.50,
        },
        "gcp_h100_spot": {
            "hourly": 5.00,
            "spot_discount": 0.7,
            "interruption_rate": 0.03,
        },
    }

    def calculate_total_cost(self, job):
        """
        计算 Job 在不同选项上的总成本
        包括：计算成本 + 中断恢复成本 + 数据传输成本
        """
        options = []
        for provider, pricing in self.PRICING.items():
            compute_cost = pricing["hourly"] * job.gpu_count * job.estimated_hours

            # Spot 中断恢复成本
            if "interruption_rate" in pricing:
                expected_interruptions = (
                    pricing["interruption_rate"] * job.estimated_hours
                )
                recovery_cost = (
                    expected_interruptions
                    * job.checkpoint_overhead_hours
                    * pricing["hourly"]
                    * job.gpu_count
                )
                compute_cost += recovery_cost

            # 数据传输成本（跨云）
            if "aws" in provider or "gcp" in provider:
                egress_cost = job.data_size_gb * 0.09  # $0.09/GB
                compute_cost += egress_cost

            options.append((provider, compute_cost))

        return sorted(options, key=lambda x: x[1])
```

### 2.3 Spot 实例管理

```
Spot 实例的挑战：
  - 价格波动：可能突然涨价导致回收
  - 2 分钟警告：收到中断通知后只有 2 分钟
  - 容量不保证：高峰期可能完全无法获取

应对策略：
  1. 多可用区/多实例类型 → 分散风险
  2. 频繁 Checkpoint → 减少中断损失
  3. 弹性训练 → 缩减 worker 而非停止训练
  4. 混合 Spot + On-Demand → 关键路径用 On-Demand
```

```yaml
# AWS Karpenter 配置 Spot + On-Demand 混合
apiVersion: karpenter.sh/v1beta1
kind: NodePool
metadata:
  name: gpu-training-spot
spec:
  template:
    spec:
      requirements:
      - key: karpenter.sh/capacity-type
        operator: In
        values: ["spot"]
      - key: node.kubernetes.io/instance-type
        operator: In
        values: ["p4d.24xlarge", "p4de.24xlarge", "p5.48xlarge"]
      - key: topology.kubernetes.io/zone
        operator: In
        values: ["us-east-1a", "us-east-1b", "us-east-1c"]
  disruption:
    consolidationPolicy: WhenEmpty
    expireAfter: 720h
  limits:
    nvidia.com/gpu: "64"  # 最多 64 GPU
---
apiVersion: karpenter.sh/v1beta1
kind: NodePool
metadata:
  name: gpu-training-ondemand
spec:
  template:
    spec:
      requirements:
      - key: karpenter.sh/capacity-type
        operator: In
        values: ["on-demand"]
      - key: node.kubernetes.io/instance-type
        operator: In
        values: ["p4d.24xlarge"]
  limits:
    nvidia.com/gpu: "16"  # On-Demand 限制较少
```

## 3. 跨集群训练任务编排

### 3.1 场景：私有集群 GPU 不够，burst 到云上

```
训练需求：32 GPU (4 节点 x 8 GPU)
私有集群：只有 8xH20（1 节点空闲）
决策：私有集群 8 GPU + 云上 24 GPU

挑战：
  - 跨集群网络延迟高（1-10ms vs 节点内 <0.1ms）
  - 数据传输带宽有限（~10Gbps vs 节点内 NVLink 900GB/s）
  - 需要一致的运行环境和配置

应对：
  - Pipeline 并行而非 Data 并行（减少通信量）
  - 梯度压缩
  - 异步训练
  - 或者：只在云上训练，数据通过 S3 同步
```

### 3.2 跨集群 Job 编排

```yaml
# 使用 Argo Workflows 编排跨集群训练
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  name: hybrid-training-pipeline
spec:
  templates:
  - name: main
    steps:
    # Step 1: 数据准备（在有数据的集群）
    - - name: prepare-data
        template: data-prep
        arguments:
          parameters:
          - name: cluster
            value: private

    # Step 2: 训练（选择最优集群）
    - - name: train
        template: training
        arguments:
          parameters:
          - name: cluster
            value: "{{steps.select-cluster.outputs.result}}"

    # Step 3: 评估（在私有集群）
    - - name: evaluate
        template: evaluation
        arguments:
          parameters:
          - name: cluster
            value: private

  - name: training
    # 根据 cluster 参数决定在哪里执行
    resource:
      action: create
      manifest: |
        apiVersion: ray.io/v1
        kind: RayJob
        metadata:
          name: distributed-training
          namespace: training
        spec:
          entrypoint: "python train.py --num-gpus 32"
          rayClusterSpec:
            workerGroupSpecs:
            - replicas: 4
              template:
                spec:
                  containers:
                  - name: worker
                    resources:
                      limits:
                        nvidia.com/gpu: 8
```

## 4. 网络连通性

### 4.1 跨集群网络方案

```
方案选择取决于安全和性能需求：

1. VPN (WireGuard/IPSec)
   私有集群 ←── VPN 隧道 ──→ 云集群
   延迟：+1-5ms
   带宽：受限于公网/专线
   适用：常规跨集群通信

2. 云专线 (AWS Direct Connect / GCP Interconnect)
   私有集群 ←── 专线 ──→ 云集群
   延迟：<5ms（同城）
   带宽：10-100Gbps
   适用：大规模数据传输、对延迟敏感的训练

3. Submariner (K8s 多集群网络)
   跨集群 Pod 网络直接互通
   支持 Service Discovery
   适用：需要跨集群 Pod-to-Pod 通信
```

### 4.2 数据同步策略

```
策略 1: 共享存储（推荐）
  所有集群挂载同一 S3/GCS/MinIO
  训练数据: s3://training-data/
  Checkpoint: s3://checkpoints/
  好处: 不需要显式同步
  
策略 2: 数据本地化
  训练前将数据复制到执行集群
  适用: 数据量大，训练时间长（复制成本可摊销）
  
策略 3: 流式读取
  训练过程中实时从远端读取
  适用: 数据量极大，无法全量复制
  需要高带宽、低延迟网络
```

## 5. 实际部署建议

### 5.1 你的 8xH20 集群的多集群策略

```
基本架构：
  私有集群（主集群）: 8xH20，常驻训练和推理
  云 burst 集群: 按需创建，用完销毁

建议：
  1. 大部分工作负载在私有集群完成
  2. 排队等待 > 2 小时的训练 Job → burst 到 Spot
  3. 使用 S3/MinIO 作为共享存储
  4. 用 Kueue + MultiKueue 管理多集群队列
  5. 训练 Checkpoint 频率：每 30 分钟（Spot 中断保护）

成本控制：
  - 设置 Spot 预算上限
  - 训练完立即销毁云资源
  - 监控 Spot 中断率，选择稳定的实例类型和区域
```

## 小结

```
多集群方案选择：
  小规模（你的场景）→ Liqo 或 MultiKueue
  中规模 → KubeFed + 自定义调度
  大规模 → 商业方案（Rancher Fleet / OpenShift ACM）

混合云 GPU 关键：
  1. 明确哪些工作负载适合跑在云上（成本 vs 便利）
  2. Spot 实例 + 频繁 Checkpoint = 最佳性价比
  3. 数据和网络是跨集群最大瓶颈
  4. 统一监控和日志（跨集群可观测性）
```
