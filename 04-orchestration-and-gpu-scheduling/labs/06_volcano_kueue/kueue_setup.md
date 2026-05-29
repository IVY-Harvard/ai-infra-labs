# Kueue 安装与配置

## 1. 安装 Kueue

```bash
# 安装最新稳定版 Kueue
kubectl apply --server-side -f \
  https://github.com/kubernetes-sigs/kueue/releases/download/v0.8.0/manifests.yaml

# 验证安装
kubectl get pods -n kueue-system
# NAME                                      READY   STATUS
# kueue-controller-manager-xxx              1/1     Running

# 查看 CRD
kubectl get crd | grep kueue
# clusterqueues.kueue.x-k8s.io
# localqueues.kueue.x-k8s.io
# resourceflavors.kueue.x-k8s.io
# workloads.kueue.x-k8s.io
```

## 2. Kueue 核心概念

```
┌────────────────────────────────────────────────────────┐
│                     Kueue 架构                          │
│                                                         │
│  ResourceFlavor (资源风味)                              │
│  └── 定义物理资源类型（如 "h20-gpu", "a100-gpu"）      │
│                                                         │
│  ClusterQueue (集群队列)                                │
│  └── 定义资源配额、借用策略                             │
│      ├── training-cq: 4 GPU 配额，可借用到 8            │
│      ├── inference-cq: 2 GPU 配额                      │
│      └── experiment-cq: 2 GPU 配额，可被抢占            │
│                                                         │
│  LocalQueue (命名空间队列)                              │
│  └── 绑定 namespace 到 ClusterQueue                    │
│      └── team-a-ns → training-cq                       │
│                                                         │
│  Workload (工作负载)                                    │
│  └── 对应一个 Job/PyTorchJob，排队等待准入              │
│      → Kueue 判断配额是否满足                          │
│      → 满足 → admit → Job 开始调度                     │
│      → 不满足 → 排队等待                               │
└────────────────────────────────────────────────────────┘
```

## 3. 配置 8x H20 集群的 Kueue

```yaml
# 步骤 1：定义 ResourceFlavor
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
# CPU 节点 flavor（用于非 GPU 任务）
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: default-cpu
spec: {}
```

```yaml
# 步骤 2：定义 ClusterQueue（3 个团队共享 8 GPU）
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: training-cq
spec:
  cohort: gpu-cluster       # 同一 cohort 内的队列可以互相借用
  queueingStrategy: BestEffortFIFO
  preemption:
    reclaimWithinCohort: Any
    withinClusterQueue: LowerPriority
  resourceGroups:
  - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
    flavors:
    - name: h20-gpu
      resources:
      - name: nvidia.com/gpu
        nominalQuota: 4         # 基础配额：4 GPU
        borrowingLimit: 4       # 最多可借用额外 4 GPU（共 8）
      - name: cpu
        nominalQuota: 32
      - name: memory
        nominalQuota: 256Gi
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: inference-cq
spec:
  cohort: gpu-cluster
  queueingStrategy: BestEffortFIFO
  preemption:
    reclaimWithinCohort: LowerPriority
  resourceGroups:
  - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
    flavors:
    - name: h20-gpu
      resources:
      - name: nvidia.com/gpu
        nominalQuota: 2
        borrowingLimit: 2
      - name: cpu
        nominalQuota: 16
      - name: memory
        nominalQuota: 128Gi
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: experiment-cq
spec:
  cohort: gpu-cluster
  queueingStrategy: BestEffortFIFO
  preemption:
    reclaimWithinCohort: Any
    withinClusterQueue: LowerPriority
  resourceGroups:
  - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
    flavors:
    - name: h20-gpu
      resources:
      - name: nvidia.com/gpu
        nominalQuota: 2
        borrowingLimit: 6       # 空闲时可以借很多
      - name: cpu
        nominalQuota: 16
      - name: memory
        nominalQuota: 128Gi
```

```yaml
# 步骤 3：定义 LocalQueue（绑定到 namespace）
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: training-queue
  namespace: team-training
spec:
  clusterQueue: training-cq
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: inference-queue
  namespace: team-inference
spec:
  clusterQueue: inference-cq
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: experiment-queue
  namespace: team-experiment
spec:
  clusterQueue: experiment-cq
```

## 4. 验证配置

```bash
# 查看 ClusterQueue 状态
kubectl get clusterqueue
# NAME            COHORT       PENDING   ADMITTED
# training-cq    gpu-cluster  0         2
# inference-cq   gpu-cluster  0         1
# experiment-cq  gpu-cluster  1         1

# 查看 LocalQueue
kubectl get localqueue -A
# NAMESPACE         NAME              CLUSTER-QUEUE   PENDING
# team-training     training-queue    training-cq     0
# team-inference    inference-queue   inference-cq    0
# team-experiment   experiment-queue  experiment-cq   1

# 查看 Workload 排队情况
kubectl get workloads -A
```

## 5. 资源借用示例

```
正常工作时间（所有队列都在使用）：
  training-cq:    使用 4/4 GPU（满配额）
  inference-cq:   使用 2/2 GPU（满配额）
  experiment-cq:  使用 2/2 GPU（满配额）
  总计：8/8 GPU

夜间（推理和实验队列空闲）：
  training-cq:    使用 8/4 GPU（借用了 4 GPU）
  inference-cq:   使用 0/2 GPU
  experiment-cq:  使用 0/2 GPU
  总计：8/8 GPU，training 充分利用

白天推理队列回来了：
  → Kueue 检测到 inference-cq 有新 workload
  → training-cq 超出 nominalQuota 的部分被标记为可回收
  → preemption 策略触发，回收借用的 GPU
  → inference-cq 获得其配额内的 GPU
```
