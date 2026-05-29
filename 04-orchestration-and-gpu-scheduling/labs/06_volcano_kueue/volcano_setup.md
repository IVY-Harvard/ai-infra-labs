# Volcano 安装与配置

## 1. 安装 Volcano

```bash
# 方式 1：使用 Helm
helm repo add volcano-sh https://volcano-sh.github.io/helm-charts
helm repo update

helm install volcano volcano-sh/volcano \
  --namespace volcano-system \
  --create-namespace \
  --set basic.image_tag_version=v1.9.0

# 方式 2：使用 kubectl
kubectl apply -f https://raw.githubusercontent.com/volcano-sh/volcano/v1.9.0/installer/volcano-development.yaml

# 验证安装
kubectl get pods -n volcano-system
# NAME                                        READY   STATUS
# volcano-admission-xxx                       1/1     Running
# volcano-controllers-xxx                     1/1     Running
# volcano-scheduler-xxx                       1/1     Running
```

## 2. Volcano 架构

```
┌─────────────────────────────────────────────────────┐
│                 Volcano 架构                          │
│                                                      │
│  ┌────────────────┐  ┌──────────────────────────┐   │
│  │  volcano-      │  │  volcano-scheduler       │   │
│  │  controllers   │  │                          │   │
│  │                │  │  调度插件：               │   │
│  │  管理 VCJob    │  │  - gang: Gang调度        │   │
│  │  生命周期      │  │  - binpack: GPU集中      │   │
│  │                │  │  - proportion: 队列配额   │   │
│  │                │  │  - drf: 公平共享         │   │
│  │                │  │  - predicates: 资源检查   │   │
│  └────────────────┘  └──────────────────────────┘   │
│                                                      │
│  ┌────────────────┐                                  │
│  │  volcano-      │  CRDs:                          │
│  │  admission     │  - vcjob.batch.volcano.sh       │
│  │                │  - queue.scheduling.volcano.sh   │
│  │  Webhook验证   │  - podgroup.scheduling.volcano.sh│
│  └────────────────┘                                  │
└─────────────────────────────────────────────────────┘
```

## 3. 配置队列

```yaml
# 创建 GPU 训练队列
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: training-queue
spec:
  reclaimable: true
  weight: 4           # 队列权重（决定公平共享比例）
  capability:         # 队列资源上限
    cpu: "64"
    memory: "256Gi"
    nvidia.com/gpu: "8"
---
# 推理队列
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: inference-queue
spec:
  reclaimable: true
  weight: 2
  capability:
    cpu: "32"
    memory: "128Gi"
    nvidia.com/gpu: "4"
---
# 实验队列（可被抢占）
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: experiment-queue
spec:
  reclaimable: true   # 资源可被回收
  weight: 1
  capability:
    cpu: "32"
    memory: "128Gi"
    nvidia.com/gpu: "4"
```

## 4. 调度器配置

```yaml
# volcano-scheduler-configmap
apiVersion: v1
kind: ConfigMap
metadata:
  name: volcano-scheduler-configmap
  namespace: volcano-system
data:
  volcano-scheduler.conf: |
    actions: "enqueue, allocate, backfill"
    tiers:
    - plugins:
      - name: priority       # 按优先级排序
      - name: gang           # Gang Scheduling
        enablePreemption: true
      - name: conformance
    - plugins:
      - name: overcommit     # 资源超卖
      - name: drf            # Dominant Resource Fairness
      - name: predicates     # 资源满足性检查
      - name: proportion     # 队列配额
      - name: nodeorder      # 节点排序
      - name: binpack        # GPU Bin Packing
        arguments:
          binpack.weight: 5
          binpack.cpu: 1
          binpack.memory: 1
          binpack.resources: nvidia.com/gpu
          binpack.resources.nvidia.com/gpu: 10  # GPU 权重最高
```

## 5. 提交 VCJob

```yaml
# 分布式 PyTorch 训练 VCJob
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: pytorch-training
spec:
  schedulerName: volcano
  queue: training-queue
  minAvailable: 4      # Gang Scheduling: 最少需要 4 个 Pod
  maxRetry: 3
  plugins:
    env: []            # 自动注入环境变量
    svc: []            # 自动创建 headless service
  tasks:
  - replicas: 4
    name: worker
    template:
      spec:
        containers:
        - name: trainer
          image: pytorch-training:latest
          command:
          - torchrun
          - --nproc_per_node=2
          - --nnodes=4
          - --node_rank=$(VK_TASK_INDEX)
          - --master_addr=$(worker-0.pytorch-training)
          - --master_port=29400
          - train.py
          resources:
            limits:
              nvidia.com/gpu: 2
              cpu: "8"
              memory: "64Gi"
        restartPolicy: OnFailure
```
