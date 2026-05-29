# 05 - Ray 架构与 KubeRay

## 引言：Ray 在 AI 基础设施中的角色

本章在 KubeRay 部署 RayCluster 的基础上，深入 Ray 的内部架构，理解它为什么适合
AI 工作负载，以及 KubeRay Operator 如何将 Ray 的灵活性与 K8s 的编排能力结合。

Ray 解决的核心问题：**让分布式计算像写单机代码一样简单。**

```python
# 单机版本
def train(config):
    model = build_model(config)
    for epoch in range(100):
        model.train()
    return model.evaluate()

# Ray 分布式版本（几乎不改代码）
@ray.remote(num_gpus=1)
def train(config):
    model = build_model(config)
    for epoch in range(100):
        model.train()
    return model.evaluate()

# 启动 8 个并行训练（自动分布到集群）
futures = [train.remote(c) for c in configs]
results = ray.get(futures)
```

## 1. Ray 核心架构

### 1.1 整体架构

```
┌──────────────────────────────────────────────────────────────┐
│                     Ray Cluster                               │
│                                                              │
│  ┌─────────────────────────────────────┐                    │
│  │           Head Node                  │                    │
│  │                                     │                    │
│  │  ┌─────────┐  ┌──────────────────┐ │                    │
│  │  │   GCS   │  │    Autoscaler    │ │                    │
│  │  │ (Global │  │                  │ │                    │
│  │  │ Control │  │  Ray Dashboard   │ │                    │
│  │  │ Store)  │  │  (port 8265)     │ │                    │
│  │  └─────────┘  └──────────────────┘ │                    │
│  │  ┌─────────┐  ┌──────────────────┐ │                    │
│  │  │ Raylet  │  │  Object Store    │ │                    │
│  │  │         │  │  (Plasma/共享内存)│ │                    │
│  │  └─────────┘  └──────────────────┘ │                    │
│  └─────────────────────────────────────┘                    │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │   Worker Node 1   │  │   Worker Node 2   │                │
│  │                   │  │                   │                │
│  │  ┌─────┐ ┌─────┐│  │  ┌─────┐ ┌─────┐│                │
│  │  │Raylet│ │Obj  ││  │  │Raylet│ │Obj  ││                │
│  │  │      │ │Store││  │  │      │ │Store││                │
│  │  └─────┘ └─────┘│  │  └─────┘ └─────┘│                │
│  │  ┌──┐┌──┐┌──┐  │  │  ┌──┐┌──┐┌──┐  │                │
│  │  │W1││W2││W3│  │  │  │W1││W2││W3│  │                │
│  │  └──┘└──┘└──┘  │  │  └──┘└──┘└──┘  │                │
│  └──────────────────┘  └──────────────────┘                │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 核心组件详解

**GCS (Global Control Store)**

```
职责：
  - 存储集群元数据（Actor 位置、资源状态）
  - 管理 Actor 生命周期
  - 存储任务规范（函数字节码）
  - 节点管理和故障检测

实现：
  - 基于 Redis（早期）→ 内置存储（Ray 2.0+）
  - 运行在 Head Node 上
  - 是集群的单点（SPOF），GCS 故障 = 集群故障
  - Ray 2.0+ 支持 GCS 容错（HA）
```

**Raylet（每个节点一个）**

```
职责：
  - 本地资源管理（CPU/GPU/Memory）
  - 任务调度（决定 task 在哪个 worker 执行）
  - Object Store 管理
  - Worker 进程管理（启动/杀死）

调度策略：
  - 数据本地性优先：task 调度到数据所在节点
  - 资源匹配：GPU task 调度到有 GPU 的节点
  - 负载均衡：避免热节点
  - 溢出调度：本地资源不够时发到其他节点
```

**Object Store（Plasma，每个节点一个）**

```
职责：
  - 存储 Ray 对象（task 返回值、ray.put() 的数据）
  - 基于共享内存（/dev/shm），零拷贝读取
  - 跨节点对象传输（通过 gRPC）

关键特性：
  - 不可变对象：写入后不能修改
  - 引用计数：无引用时自动回收
  - LRU 淘汰：内存不足时淘汰最久未用的对象
  - 溢出到磁盘：对象太大时溢出到 NVMe

GPU 场景注意：
  - Object Store 使用 CPU 内存（非 GPU 显存）
  - GPU tensor 需要先 copy 到 CPU → Object Store → 远端 CPU → GPU
  - 大 tensor 传输走 Object Store，小 tensor 直接 gRPC inline
```

### 1.3 Task 和 Actor 执行模型

```python
# Task：无状态函数调用
@ray.remote(num_gpus=1)
def preprocess(data):
    # 在一个 worker 上执行，执行完 worker 可复用
    return transformed_data

# Actor：有状态对象
@ray.remote(num_gpus=1)
class ModelServer:
    def __init__(self, model_path):
        self.model = load_model(model_path)  # 模型常驻 GPU
    
    def predict(self, input):
        return self.model(input)

# Actor 生命周期 = 对象引用存在期间
server = ModelServer.remote("/models/llm")
# server 占据 1 GPU 直到被 del 或程序退出
```

### 1.4 Ray 的资源模型

```python
# Ray 的资源是逻辑资源，由用户声明
ray.init(resources={
    "GPU": 8,
    "CPU": 64,
    "memory": 256 * 1024 * 1024 * 1024,
    # 自定义资源
    "accelerator_type:H20": 1,
    "NVLink_group_0": 4,  # 用自定义资源表达拓扑
})

# Task 声明资源需求
@ray.remote(
    num_gpus=2,
    num_cpus=8,
    memory=32 * 1024 * 1024 * 1024,
    resources={"accelerator_type:H20": 0.001}  # 约束 GPU 类型
)
def train_step():
    pass
```

## 2. Ray AI 库生态

### 2.1 Ray Train（分布式训练）

```python
import ray.train
from ray.train.torch import TorchTrainer

def train_func(config):
    # 自动设置 DDP
    model = ray.train.torch.prepare_model(model)
    dataloader = ray.train.torch.prepare_data_loader(dataloader)
    
    for epoch in range(config["epochs"]):
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
        
        # 自动同步 metrics
        ray.train.report({"loss": loss.item()})
        
        # Checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(model.state_dict(), f"{tmpdir}/model.pt")
            ray.train.report(
                {"loss": loss.item()},
                checkpoint=ray.train.Checkpoint.from_directory(tmpdir)
            )

trainer = TorchTrainer(
    train_func,
    train_loop_config={"epochs": 100},
    scaling_config=ray.train.ScalingConfig(
        num_workers=8,           # 8 个 worker
        use_gpu=True,            # 每个 worker 1 GPU
        resources_per_worker={"GPU": 1, "CPU": 8},
    ),
    run_config=ray.train.RunConfig(
        storage_path="s3://my-bucket/checkpoints",
        failure_config=ray.train.FailureConfig(max_failures=3),
    ),
)
result = trainer.fit()
```

### 2.2 Ray Serve（在线推理）

```python
from ray import serve
import ray

@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_gpus": 1},
    max_ongoing_requests=100,
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 8,
        "target_ongoing_requests": 10,
    },
)
class LLMDeployment:
    def __init__(self):
        from vllm import LLM
        self.llm = LLM(model="meta-llama/Llama-3-8B", gpu_memory_utilization=0.9)
    
    async def __call__(self, request):
        prompt = request.query_params["prompt"]
        outputs = self.llm.generate([prompt])
        return outputs[0].outputs[0].text

app = LLMDeployment.bind()
serve.run(app, host="0.0.0.0", port=8000)
```

### 2.3 Ray Data（数据处理）

```python
import ray.data

# 分布式数据加载和预处理
ds = ray.data.read_parquet("s3://bucket/training-data/")
ds = ds.map_batches(
    preprocess_fn,
    batch_size=1024,
    num_gpus=0.5,  # 使用半个 GPU 做数据预处理
)
# 直接喂给 Ray Train
trainer = TorchTrainer(
    train_func,
    datasets={"train": ds},
    # ...
)
```

## 3. KubeRay Operator

### 3.1 架构

```
┌──────────────────────────────────────────────────────────────┐
│                    KubeRay Operator                           │
│                                                              │
│  ┌──────────────────────────────────────────────────┐       │
│  │            Controller Manager                      │       │
│  │                                                   │       │
│  │  ┌─────────────┐ ┌──────────┐ ┌──────────────┐  │       │
│  │  │ RayCluster  │ │  RayJob  │ │  RayService  │  │       │
│  │  │ Controller  │ │Controller│ │  Controller   │  │       │
│  │  └──────┬──────┘ └────┬─────┘ └──────┬───────┘  │       │
│  └─────────┼─────────────┼──────────────┼───────────┘       │
│            │             │              │                     │
│            ▼             ▼              ▼                     │
│  ┌──────────────────────────────────────────────────┐       │
│  │               K8s API Server                       │       │
│  │                                                   │       │
│  │  Pods, Services, Ingress, ConfigMaps              │       │
│  └──────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 三种 CR 详解

**RayCluster — 长期运行的 Ray 集群**

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: gpu-cluster
spec:
  rayVersion: '2.9.0'
  
  # Head Node
  headGroupSpec:
    rayStartParams:
      dashboard-host: '0.0.0.0'
      num-cpus: '0'  # Head 不跑计算任务
    template:
      spec:
        containers:
        - name: ray-head
          image: rayproject/ray-ml:2.9.0-py310-gpu
          resources:
            limits:
              cpu: "8"
              memory: "32Gi"
              # Head 通常不需要 GPU
          ports:
          - containerPort: 6379   # GCS
          - containerPort: 8265   # Dashboard
          - containerPort: 10001  # Client
          volumeMounts:
          - name: shared-data
            mountPath: /data
  
  # Worker Groups（可以有多个不同配置的组）
  workerGroupSpecs:
  - replicas: 4
    minReplicas: 2
    maxReplicas: 8
    groupName: gpu-workers
    rayStartParams:
      num-gpus: '8'
    template:
      spec:
        containers:
        - name: ray-worker
          image: rayproject/ray-ml:2.9.0-py310-gpu
          resources:
            limits:
              cpu: "64"
              memory: "256Gi"
              nvidia.com/gpu: 8
          volumeMounts:
          - name: shared-data
            mountPath: /data
          - name: dshm
            mountPath: /dev/shm
        volumes:
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 64Gi  # Object Store 使用共享内存
        
        tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
        
        nodeSelector:
          nvidia.com/gpu.product: "NVIDIA-H20"
  
  # 第二个 Worker Group（如有不同硬件）
  - replicas: 0
    minReplicas: 0
    maxReplicas: 4
    groupName: cpu-workers
    rayStartParams: {}
    template:
      spec:
        containers:
        - name: ray-worker
          image: rayproject/ray-ml:2.9.0-py310
          resources:
            limits:
              cpu: "32"
              memory: "128Gi"
```

**RayJob — 提交一次性 Job 到 Ray 集群**

```yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: training-job
  labels:
    kueue.x-k8s.io/queue-name: team-a-queue  # 与 Kueue 集成
spec:
  # 提交到已有集群 或 自动创建临时集群
  # clusterSelector:
  #   ray.io/cluster: gpu-cluster  # 使用已有集群
  
  # 或者 inline 定义集群（Job 结束后自动删除）
  rayClusterSpec:
    headGroupSpec:
      rayStartParams:
        num-cpus: '0'
      template:
        spec:
          containers:
          - name: ray-head
            image: my-training:v1
            resources:
              limits:
                cpu: "4"
                memory: "16Gi"
    workerGroupSpecs:
    - replicas: 4
      groupName: workers
      rayStartParams:
        num-gpus: '8'
      template:
        spec:
          containers:
          - name: ray-worker
            image: my-training:v1
            resources:
              limits:
                cpu: "64"
                memory: "256Gi"
                nvidia.com/gpu: 8

  # Job 入口
  entrypoint: "python /app/train.py --num-workers 4 --gpus-per-worker 8"
  runtimeEnvYAML: |
    working_dir: /app
    pip:
      - torch==2.2.0
      - deepspeed==0.13.0
  
  # 生命周期
  shutdownAfterJobFinishes: true   # Job 完成后清理集群
  ttlSecondsAfterFinished: 300     # 保留 5 分钟供查看日志
  submitterPodTemplate:
    spec:
      restartPolicy: Never
```

**RayService — 长期运行的推理服务**

```yaml
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: llm-service
spec:
  serveConfigV2: |
    applications:
    - name: llm
      import_path: serve_app:app
      route_prefix: /
      runtime_env:
        working_dir: "https://github.com/my-org/serve-app/archive/main.zip"
      deployments:
      - name: LLMDeployment
        num_replicas: 2
        ray_actor_options:
          num_gpus: 1
        autoscaling_config:
          min_replicas: 1
          max_replicas: 4
          target_ongoing_requests: 5
  
  rayClusterConfig:
    headGroupSpec:
      template:
        spec:
          containers:
          - name: ray-head
            image: rayproject/ray-ml:2.9.0-py310-gpu
            resources:
              limits:
                cpu: "8"
                memory: "32Gi"
    workerGroupSpecs:
    - replicas: 2
      maxReplicas: 4
      groupName: gpu-workers
      template:
        spec:
          containers:
          - name: ray-worker
            resources:
              limits:
                nvidia.com/gpu: 1
```

### 3.3 KubeRay Autoscaler 集成

```
Ray Autoscaler ←→ KubeRay Operator 协作：

1. Ray 任务需要更多资源
   │
2. Ray Autoscaler 检测到 pending tasks
   │
3. Ray Autoscaler 计算需要多少新 worker
   │
4. Ray Autoscaler 请求 KubeRay Operator 扩容
   │
5. KubeRay Operator 增加 worker Pod replicas
   │
6. K8s 调度 Pod（可能触发 Cluster Autoscaler 加节点）
   │
7. 新 worker 加入 Ray 集群，开始执行任务
```

```yaml
# 关键配置
workerGroupSpecs:
- replicas: 2       # 初始副本
  minReplicas: 1     # 最小（缩容下限）
  maxReplicas: 8     # 最大（扩容上限）
  scaleStrategy:
    workersToDelete: []  # 指定缩容时删除哪些 worker
```

## 4. Ray 在你的 8xH20 集群中的实践

### 4.1 分布式训练

```python
# 8 GPU 分布式训练
import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

trainer = TorchTrainer(
    train_func,
    scaling_config=ScalingConfig(
        num_workers=8,
        use_gpu=True,
        resources_per_worker={
            "GPU": 1,
            "CPU": 8,
        },
        # Placement Strategy
        placement_strategy="PACK",  # 尽量放在同一节点（NVLink通信）
    ),
)
```

### 4.2 超参搜索

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

# 在 8 GPU 上并行搜索超参数
tuner = tune.Tuner(
    tune.with_resources(train_func, {"gpu": 1}),
    param_space={
        "lr": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([16, 32, 64]),
        "hidden_dim": tune.choice([256, 512, 1024]),
    },
    tune_config=tune.TuneConfig(
        num_samples=32,  # 32 组超参数
        max_concurrent_trials=8,  # 8 GPU 并行
        scheduler=ASHAScheduler(
            max_t=100,
            grace_period=10,
            reduction_factor=3,
        ),
    ),
)
results = tuner.fit()
```

### 4.3 推理 + 训练混合

```yaml
# 在同一 Ray 集群中混合运行
# 4 GPU 给训练，4 GPU 给推理

# 通过 placement group 隔离
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: mixed-cluster
spec:
  workerGroupSpecs:
  # 训练 Worker Group
  - replicas: 1
    groupName: training
    rayStartParams:
      num-gpus: '4'
      resources: '{"training_gpu": 4}'
    template:
      spec:
        containers:
        - name: worker
          resources:
            limits:
              nvidia.com/gpu: 4

  # 推理 Worker Group
  - replicas: 4
    groupName: inference
    rayStartParams:
      num-gpus: '1'
      resources: '{"inference_gpu": 1}'
    template:
      spec:
        containers:
        - name: worker
          resources:
            limits:
              nvidia.com/gpu: 1
```

## 5. KubeRay 运维要点

### 5.1 监控

```bash
# Ray Dashboard（最重要的监控入口）
kubectl port-forward svc/gpu-cluster-head-svc 8265:8265

# Dashboard 提供：
# - 集群资源使用（CPU/GPU/Memory）
# - 活跃 Actor 列表和资源占用
# - 任务调度和执行状态
# - 日志查看
# - Profiling 工具
```

### 5.2 故障恢复

```yaml
# GCS 容错（Ray 2.0+）
headGroupSpec:
  rayStartParams:
    # 启用 GCS HA
    redis-password: "secret"
    # 外部 Redis 作为 GCS 后端
    # 或使用 RAY_GCS_HA_ENABLED=1
  template:
    spec:
      containers:
      - name: ray-head
        env:
        - name: RAY_GCS_FT_ENABLED
          value: "1"
        - name: RAY_REDIS_ADDRESS
          value: "redis-ha:6379"
```

### 5.3 常见问题

```
问题：Worker Pod 频繁 OOMKilled
原因：Object Store 默认使用 /dev/shm（共享内存）
     K8s 默认 /dev/shm 只有 64MB
解决：
  volumes:
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 64Gi  # 根据需要调整

问题：GPU 显示 0% 利用率但无法分配
原因：Actor 持有 GPU 但空闲
解决：设置 Actor 的 max_restarts 和 lifetime
      使用 serve 的 autoscaling 回收空闲副本

问题：跨节点通信慢
原因：Object Store 传输走 gRPC，无 RDMA
解决：尽量将需要大量通信的 task 放在同一节点
      使用 placement_strategy="PACK"
```

## 小结

```
Ray 的核心价值：
  1. 统一 API：训练 + 推理 + 数据处理，一个框架
  2. 弹性：自动扩缩容，与 K8s 深度集成
  3. 容错：自动重试，checkpoint 恢复
  4. 生态：与 PyTorch, vLLM, DeepSpeed 等集成

KubeRay 的核心价值：
  1. 声明式管理 Ray 集群生命周期
  2. 与 K8s 资源管理集成（GPU, Node Affinity 等）
  3. 与 Kueue 集成实现队列管理
  4. 滚动更新和零停机部署（RayService）

你的 8xH20 集群建议：
  ✅ 用 RayJob 替代手动 ssh 到节点启动训练
  ✅ 用 RayService 部署推理服务（自动扩缩）
  ✅ 配合 Kueue 做多团队资源管理
  ✅ 配置足够大的 /dev/shm（至少 64GB）
  ✅ 启用 GCS HA 避免 Head Node 单点故障
```
