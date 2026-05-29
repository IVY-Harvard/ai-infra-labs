# 07 - AI 工作负载的容错与高可用

## 引言：GPU 任务为什么需要特殊的容错

传统微服务的容错很直观——Pod 挂了重启就好。但 AI 工作负载完全不同：

```
训练任务：
  - 一个 8-GPU 分布式训练跑了 72 小时，一张卡 ECC 错误 → 全部 worker 失败
  - 没有 checkpoint → 72 小时白跑
  - 有 checkpoint 但恢复逻辑没写好 → 从头开始

推理服务：
  - 一个 LLM 推理 Pod 加载模型需要 5 分钟
  - Node 故障 → Pod 重调度 → 5 分钟加载 → 用户超时
  - 如果只有一个副本 → 5 分钟完全不可用

关键区别：GPU 任务有状态、启动慢、资源稀缺、相互依赖
```

对你的 8 张 H20 集群来说，任何一张卡的故障都意味着 12.5% 的算力损失。
设计好容错策略，才能真正把 GPU 利用率从 60% 拉到 90%+。

## 1. 训练任务自动恢复

### 1.1 Checkpoint 机制设计

```
训练容错的核心：Checkpoint + 自动重启

┌──────────────────────────────────────────────────┐
│                Training Loop                      │
│                                                   │
│  for epoch in range(epochs):                     │
│      train_one_epoch()                           │
│      if step % checkpoint_interval == 0:         │
│          save_checkpoint(model, optimizer, step)  │  ← 定期保存
│                                                   │
│  故障发生 ──► 进程退出 ──► K8s 检测 ──► 重启 Pod  │
│                                                   │
│  重启后：                                         │
│      load_checkpoint(latest)                     │  ← 自动恢复
│      resume_training(from_step)                  │
└──────────────────────────────────────────────────┘
```

### 1.2 分布式训练的故障恢复

```
                    ┌─────────────────┐
                    │   Job Controller │
                    │  (监控 worker)   │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Worker 0    │   │  Worker 1    │   │  Worker 2    │
│  (Rank 0)    │   │  (Rank 1)    │   │  (Rank 2)    │
│  GPU 0,1     │   │  GPU 2,3     │   │  GPU 4,5     │
└──────────────┘   └──────────────┘   └──────────────┘
        │                    │                    │
        └─────── NCCL AllReduce Ring ─────────────┘

Worker 1 故障 → 整个通信组断裂 → 所有 worker 需要重新初始化
```

#### PyTorch Elastic (torchrun) 恢复流程

```
故障检测 → 终止所有 worker → 等待节点恢复/替换
    → 重新建立通信组 → 加载 checkpoint → 继续训练

关键参数：
  --nproc_per_node=2          # 每节点 2 GPU
  --nnodes=1:4                # 弹性节点数 1~4
  --max_restarts=3            # 最大重启次数
  --rdzv_backend=c10d         # rendezvous 后端
  --rdzv_endpoint=master:29400
```

### 1.3 K8s 中的训练任务自动恢复

```yaml
# 使用 PyTorchJob (Kubeflow Training Operator) 实现自动恢复
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: distributed-training
spec:
  elasticPolicy:
    rdzvBackend: c10d
    minReplicas: 2
    maxReplicas: 4
    maxRestarts: 5           # 最多重启 5 次
  pytorchReplicaSpecs:
    Worker:
      replicas: 4
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: trainer
            image: training:latest
            resources:
              limits:
                nvidia.com/gpu: 2
            volumeMounts:
            - name: checkpoint
              mountPath: /checkpoints
          volumes:
          - name: checkpoint
            persistentVolumeClaim:
              claimName: training-checkpoints  # 共享存储
```

## 2. 推理服务高可用

### 2.1 多副本与滚动更新

```
推理服务 HA 架构（H20 集群示例）：

                    ┌─────────────┐
                    │   Ingress   │
                    │  /Gateway   │
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │ Service (LB) │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │  Pod A      │ │  Pod B      │ │  Pod C      │
    │  Model v1   │ │  Model v1   │ │  Model v1   │
    │  GPU 0,1    │ │  GPU 2,3    │ │  GPU 4,5    │
    │  (Active)   │ │  (Active)   │ │  (Standby)  │
    └─────────────┘ └─────────────┘ └─────────────┘

GPU 资源有限时的策略：
  - 2 个 Active 副本 + 1 个 Warm Standby
  - Standby 已加载模型到 GPU 显存，随时可接流量
  - PodDisruptionBudget 保证至少 2 个副本可用
```

### 2.2 模型加载优化

```yaml
# 减少故障恢复时间的关键：缩短模型加载时间
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  template:
    spec:
      initContainers:
      - name: model-loader
        image: model-cache:latest
        command: ["cp", "-r", "/models/llama-70b", "/shared/model"]
        volumeMounts:
        - name: model-vol
          mountPath: /shared
      containers:
      - name: inference
        image: vllm:latest
        startupProbe:            # 给模型加载足够时间
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
          failureThreshold: 30   # 最多等 5 分钟
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          periodSeconds: 10
          failureThreshold: 3
```

### 2.3 PodDisruptionBudget

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: llm-inference-pdb
spec:
  minAvailable: 2              # 任何时候至少保证 2 个副本
  selector:
    matchLabels:
      app: llm-inference
```

## 3. 节点故障检测与 Pod 迁移

### 3.1 GPU 故障的独特挑战

```
GPU 故障类型与检测难度：

故障类型           | 影响范围        | 检测方式          | 恢复时间
-----------------+---------------+-----------------+---------
GPU 硬件故障(ECC)  | 单卡           | nvidia-smi/DCGM  | 需换卡
GPU 挂死(XID 错误) | 单卡或多卡      | DCGM Exporter   | 重置驱动
NVLink 故障       | 卡间通信        | DCGM 诊断       | 需换卡
显存溢出(OOM)     | 单 Pod         | 进程退出码       | 秒级重启
驱动崩溃          | 整个节点所有 GPU | dmesg/kubelet   | 重启节点
节点宕机          | 节点所有 Pod    | Node NotReady   | Pod 迁移
```

### 3.2 故障检测与自动迁移流程

```
GPU 故障检测与 Pod 迁移完整流程：

1. 检测层
   ┌──────────────────┐    ┌──────────────────┐
   │  DCGM Exporter   │    │  Node Problem    │
   │  (GPU 指标采集)   │    │  Detector        │
   └────────┬─────────┘    └────────┬─────────┘
            │                        │
            ▼                        ▼
2. 判断层
   ┌──────────────────────────────────────────┐
   │  GPU Health Controller                    │
   │  - ECC uncorrectable errors > threshold  │
   │  - GPU 温度持续超过阈值                    │
   │  - XID 错误连续出现                       │
   │  - nvidia-smi 无响应                      │
   └────────────────────┬─────────────────────┘
                        │
                        ▼
3. 响应层
   ┌──────────────────────────────────────────┐
   │  自动响应策略                              │
   │  a) 给节点打 Taint: NoSchedule           │
   │  b) 驱逐该 GPU 上的 Pod                   │
   │  c) 触发 Pod 重调度到健康节点              │
   │  d) 发送告警通知运维                       │
   └──────────────────────────────────────────┘
```

### 3.3 Taint 自动隔离策略

```
节点状态            | Taint                                  | Effect
-----------------+---------------------------------------+-------------
GPU ECC 错误      | nvidia.com/gpu-unhealthy=true         | NoSchedule
GPU 温度过高       | nvidia.com/gpu-overheating=true       | NoSchedule
GPU 完全不可用     | nvidia.com/gpu-failed=true            | NoExecute
节点维护中         | node.kubernetes.io/maintenance=true   | NoExecute
```

## 4. 综合容错架构

### 4.1 关键指标

```
容错系统需要关注的 SLO/SLI：

指标                    | 目标值          | 说明
----------------------+----------------+------------------
故障检测时间(MTTD)     | < 30 秒         | 从故障发生到系统感知
故障恢复时间(MTTR)     | < 5 分钟（训练） | 从故障到恢复训练
                      | < 30 秒（推理）  | 从故障到推理服务恢复
Checkpoint 丢失步数    | < 100 步        | 每次故障最多回退的训练步数
服务可用性             | 99.9%          | 推理服务年度可用性
训练有效 GPU 时间占比   | > 95%          | 去除故障和恢复时间后的有效训练比例
```

## 5. 实践建议

### 对你的 8x H20 环境

1. **必须做**：配置 DCGM Exporter 监控所有 GPU 健康状态
2. **必须做**：训练代码集成 checkpoint，间隔根据成本分析决定
3. **推荐做**：部署 Node Problem Detector，自动隔离故障 GPU
4. **推荐做**：推理服务配置 PDB，保证最小可用副本数
5. **按需做**：实现自定义 GPU Health Controller 做细粒度故障分类

### Checkpoint 间隔计算

```
最优间隔 = sqrt(2 * checkpoint_cost * MTBF)

示例（8x H20, 70B 模型）：
  checkpoint_cost = 60 秒, MTBF = 168 小时
  最优间隔 ≈ 2.4 小时 → 实际建议每 1-2 小时 checkpoint 一次
```
