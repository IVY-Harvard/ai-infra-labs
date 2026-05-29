# MPS (Multi-Process Service) 在 K8s 中的配置

## 1. MPS 概述

```
MPS 工作原理：

传统模式（时间复用）：
  Process A ─────┐    ┌───── GPU ──────┐
  Process B ─────┤────│  时间片轮转执行  │
  Process C ─────┘    └────────────────┘
  → 上下文切换开销大，不能真正并行

MPS 模式（空间复用）：
  Process A ─────┐    ┌───── GPU ──────┐
  Process B ─────┤────│  MPS Server    │
  Process C ─────┘    │  并行执行多个   │
                      │  CUDA Context  │
                      └────────────────┘
  → 多个进程共享 GPU 计算单元，真正并行
  → 适合小 kernel、推理场景

H20 上使用 MPS 的优势：
  - 96GB 显存可以同时跑多个小模型
  - 推理 kernel 通常很小，GPU SM 利用率低
  - MPS 让多个推理服务并行使用 SM
```

## 2. 手动配置 MPS

```bash
# 在 GPU 节点上启动 MPS
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log

# 启动 MPS 控制守护进程
nvidia-cuda-mps-control -d

# 验证 MPS 运行
echo "get_server_list" | nvidia-cuda-mps-control
# 应该看到 MPS server PID

# 设置每个客户端的资源限制（可选）
echo "set_default_active_thread_percentage 25" | nvidia-cuda-mps-control
# 每个客户端最多使用 25% 的 SM

# 停止 MPS
echo quit | nvidia-cuda-mps-control
```

## 3. K8s 中的 MPS 配置

### 3.1 MPS DaemonSet

```yaml
# 在每个 GPU 节点上运行 MPS Server
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: nvidia-mps-server
  namespace: gpu-system
spec:
  selector:
    matchLabels:
      app: nvidia-mps
  template:
    metadata:
      labels:
        app: nvidia-mps
    spec:
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-H20
      containers:
      - name: mps-server
        image: nvidia/cuda:12.2.0-base-ubuntu22.04
        command:
        - sh
        - -c
        - |
          # 为每张 GPU 启动一个 MPS server
          for gpu_id in $(seq 0 7); do
            export CUDA_VISIBLE_DEVICES=$gpu_id
            export CUDA_MPS_PIPE_DIRECTORY=/mps/pipe/gpu$gpu_id
            export CUDA_MPS_LOG_DIRECTORY=/mps/log/gpu$gpu_id
            mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
            
            # 设置默认线程百分比（4 个客户端各 25%）
            nvidia-cuda-mps-control -d
            echo "set_default_active_thread_percentage 25" | \
              nvidia-cuda-mps-control
          done
          
          # 保持运行
          sleep infinity
        securityContext:
          privileged: true
        volumeMounts:
        - name: mps-shared
          mountPath: /mps
        resources:
          limits:
            nvidia.com/gpu: 8   # 需要访问所有 GPU
      volumes:
      - name: mps-shared
        hostPath:
          path: /var/run/nvidia-mps
          type: DirectoryOrCreate
```

### 3.2 使用 MPS 的 Pod

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: inference-with-mps
spec:
  containers:
  - name: inference
    image: my-inference:latest
    env:
    # 指向 MPS server 的 pipe 目录
    - name: CUDA_MPS_PIPE_DIRECTORY
      value: "/mps/pipe/gpu0"
    volumeMounts:
    - name: mps-shared
      mountPath: /mps
      readOnly: true
    resources:
      limits:
        # 注意：使用 MPS 时仍然需要声明 GPU 资源
        # 但实际上多个 Pod 共享同一张物理 GPU
        nvidia.com/gpu: 1
  volumes:
  - name: mps-shared
    hostPath:
      path: /var/run/nvidia-mps
```

## 4. MPS 资源限制

```bash
# 限制每个客户端可用的 SM 百分比
echo "set_default_active_thread_percentage 25" | nvidia-cuda-mps-control
# → 每个客户端最多用 25% SM（适合 4 路共享）

# 限制每个客户端的 Pinned Memory
echo "set_default_device_pinned_mem_limit 0 24G" | nvidia-cuda-mps-control
# → GPU 0 的每个客户端最多 24GB pinned memory（96GB/4=24GB）
```

## 5. MPS 监控

```bash
# 查看 MPS 客户端状态
echo "get_server_list" | nvidia-cuda-mps-control
echo "ps" | nvidia-cuda-mps-control

# 通过 DCGM 监控
# MPS 模式下 DCGM 看到的是物理 GPU 的总利用率
# 需要结合进程级监控来区分不同客户端
nvidia-smi pmon -i 0  # 进程级监控
```

## 6. 注意事项

```
MPS 的局限性：
  1. 没有显存隔离 — 一个客户端 OOM 可能影响其他客户端
  2. 没有故障隔离 — 一个客户端的 CUDA 错误会影响所有客户端
  3. 需要所有客户端使用相同的 CUDA 版本
  4. MPS Server 崩溃 → 所有客户端都会失败

适用场景：
  ✅ 多个小模型推理服务共享 GPU
  ✅ 模型 serving（每个请求的 GPU 计算量小）
  ✅ 开发环境多人共享

不适用场景：
  ❌ 大模型训练（需要独占全部显存和带宽）
  ❌ 对隔离性要求高的生产环境
  ❌ 不同 CUDA 版本的工作负载混合
```
