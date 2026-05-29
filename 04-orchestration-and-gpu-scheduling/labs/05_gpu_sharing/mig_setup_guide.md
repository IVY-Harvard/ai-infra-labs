# MIG (Multi-Instance GPU) 配置指南

> 注意：H20 不支持 MIG。本指南适用于 A100/H100/A30 GPU。
> 如果你的集群包含这些 GPU，本指南可以帮助你配置 MIG。

## 1. MIG 基本概念

```
A100 80GB MIG 分区示例：

┌──────────────────────────────────────────────────────┐
│                    A100 80GB                          │
│                                                       │
│  默认模式：1 个完整 GPU (80GB, 108 SM)                │
│                                                       │
│  MIG 模式：最多 7 个 GPU Instance                     │
│                                                       │
│  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐           │
│  │ 1g.10gb│ │ 1g.10gb│ │ 1g.10gb│ │ 1g.10gb│  ...    │
│  │  10GB  │ │  10GB  │ │  10GB  │ │  10GB  │         │
│  │  14 SM │ │  14 SM │ │  14 SM │ │  14 SM │         │
│  └───────┘ └───────┘ └───────┘ └───────┘           │
│                                                       │
│  可选配置：                                           │
│  - 7x 1g.10gb (7 个小实例)                           │
│  - 3x 2g.20gb + 1x 1g.10gb                          │
│  - 2x 3g.40gb                                        │
│  - 1x 4g.40gb + 1x 3g.40gb                          │
│  - 1x 7g.80gb (等同于完整 GPU)                       │
└──────────────────────────────────────────────────────┘
```

## 2. 启用 MIG 模式

```bash
# 检查 GPU 是否支持 MIG
nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv
# 输出：Disabled

# 启用 MIG（需要重置 GPU，会中断所有 GPU 工作负载）
sudo nvidia-smi -i 0 -mig 1
# 需要重启或重置 GPU
sudo nvidia-smi -i 0 --gpu-reset

# 验证 MIG 已启用
nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv
# 输出：Enabled
```

## 3. 创建 MIG 实例

```bash
# 查看支持的 MIG profile
nvidia-smi mig -lgip
# GPU 0: A100-SXM4-80GB
#   GPU Instance Profiles:
#     Profile  ID   Instances  Memory     SM
#     1g.10gb   19      7      9.75 GB    14
#     2g.20gb   14      3      19.5 GB    28
#     3g.40gb    9      2      39.25 GB   42
#     4g.40gb    5      1      39.25 GB   56
#     7g.80gb    0      1      79.0 GB   108

# 创建 GPU Instance（示例：2 个 3g.40gb）
sudo nvidia-smi mig -cgi 9,9 -i 0

# 创建 Compute Instance
sudo nvidia-smi mig -cci -i 0

# 查看创建的实例
nvidia-smi mig -lgi
nvidia-smi mig -lci
```

## 4. 在 K8s 中使用 MIG

### 4.1 GPU Operator MIG Manager 配置

```yaml
# MIG 分区策略配置（ConfigMap）
apiVersion: v1
kind: ConfigMap
metadata:
  name: mig-parted-config
  namespace: gpu-operator
data:
  config.yaml: |
    version: v1
    mig-configs:
      # 策略 1：全部切成小实例（推理）
      all-1g.10gb:
        - device-filter: ["A100-SXM4-80GB"]
          devices: all
          mig-enabled: true
          mig-devices:
            "1g.10gb": 7
      
      # 策略 2：混合配置（推理 + 小训练）
      mixed:
        - device-filter: ["A100-SXM4-80GB"]
          devices: all
          mig-enabled: true
          mig-devices:
            "3g.40gb": 1
            "2g.20gb": 1
            "1g.10gb": 2
      
      # 策略 3：大实例（训练）
      all-3g.40gb:
        - device-filter: ["A100-SXM4-80GB"]
          devices: all
          mig-enabled: true
          mig-devices:
            "3g.40gb": 2
```

### 4.2 使用 MIG 实例的 Pod

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: mig-inference
spec:
  containers:
  - name: inference
    image: my-inference:latest
    resources:
      limits:
        # 请求特定的 MIG profile
        nvidia.com/mig-1g.10gb: 1
        # 或者：nvidia.com/mig-3g.40gb: 1
```

## 5. MIG 策略切换

```bash
# 通过给节点打标签触发 MIG 重配置
# （GPU Operator MIG Manager 会监听标签变化）
kubectl label node gpu-node-0 nvidia.com/mig.config=all-1g.10gb --overwrite

# 注意：切换 MIG 配置需要：
# 1. 驱逐该节点上所有 GPU Pod
# 2. 重置 GPU
# 3. 重新创建 MIG 实例
# 过程大约需要 2-5 分钟
```

## 6. MIG 监控

```bash
# 查看 MIG 实例利用率
nvidia-smi mig -lgi
dcgmi dmon -e 1001,1002,1003  # 分实例监控

# DCGM Exporter 会自动为每个 MIG 实例生成独立指标
# DCGM_FI_DEV_GPU_UTIL{gpu="0", GPU_I_ID="0", GPU_I_PROFILE="1g.10gb"}
```
