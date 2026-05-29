# Lab 07: GPU 推理服务容量规划

## 概述

容量规划回答三个核心问题:
1. **现在**: 当前集群能支撑多少 QPS / 并发?
2. **趋势**: 按当前增速, 何时会触及容量上限?
3. **未来**: 如果业务增长 50%, 需要多少 GPU?

本 Lab 实现基于数据驱动的 GPU 推理容量规划系统。

---

## 文件结构

```
07_capacity_planning/
├── README.md                  ← 本文件
├── capacity_calculator.py     ← 容量计算器 (KV Cache / 吞吐 / 并发)
├── traffic_forecaster.py      ← 流量预测 (Holt-Winters / Prophet)
└── scaling_simulator.py       ← 扩缩容模拟器
```

---

## 容量瓶颈分析

GPU 推理服务的容量由以下因素共同决定:

```
                    ┌─────────────────────────────┐
                    │       容量瓶颈层级            │
                    ├─────────────────────────────┤
                    │                             │
                    │  Layer 1: GPU 计算           │
                    │  ├── SM Active < 95%?       │
                    │  ├── Tensor Core 利用率?     │
                    │  └── 瓶颈 = max_num_seqs    │
                    │                             │
                    │  Layer 2: GPU 显存 (KV Cache)│  ← 通常是首要瓶颈
                    │  ├── KV Cache < 90%?        │
                    │  ├── 可容纳的并发 × 长度?    │
                    │  └── 瓶颈 = gpu_mem_util     │
                    │                             │
                    │  Layer 3: 网络 (TP 通信)     │
                    │  ├── NVLink 带宽利用率?      │
                    │  └── All-Reduce 延迟?        │
                    │                             │
                    │  Layer 4: Host (CPU/PCIe)   │
                    │  ├── Tokenizer CPU?          │
                    │  ├── PCIe 带宽 (swap)?       │
                    │  └── 网络 I/O?               │
                    │                             │
                    └─────────────────────────────┘
```

---

## 关键公式

### KV Cache 容量

```
单请求 KV Cache (bytes) = 
    num_layers × 2 × num_kv_heads_per_tp × head_dim × seq_len × dtype_size

Qwen2.5-72B (TP=8, BF16):
    80 × 2 × 1 × 128 × seq_len × 2 = 40,960 × seq_len bytes

可用 KV Cache 总量 (per GPU) = 
    GPU_memory × gpu_memory_utilization - model_weights - activation - overhead

H20 (96GB), gpu_mem_util=0.9:
    96GB × 0.9 - 9GB(weights) - 4GB(activation) - 5GB(overhead) ≈ 68.4 GB

最大并发数 × 平均长度:
    68.4 GB / 40,960 bytes = ~1,750,000 tokens
    如果平均 prompt+output = 4096 tokens:
    最大并发 ≈ 1,750,000 / 4096 ≈ 427 请求
```

### 吞吐上限

```
理论 Decode 吞吐 (tokens/s) ≈ 
    GPU_memory_bandwidth / (2 × model_params_per_gpu × dtype_size)

H20: 4 TB/s bandwidth, Qwen2.5-72B TP=8:
    4,000 GB/s / (2 × 9B × 2 bytes) ≈ 111 tokens/s per request

Batch 吞吐:
    随 batch_size 近线性增长直到 Memory Bandwidth 饱和
    实测 batch=64: ~1500-2000 tokens/s (total)
```

---

## 运行前提

```bash
pip install numpy pandas scipy prometheus-api-client statsmodels
```
