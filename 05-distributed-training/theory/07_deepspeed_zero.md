# 07 — DeepSpeed ZeRO：Stage 1/2/3 原理与显存分析

## 1. ZeRO 的动机

### 1.1 数据并行的显存浪费

标准 DDP 中，每张 GPU 存储完整的：
- 模型参数 (Φ)
- 梯度 (Φ)
- 优化器状态 (KΦ)，Adam K=12 (FP32: master_weight + momentum + variance)

```
混合精度 (BF16 forward/backward + FP32 optimizer):

参数 Φ (BF16):    2Φ bytes
梯度 Φ (BF16):    2Φ bytes  
优化器状态:
  Master weight (FP32): 4Φ bytes
  Momentum (FP32):      4Φ bytes
  Variance (FP32):      4Φ bytes
─────────────────────────────────
总计: 2Φ + 2Φ + 12Φ = 16Φ bytes per GPU

7B 模型: 16 × 7B = 112 GB per GPU
8 GPU DDP: 8 × 112 = 896 GB 总显存使用

但模型只有 7B 参数！冗余 8 倍！
```

### 1.2 ZeRO 的核心思想

ZeRO (Zero Redundancy Optimizer)：将冗余状态切分到所有 GPU，需要时再收集。

```
ZeRO Stage 1: 切分优化器状态
ZeRO Stage 2: 切分优化器状态 + 梯度
ZeRO Stage 3: 切分优化器状态 + 梯度 + 参数
```

## 2. ZeRO Stage 1：优化器状态切分

### 2.1 原理

每张 GPU 只存储 1/N 的优化器状态，但保留完整的参数和梯度。

```
GPU 0: 完整参数 + 完整梯度 + Optimizer State (params 0 to Φ/N)
GPU 1: 完整参数 + 完整梯度 + Optimizer State (params Φ/N to 2Φ/N)
...
GPU N-1: 完整参数 + 完整梯度 + Optimizer State (params (N-1)Φ/N to Φ)
```

### 2.2 训练流程

```
Step 1: Forward (正常，每 GPU 用完整参数)
Step 2: Backward (正常，每 GPU 计算完整梯度)
Step 3: AllReduce 梯度 (与 DDP 相同)
Step 4: 每 GPU 只更新自己负责的参数分片 (用本地优化器状态)
Step 5: AllGather 更新后的参数 (收集完整参数)
```

### 2.3 显存分析

```
每 GPU:
  参数 (BF16):         2Φ bytes (完整)
  梯度 (BF16):         2Φ bytes (完整)
  优化器状态:           12Φ/N bytes (1/N)
─────────────────────────────────
  总计: 4Φ + 12Φ/N bytes

7B 模型, N=8:
  4×14 + 12×14/8 = 56 + 21 = 77 GB  (vs DDP 的 112 GB)
  节省: 32%
```

### 2.4 通信量

```
与 DDP 相同: 2Φ (AllReduce)
额外: AllGather 更新后参数 = Φ (但可与计算 overlap)

总通信量: 3Φ → 比 DDP 多 50%
但优化器更新可以并行，实际影响小
```

## 3. ZeRO Stage 2：梯度 + 优化器状态切分

### 3.1 原理

在 Stage 1 基础上，梯度也切分存储。每张 GPU 只保留自己负责的参数分片对应的梯度。

```
GPU 0: 完整参数 + Gradient(0:Φ/N) + Optimizer State(0:Φ/N)
GPU 1: 完整参数 + Gradient(Φ/N:2Φ/N) + Optimizer State(Φ/N:2Φ/N)
...
```

### 3.2 训练流程

```
Step 1: Forward (正常，用完整参数)
Step 2: Backward
  - 每个参数的梯度计算完成后，立即做 ReduceScatter
  - ReduceScatter: 每 GPU 得到自己负责的梯度分片（已 reduce）
  - 非本地的梯度可以立即释放！→ 显存节省

Step 3: 每 GPU 用本地梯度分片更新本地参数分片

Step 4: AllGather 更新后的参数
```

### 3.3 显存分析

```
每 GPU:
  参数 (BF16):         2Φ bytes (完整)
  梯度 (BF16):         2Φ/N bytes (1/N)
  优化器状态:           12Φ/N bytes (1/N)
─────────────────────────────────
  总计: 2Φ + 14Φ/N bytes

7B 模型, N=8:
  2×14 + 14×14/8 = 28 + 24.5 = 52.5 GB  (vs DDP 112 GB, Stage1 77 GB)
  节省: 53%

注意: 反向传播期间峰值显存可能更高
  (某一时刻可能同时持有多个层的完整梯度)
```

### 3.4 通信量

```
ReduceScatter (替代 AllReduce 的 reduce 部分): Φ × (N-1)/N ≈ Φ
AllGather (参数恢复): Φ × (N-1)/N ≈ Φ

总通信量: 2Φ → 与 DDP 相同！

关键洞察:
  AllReduce = ReduceScatter + AllGather
  Stage 2 只是把 AllReduce 拆开了:
    ReduceScatter 在反向传播时做 (梯度)
    AllGather 在优化器更新后做 (参数)
  总通信量不变，但显存节省了！
```

## 4. ZeRO Stage 3：全切分

### 4.1 原理

参数、梯度、优化器状态全部切分。每张 GPU 只存 1/N 的所有状态。

```
GPU i: Params(iΦ/N : (i+1)Φ/N) + Grad(同) + OptState(同)

需要用到某层参数时: AllGather 从所有 GPU 收集
计算完成后: 释放非本地参数
```

### 4.2 训练流程

```
Step 1: Forward
  for each layer:
    AllGather: 收集该层完整参数
    前向计算
    释放非本地参数 (只保留本地分片)

Step 2: Backward
  for each layer (reverse):
    AllGather: 再次收集该层完整参数
    反向计算梯度
    ReduceScatter: 将梯度 reduce 到各自负责的分片
    释放非本地参数和完整梯度

Step 3: Optimizer
  每 GPU 用本地梯度分片更新本地参数分片
  (不需要额外通信)
```

### 4.3 显存分析

```
每 GPU (稳态):
  参数分片 (BF16):     2Φ/N bytes
  梯度分片 (BF16):     2Φ/N bytes
  优化器状态分片:       12Φ/N bytes
─────────────────────────────────
  总计: 16Φ/N bytes

7B 模型, N=8:
  16 × 14 / 8 = 28 GB  (vs DDP 112 GB)
  节省: 75%

峰值显存 (Forward 期间 AllGather 后):
  16Φ/N + 2Φ (当前层完整参数) 
  = 28 + 14 = 42 GB
  (但层计算完后就释放，实际峰值取决于实现)
```

### 4.4 通信量

```
Forward AllGather: L × (layer_params) × (N-1)/N ≈ Φ
Backward AllGather: L × (layer_params) × (N-1)/N ≈ Φ
Backward ReduceScatter: L × (layer_params) × (N-1)/N ≈ Φ

总通信量: 3Φ → 比 DDP 多 50%

对比:
  DDP:     2Φ     (AllReduce 梯度)
  Stage 1: 3Φ     (AllReduce 梯度 + AllGather 参数)
  Stage 2: 2Φ     (ReduceScatter 梯度 + AllGather 参数)
  Stage 3: 3Φ     (2× AllGather 参数 + ReduceScatter 梯度)
```

## 5. ZeRO 各 Stage 对比

```
                Stage 1     Stage 2     Stage 3
────────────────────────────────────────────────────
参数存储         完整(2Φ)    完整(2Φ)    分片(2Φ/N)
梯度存储         完整(2Φ)    分片(2Φ/N)  分片(2Φ/N)
优化器存储       分片(12Φ/N) 分片(12Φ/N) 分片(12Φ/N)
────────────────────────────────────────────────────
每GPU显存        4Φ+12Φ/N   2Φ+14Φ/N   16Φ/N
通信量           ~3Φ         ~2Φ         ~3Φ
编程复杂度       低          低          中
────────────────────────────────────────────────────

7B, 8GPU:
  Stage 1: 77 GB
  Stage 2: 52.5 GB
  Stage 3: 28 GB  (峰值 ~42 GB)
  DDP:     112 GB
```

## 6. ZeRO-Infinity

### 6.1 核心思想

将 ZeRO-3 的切分扩展到 CPU 内存和 NVMe SSD：

```
存储层级:
  GPU HBM (96 GB per H20) → 计算用
  CPU DRAM (通常 512GB - 2TB) → 缓存
  NVMe SSD (通常 TB 级) → 最大存储

ZeRO-Infinity 允许:
  - 优化器状态放 CPU/NVMe
  - 参数放 CPU，按需传到 GPU
  - 理论上可训练无限大模型（只要有足够的 SSD）
```

### 6.2 Offload 策略

```
ZeRO-Offload (Stage 2 + CPU offload):
  - 优化器状态和优化器计算放 CPU
  - GPU 只做 forward/backward
  - CPU-GPU 通信: PCIe (32 GB/s)
  
ZeRO-Infinity (Stage 3 + CPU + NVMe):
  - 参数/梯度/优化器全部可以 offload
  - 利用 NVMe 的带宽 (3-7 GB/s per drive, RAID 可更高)
  - Prefetching: 提前将下一层参数从 NVMe → CPU → GPU
```

### 6.3 性能影响

```
ZeRO-Offload 吞吐量 (7B, 单机 8 GPU):
  纯 GPU (ZeRO-2): 100% 基线
  + CPU Offload:    60-80% (PCIe 成为瓶颈)
  + NVMe Offload:   30-50%

适用场景:
  - GPU 显存不够但有大量 CPU 内存
  - 训练超大模型的小 batch (fine-tuning)
  - 不需要极高吞吐的场景 (实验/研究)
```

## 7. DeepSpeed ZeRO 配置详解

### 7.1 Stage 1 配置

```json
{
  "zero_optimization": {
    "stage": 1,
    "reduce_bucket_size": 5e8,
    "allgather_bucket_size": 5e8
  },
  "bf16": {
    "enabled": true
  },
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 1e-4,
      "betas": [0.9, 0.999]
    }
  },
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 8
}
```

### 7.2 Stage 2 配置

```json
{
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "allgather_bucket_size": 5e8,
    "contiguous_gradients": true
  }
}
```

### 7.3 Stage 3 配置

```json
{
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

### 7.4 ZeRO-Offload 配置

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

## 8. ZeRO vs FSDP 对比

```
                    DeepSpeed ZeRO-3          PyTorch FSDP
──────────────────────────────────────────────────────────────
底层框架            DeepSpeed                  PyTorch native
切分粒度            Parameter group            nn.Module (FSDP unit)
通信实现            自定义                      NCCL
CPU Offload        原生支持                    支持 (limited)
NVMe Offload       支持                       不支持
torch.compile      兼容性差                    FSDP2 支持
维护活跃度          持续更新                    PyTorch 核心团队
生态集成            HuggingFace/Megatron-DS    PyTorch 原生
──────────────────────────────────────────────────────────────

选择建议:
  - 需要 NVMe offload / 极大模型: DeepSpeed ZeRO
  - 需要 torch.compile / 组合其他 PyTorch 特性: FSDP2
  - 企业生产 / 长期维护: FSDP (PyTorch 官方)
  - 快速实验 / HuggingFace 生态: DeepSpeed
```

## 9. ZeRO 的通信优化

### 9.1 Communication Overlap

```
Stage 2:
  Backward 过程中:
    Layer N 梯度 → 立即 ReduceScatter
    Layer N-1 梯度 → 立即 ReduceScatter
    ...
  与反向计算重叠 → 隐藏通信延迟

Stage 3:
  Forward: AllGather(layer i+1) 与 compute(layer i) 重叠
  Backward: AllGather(layer i-1) 与 compute(layer i) 重叠
  
  Prefetch 策略: 提前 fetch 下 K 层的参数
```

### 9.2 Gradient Accumulation

```
当使用 gradient accumulation 时:
  前 K-1 个 micro-batch: 只累积梯度，不通信
  第 K 个 micro-batch: 做 ReduceScatter + AllGather

好处: 通信频率降为 1/K
     每次通信的数据量不变
     → 有效减少通信开销的占比
```

### 9.3 Stage 3 的 Parameter Persistence

```
stage3_param_persistence_threshold: 
  小于此阈值的参数不做切分（保持完整）
  
典型: LayerNorm 的参数很小 (H = 4096 → 16KB)
  切分这些小参数的通信 overhead > 显存节省
  → 保持完整存储

stage3_max_live_parameters:
  同时存在于 GPU 的最大参数量
  控制 prefetch 的激进程度
```

## 10. 实践建议

### 10.1 Stage 选择

```
模型 < 3B, 8 GPU:
  DDP 足够 (每卡 <24 GB)
  或 Stage 1 增大 batch

模型 3B-7B, 8 GPU:
  Stage 2 (显存够, 通信与 DDP 相同)
  如果激活值显存紧张: + Activation Checkpoint

模型 7B-13B, 8 GPU:
  Stage 3 (必须切分参数)
  或 FSDP FULL_SHARD
  
模型 > 13B, 8 GPU:
  Stage 3 + CPU offload
  或 TP + ZeRO-1/2
```

### 10.2 性能调优

```
1. reduce_bucket_size:
   太小 → 通信次数多, latency 累积
   太大 → overlap 效果差
   建议: 5e8 (500M elements) 作为起点

2. stage3_prefetch_bucket_size:
   控制预取粒度
   太大 → 显存峰值高
   太小 → 预取不及时
   建议: 等于 reduce_bucket_size

3. overlap_comm:
   几乎总是应该开启
   例外: 调试时关闭以简化问题

4. contiguous_gradients (Stage 2):
   将梯度存储为连续内存
   减少内存碎片，略微增加显存
```

## 11. 下一步

- [08_megatron_core.md](08_megatron_core.md)：Megatron-Core 如何整合 ZeRO + TP + PP
- [Lab 06](../labs/06_deepspeed_zero/)：ZeRO Stage 1/2/3 实战
