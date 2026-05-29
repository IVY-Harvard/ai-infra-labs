# 05 — 3D 并行组合策略

## 1. 为什么需要 3D 并行

单一并行策略各有局限：

| 策略 | 优势 | 局限 |
|------|------|------|
| DP | 扩展吞吐量，编程简单 | 每卡存完整模型，不解决模型过大 |
| TP | 切分层内参数和激活 | 通信频率高，受限于 NVLink |
| PP | 切分层间，通信量小 | Pipeline Bubble，编程复杂 |

3D 并行 = TP + PP + DP 组合，利用各策略优势：

```
TP: 解决单层太大（参数+激活值切分），放在高带宽互联内
PP: 解决层数太多（跨节点切分），用低带宽链路
DP: 扩展吞吐量（数据并行），每步一次同步
```

## 2. 通信量 vs 带宽匹配原则

### 2.1 三层通信需求

```
通信层级          频率        数据量/次      总通信量/步     带宽需求
─────────────────────────────────────────────────────────────────────
TP (层内)        每层 4 次    B×S×H          4L×B×S×H        高 (NVLink)
PP (层间)        每 micro-batch  B_m×S×H    2m×B_m×S×H      低 (PCIe/网络)
DP (步间)        每步 1 次    2M             2M              中 (网络+overlap)
```

### 2.2 硬件带宽层级

```
典型 8×H20 单机:
  NVLink:     ~450 GB/s bidirectional (GPU 间)
  PCIe 5.0:   ~64 GB/s (CPU-GPU / 跨 NUMA)
  
多机:
  InfiniBand: ~50 GB/s (400Gbps HDR)
  RoCE:       ~25 GB/s (200Gbps)
  Ethernet:   ~12.5 GB/s (100GbE)
```

### 2.3 匹配逻辑

```
TP 通信量最大、频率最高 → 放在带宽最高的 NVLink 互联内（同机/同 NVSwitch）
PP 通信量小但有延迟要求 → 放在中等带宽（跨机 IB 或同机 PCIe）
DP 通信量大但频率低（一步一次）→ 可以走网络，且能与计算 overlap

这就是 3D 并行的核心设计哲学：
  通信需求与硬件拓扑匹配
```

## 3. 并行组 (Process Groups) 的概念

### 3.1 三维并行组

```
总 GPU 数: N = TP_size × PP_size × DP_size

例: 16 GPU, TP=4, PP=2, DP=2

GPU 编号 0-15，形成一个 3D 网格:

         DP=0          DP=1
PP=0: [0,1,2,3]    [8,9,10,11]     ← TP group
PP=1: [4,5,6,7]    [12,13,14,15]   ← TP group
       ↑                ↑
    PP group          PP group

DP group: {0,8}, {1,9}, {2,10}, {3,11}, {4,12}, {5,13}, {6,14}, {7,15}
TP group: {0,1,2,3}, {4,5,6,7}, {8,9,10,11}, {12,13,14,15}
PP group: {0,4}, {1,5}, {2,6}, {3,7}, {8,12}, {9,13}, {10,14}, {11,15}
```

### 3.2 8×H20 单机配置

```
8 GPU, TP=4, PP=2, DP=1:

         DP=0
PP=0: [0,1,2,3]    ← TP group 0 (NVLink 互联)
PP=1: [4,5,6,7]    ← TP group 1 (NVLink 互联)

PP group: {0,4}, {1,5}, {2,6}, {3,7}
DP group: 只有自己（DP=1，无数据并行）

通信分析:
  TP: GPU 0-3 之间 AllReduce (NVLink, ~450 GB/s) ✓
  PP: GPU 0→4 之间 P2P Send (NVLink, 足够) ✓
  DP: 无
```

```
8 GPU, TP=4, PP=1, DP=2:

PP=0: [0,1,2,3]  [4,5,6,7]
       TP group 0   TP group 1

DP group: {0,4}, {1,5}, {2,6}, {3,7}

通信分析:
  TP: NVLink AllReduce ✓
  PP: 无
  DP: AllReduce between {0,4} 等 (NVLink, ~450 GB/s) ✓
```

## 4. 3D 并行的显存分析

### 4.1 参数显存

```
模型参数 M (bytes):

每 GPU 参数量 = M / (TP × PP)
  (TP 切分层内参数，PP 切分层间参数)
  DP 不切分参数（每个 DP rank 存完整一份）

例: 13B 模型, BF16, TP=4, PP=2:
  每 GPU 参数 = 13B × 2 / (4 × 2) = 3.25 GB
```

### 4.2 优化器状态

```
Adam 优化器 (BF16 混合精度):
  Master weight (FP32):  M / (TP × PP) × 2  (FP32 是 BF16 的 2 倍)
  Momentum (FP32):       M / (TP × PP) × 2
  Variance (FP32):       M / (TP × PP) × 2

  优化器状态 = M / (TP × PP) × 6

例: 13B × 2 / (4 × 2) × 6 = 19.5 GB per GPU
```

### 4.3 激活值显存

```
不使用 Activation Checkpointing:
  每层激活值 ≈ B_micro × S × H × factor × dtype
  总激活值 = layers_per_stage × 上式

  PP 切分后每 stage 的层数 = L / PP
  TP 切分后激活值部分减少（取决于是否有 SP）

使用 Activation Checkpointing:
  只保存每层的输入，减少到 ≈ layers_per_stage × B_micro × S × H × dtype
```

### 4.4 总显存预算

```
13B 模型, BF16, TP=4, PP=2, 8×H20 (96GB):

每 GPU:
  参数:     3.25 GB
  梯度:     3.25 GB
  优化器:    19.5 GB
  激活值:    ~10-30 GB (取决于 batch size 和 checkpoint)
  ─────────────────────
  合计:      ~36-56 GB  (远小于 96 GB，有余量增大 batch)
```

## 5. 3D 并行的通信调度

### 5.1 一步训练的通信流程

```
Step 1: PP Forward
  Stage 0 做前向 → P2P Send activation → Stage 1
  (TP AllReduce 在每个 stage 内部发生)

Step 2: PP Backward  
  Stage 1 做反向 → P2P Send gradient → Stage 0
  (TP AllReduce 在每个 stage 内部发生)

Step 3: DP Gradient Sync
  每个 DP group 内 AllReduce 梯度
  (可以与 PP backward 的后半段 overlap)

Step 4: Optimizer Update
  每个 GPU 更新自己的参数分片
```

### 5.2 通信重叠策略

```
关键优化：让不同维度的通信 overlap:

1. TP AllReduce 与 PP P2P overlap:
   不同 micro-batch 的 TP 通信和 PP 通信可以在不同 CUDA stream 上并行

2. DP AllReduce 与 PP Backward overlap:
   Pipeline 的最后几个 micro-batch 做反向时，先完成的层可以开始 DP AllReduce

3. PP warmup 期间的 DP prefetch:
   Warmup 阶段 GPU 有空闲，可以做数据预取

Timeline:
─────────────────────────────────────────────────────
PP:   [F0][F1][F2][B0,F3][B1,F4]...[Bn-2][Bn-1]
TP:   ↕    ↕    ↕    ↕       ↕        ↕      ↕
DP:                           [AllReduce layer N]....[AllReduce layer 0]
                              ← overlap with backward →
─────────────────────────────────────────────────────
```

## 6. 配置选择决策树

### 6.1 给定 N 张 GPU，如何选择 TP/PP/DP

```
                      ┌─────────────────────┐
                      │ 模型能放入单卡？       │
                      └──────────┬──────────┘
                          Yes    │    No
                      ┌──────────┴──────────┐
                      ▼                      ▼
                 纯 DP (DDP)         ┌─────────────────┐
                 或 FSDP             │ 需要多少 GPU     │
                                    │ 才能放下模型？    │
                                    └────────┬────────┘
                                             ▼
                                    ┌─────────────────┐
                                    │ 计算 min(TP×PP)  │
                                    │ 使每卡显存 < 80% │
                                    └────────┬────────┘
                                             ▼
                              ┌───────────────────────────┐
                              │ 优先最大化 TP (≤NVLink数)  │
                              │ 然后分配 PP               │
                              │ 剩余 GPU 做 DP            │
                              └───────────────────────────┘
```

### 6.2 具体决策准则

```
Step 1: 确定 TP_size
  - TP ≤ NVLink 互联的 GPU 数（单机通常 4 或 8）
  - TP 必须整除 num_attention_heads
  - TP 越大，通信开销越大，但单卡显存越小
  - 建议: 先试 TP=4，不够再 TP=8

Step 2: 确定 PP_size
  - PP 使得每个 stage 的层数合理（至少 4-8 层）
  - PP 越大，Bubble 越大 → 需要更多 micro-batch 弥补
  - 建议: PP=2 (Bubble ≈ 10% with m=8)

Step 3: 确定 DP_size
  - DP = Total_GPUs / (TP × PP)
  - DP 越大，global batch 越大 → 注意学习率调整
  - DP 的通信可以完全被计算 overlap

Step 4: 确定 micro-batch 数 m
  - m ≥ 4 × PP_size (使 Bubble < 20%)
  - m × micro_batch_size × DP = global_batch_size
  - 太大的 global batch 可能影响收敛
```

## 7. 典型配置案例

### 7.1 8×H20, 7B 模型

```
方案 A: TP=4, PP=1, DP=2
  优势: 无 Pipeline Bubble
  显存: 每卡约 14GB 参数 → 够
  通信: TP AllReduce + DP AllReduce

方案 B: TP=2, PP=1, DP=4
  优势: 更大 DP → 更高吞吐
  风险: TP=2 时矩阵太小，MFU 可能下降

方案 C: ZeRO-3 (DP=8)
  优势: 无需改模型代码
  劣势: 通信量 3M（比 DDP 多 50%）

推荐: 方案 A (TP=4, DP=2)
```

### 7.2 8×H20, 13B 模型

```
方案 A: TP=4, PP=2, DP=1
  每卡参数: 13B×2/(4×2) = 3.25 GB
  优化器: 19.5 GB
  总: ~26 GB + 激活值 → 可以用大 batch

方案 B: TP=8, PP=1, DP=1
  每卡参数: 13B×2/8 = 3.25 GB
  无 Bubble，但 TP=8 通信量大
  
方案 C: FSDP (ZeRO-3)
  不需要 TP/PP
  通信量大，但编程简单

推荐: 方案 A (TP=4, PP=2) — 平衡 Bubble 和通信
```

### 7.3 多机 (32 GPU = 4 nodes × 8), 70B 模型

```
推荐: TP=8, PP=4, DP=1
  
  Node 0: GPU 0-7 → TP group, PP Stage 0
  Node 1: GPU 8-15 → TP group, PP Stage 1
  Node 2: GPU 16-23 → TP group, PP Stage 2
  Node 3: GPU 24-31 → TP group, PP Stage 3

  TP: NVLink 内 (450 GB/s) ✓
  PP: IB 跨机 (50 GB/s) ✓ (P2P 量小)
  DP: 无 (所有 GPU 用于模型并行)

  每卡参数: 70B×2/(8×4) = 4.375 GB → 很舒服
```

## 8. 性能估算

### 8.1 MFU (Model FLOPs Utilization)

```
理论 FLOPs per step (forward + backward):
  FLOPs = 6 × N_params × tokens_per_step
        = 6 × N × B × S  (B = global batch, S = seq_len)

硬件峰值:
  8 × H20 BF16 = 8 × 148 TFLOPS = 1184 TFLOPS

MFU = actual_FLOPs / (peak_FLOPs × step_time)

目标 MFU:
  DDP: 50-60% (通信开销小)
  TP=4: 40-50% (通信较多)
  TP=4+PP=2: 35-45% (Bubble + 通信)
```

### 8.2 吞吐量预估

```
8×H20, 7B 模型, TP=4, DP=2, BF16, S=2048:
  
  FLOPs/token = 6 × 7B = 42 GFLOPs
  每 GPU 有效算力 = 148 × 0.45 (MFU) = 66.6 TFLOPS
  
  Tokens/sec per GPU = 66.6e12 / 42e9 = 1585 tokens/sec
  Total = 1585 × 8 = 12,680 tokens/sec

  每天训练: 12,680 × 86400 = 1.1B tokens/day
  
  训练 1T tokens 需要: 1000 / 1.1 ≈ 909 天
  → 8 卡训练 7B 到 1T tokens 不现实，但 fine-tune (几 B tokens) OK
```

## 9. 常见陷阱和调优

### 9.1 TP-PP 交界处理

```
PP Stage 边界的 tensor 需要特殊处理:
  - TP group 内的 tensor 是切分的
  - 跨 PP stage 传递时需要 AllGather 再 Send? 还是直接 P2P?

方案 1: 每个 TP rank 独立 P2P
  GPU 0 → GPU 4, GPU 1 → GPU 5, ... (一一对应)
  → 简单，但需要 TP rank 对齐

方案 2: AllGather → Send → Scatter
  → 额外通信，不推荐

Megatron-LM 用方案 1 (TP rank 对应的 PP Send/Recv)
```

### 9.2 数据加载

```
3D 并行中，同一个 DP group 的 rank 需要不同的数据:
  TP group: 相同数据（切分模型，不切分数据）
  PP group: 相同数据（同一个 micro-batch 流过所有 stage）
  DP group: 不同数据（各自处理不同的 mini-batch）

DataLoader 需要按 DP rank 设置不同的 seed/sampler
```

### 9.3 Learning Rate 和 Batch Size

```
Global batch size = micro_batch × num_microbatches × DP_size
  = B_micro × m × DP

例: B_micro=2, m=8, DP=2 → Global batch = 32

学习率调整:
  Linear scaling: lr = base_lr × (global_batch / base_batch)
  Warmup: 建议 2000+ steps
```

## 10. 下一步

- [06_communication_primitives.md](06_communication_primitives.md)：深入理解 AllReduce 等通信原语
- [Lab 05](../labs/05_3d_parallelism/)：3D 并行配置生成器
