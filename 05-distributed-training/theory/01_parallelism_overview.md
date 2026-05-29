# 01 — 并行策略全景：DP / TP / PP / EP / CP / SP

## 1. 为什么需要并行训练

### 1.1 单卡显存瓶颈

以一个 7B 参数的 LLM 为例，分析训练时的显存占用：

```
模型参数:     7B × 4 bytes (FP32)  = 28 GB
梯度:         7B × 4 bytes (FP32)  = 28 GB
优化器状态:    7B × 8 bytes (Adam)  = 56 GB  (momentum + variance 各 4 bytes)
───────────────────────────────────────────
合计（不含激活值）                    = 112 GB
```

一张 H20（96GB HBM3）连不含激活值的参数都放不下。若用 BF16 混合精度：

```
模型参数:     7B × 2 bytes (BF16)  = 14 GB
梯度:         7B × 2 bytes (BF16)  = 14 GB
优化器状态:    7B × 12 bytes        = 84 GB  (FP32 master weight + momentum + variance)
───────────────────────────────────────────
合计（不含激活值）                    = 112 GB  → 仍然超出单卡
```

注：优化器状态在混合精度下并未减少，因为 Adam 优化器需要维护 FP32 master weights。

### 1.2 计算效率瓶颈

即便模型放得下，单卡训练的时间也是不可接受的：

```
7B 模型, 1T tokens 训练:
  总 FLOPs ≈ 6 × 7B × 1T = 4.2 × 10^22 FLOPs
  H20 BF16 算力 ≈ 148 TFLOPS
  理论训练时间 ≈ 4.2e22 / (148e12) ≈ 2.84 × 10^8 秒 ≈ 9 年

  8 张 H20 (60% MFU):
  实际训练时间 ≈ 9 年 / 8 / 0.6 ≈ 228 天
```

因此分布式训练同时解决**显存不足**和**计算效率**两个核心问题。

## 2. 六大并行策略概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                     分布式训练并行策略                                │
├─────────────┬───────────────┬───────────────┬──────────────────────┤
│  数据并行    │  模型并行      │  序列并行      │  专家并行             │
│  (DP)       │               │  (SP/CP)      │  (EP)               │
│             │  ┌───────┐    │               │                     │
│  DDP        │  │ 张量  │    │  SP: 切分     │  MoE 模型专用        │
│  FSDP       │  │ 并行  │    │  LayerNorm    │  Expert 分布到       │
│  ZeRO       │  │ (TP)  │    │  & Dropout    │  不同 GPU            │
│             │  ├───────┤    │               │                     │
│             │  │ 流水线 │    │  CP: 切分     │  All-to-All 通信     │
│             │  │ 并行  │    │  长序列的      │                     │
│             │  │ (PP)  │    │  Attention     │                     │
│             │  └───────┘    │               │                     │
└─────────────┴───────────────┴───────────────┴──────────────────────┘
```

## 3. 数据并行 (Data Parallelism, DP)

### 核心思想
每张 GPU 持有模型的完整副本，但处理不同的数据 mini-batch。前向/反向后通过 AllReduce 同步梯度。

### 解决什么问题
- **吞吐量**：线性扩展训练速度（理想情况下 N 卡 = N 倍速度）
- **简单易用**：PyTorch DDP 几行代码即可启用

### 局限性
- 每张卡都要放完整模型 → **不解决单卡放不下的问题**
- 梯度 AllReduce 通信量 = 2 × model_size（与卡数无关，Ring AllReduce）
- 全局 batch size 随卡数线性增长，可能影响收敛

### 适用场景
- 模型能放入单卡（< 10B with BF16 mixed precision on H20）
- 大规模数据、高吞吐需求

### 变体
| 变体 | 显存占用 | 通信量 | 备注 |
|------|---------|--------|------|
| DDP | 高（完整模型 × N） | 2M | M = model_size |
| FSDP / ZeRO-3 | 低（model/N） | 3M | 多了 AllGather |

## 4. 张量并行 (Tensor Parallelism, TP)

### 核心思想
将单个 Transformer 层的权重矩阵切分到多张 GPU 上。每张 GPU 只计算部分矩阵乘法，再通过 AllReduce 合并结果。

### Megatron-style 实现

**列并行 (Column Parallel)**：将权重矩阵按列切分
```
W ∈ R^{h×4h}  →  W₁ ∈ R^{h×2h}, W₂ ∈ R^{h×2h}   (TP=2)

GPU 0: Y₀ = X @ W₁    # 输入 X 需要广播（或每卡各有一份）
GPU 1: Y₁ = X @ W₂
# Y₀, Y₁ 是输出的不同列，后续接 GeLU 可以本地做
```

**行并行 (Row Parallel)**：将权重矩阵按行切分
```
W ∈ R^{4h×h}  →  W₁ ∈ R^{2h×h}, W₂ ∈ R^{2h×h}   (TP=2)

GPU 0: Y₀ = X₀ @ W₁   # X₀ 是上一层列并行的本地输出
GPU 1: Y₁ = X₁ @ W₂
Y = AllReduce(Y₀ + Y₁)  # 最终需要 AllReduce
```

### 通信量分析

每个 Transformer 层有 2 次 AllReduce（前向 + 反向各更多，合计约 4 次通信）：
- MLP 块：列并行 → 行并行 = 1 次 AllReduce
- Attention 块：列并行 → 行并行 = 1 次 AllReduce

每次 AllReduce 的数据量 = `batch_size × seq_len × hidden_size × dtype_bytes`

### 解决什么问题
- **单层参数太大**：切分权重矩阵，降低单卡显存
- **减少激活值显存**：每卡只存部分激活

### 局限性
- **频繁通信**：每层都要 AllReduce → 对互联带宽要求极高
- **NVLink 约束**：TP 通常限制在 NVLink 互联的 GPU 内（同一节点）
- 编程复杂度高，需要修改模型代码

### 适用场景
- NVLink 互联的 GPU 组内（H20：8 卡 NVLink，典型 TP=4 或 TP=8）
- 模型 hidden_size 很大（> 4096）

## 5. 流水线并行 (Pipeline Parallelism, PP)

### 核心思想
将模型按层切分为多个 stage，每个 stage 放在不同 GPU 上。数据像流水线一样在 stage 间流动。

### 解决什么问题
- **模型太深**：将不同层放到不同 GPU，降低单卡显存
- **通信量小**：只在 stage 边界做 P2P Send/Recv，数据量 = activation size
- **跨机友好**：P2P 通信对带宽要求低，可走以太网

### 调度算法

```
GPipe:       [F0 F1 F2 F3] [B3 B2 B1 B0]     ← 大 Bubble
1F1B:        [F0 F1] [F2 B0] [F3 B1] [B2 B3]  ← 稳态阶段无 Bubble
Interleaved: 多个 chunk 交替，进一步减少 Bubble
```

### Bubble 率

$$\text{Bubble ratio} = \frac{p - 1}{m + p - 1}$$

其中 p = pipeline stages, m = micro-batches。

- GPipe (m=4, p=4): Bubble = 3/7 ≈ 43%
- 1F1B (m=8, p=4): Bubble = 3/11 ≈ 27%
- 实践中 m >> p 时 Bubble → 0

### 局限性
- Pipeline bubble 导致 GPU 空闲（浪费算力）
- 需要 micro-batch 切分，增加编程复杂度
- 层数不均匀会导致负载不平衡

### 适用场景
- 跨机并行（PP stage 间只需 P2P，带宽需求低）
- 超深模型（100+ 层）

## 6. 序列并行 (Sequence Parallelism, SP)

### 核心思想
将序列维度切分到多张 GPU 上。Megatron-SP 切分的是 LayerNorm 和 Dropout 的输入（这些操作不依赖 hidden 维度切分，但序列维度独立）。

### 解决什么问题
- **激活值显存**：TP 已经切分了权重，但 LayerNorm/Dropout 的激活值仍是完整的
- SP 在 TP 基础上进一步切分激活值，显存节省约 TP_size 倍

### 通信模式
- AllReduce 拆为 ReduceScatter + AllGather
- 总通信量不变，但激活值显存减少

### 适用场景
- 配合 TP 一起使用（Megatron-Core 中默认开启）
- 长序列训练

## 7. 上下文并行 (Context Parallelism, CP)

### 核心思想
将超长序列（128K+）切分到多张 GPU 上，每张 GPU 处理一段 token。Attention 计算时通过 Ring Attention 或 All-to-All 交换 KV。

### 解决什么问题
- **超长序列的激活值爆炸**：Attention 的显存 = O(seq_len²)
- 当 seq_len = 128K 时，单卡放不下 Attention 激活值

### 通信模式
- Ring Attention：KV 在 GPU 间环形传递，每步计算部分 Attention
- 通信量 = O(seq_len × hidden_size / CP_size × CP_size) = O(seq_len × hidden_size)

### 适用场景
- 超长序列训练（128K, 1M tokens）
- 配合 TP + SP 使用

## 8. 专家并行 (Expert Parallelism, EP)

### 核心思想
MoE（Mixture of Experts）模型中，不同 Expert 放在不同 GPU 上。Router 决定 token 发送到哪个 Expert，通过 All-to-All 通信分发。

### 解决什么问题
- **MoE 的参数量**：MoE 模型总参数量大（如 8 个 Expert → 8 倍 FFN 参数）
- 每个 token 只激活少数 Expert → 计算量可控，但参数需要分布

### 通信模式
- All-to-All：Token dispatch + Token combine
- 通信量取决于 Expert 数量和 top-k

### 适用场景
- MoE 架构模型（Mixtral, DeepSeek-MoE）
- 大量 GPU 集群（EP 可跨机）

## 9. 并行策略对比总结

| 策略 | 切分维度 | 通信原语 | 通信量 | 带宽要求 | 显存节省 | 编程复杂度 |
|-----|---------|---------|--------|---------|---------|-----------|
| DP  | batch   | AllReduce | 2M | 低 | 无 | ★ |
| FSDP| batch + params | AllGather + ReduceScatter | 3M | 中 | 高 | ★★ |
| TP  | hidden  | AllReduce | ~4×B×S×H per layer | 高(NVLink) | 中 | ★★★ |
| PP  | layer   | P2P | B×S×H per stage | 低 | 高 | ★★★ |
| SP  | sequence (LN/DO) | AllGather + ReduceScatter | 同TP | 高(NVLink) | 中 | ★★ |
| CP  | sequence (Attn) | Ring/All-to-All | S×H | 中 | 高 | ★★★★ |
| EP  | expert  | All-to-All | 取决于路由 | 中 | 高 | ★★★★ |

> M = model_size, B = batch, S = seq_len, H = hidden_size

## 10. 8 × H20 环境下的典型配置

### 配置一：7B 模型训练
```
TP=4, PP=1, DP=2
- 4 张卡做 TP（NVLink 组内）
- 2 组做 DP
- 适合: 模型不大，优先提升吞吐
```

### 配置二：13B 模型训练
```
TP=4, PP=2, DP=1
- 4 张卡做 TP
- 2 个 PP stage
- 适合: 单卡放不下，需要 PP 切分层
```

### 配置三：7B 模型 + ZeRO-3
```
ZeRO-3 on 8 GPUs (等效 FSDP)
- 不需要 TP/PP
- 参数/梯度/优化器三切分
- 适合: 简单部署，显存优化好
```

### 为什么 TP 放在 NVLink 内？

```
NVLink 带宽: ~450 GB/s (H20 bidirectional)
PCIe 5.0 带宽: ~64 GB/s
网络(IB/RoCE): ~25-50 GB/s

TP 每层 AllReduce: 高频 + 大数据量 → 需要最高带宽
PP P2P Send/Recv: 低频 + 小数据量 → 可用低带宽
DP AllReduce: 一步一次 + 可 overlap → 可用网络
```

## 11. 下一步

- [02_data_parallelism.md](02_data_parallelism.md)：深入 DDP 和 FSDP 原理
- [Lab 01](../labs/01_ddp_basics/)：从单卡到 DDP 的实战
