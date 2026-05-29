# 04 — 流水线并行：GPipe / 1F1B / Interleaved 调度

## 1. 流水线并行基本原理

### 1.1 层切分

将 Transformer 的 L 层切分为 p 个 stage：

```
模型: [Layer 0, Layer 1, ..., Layer 31]   (L=32)
PP=4: Stage 0: [Layer 0-7]    → GPU 0
      Stage 1: [Layer 8-15]   → GPU 1
      Stage 2: [Layer 16-23]  → GPU 2
      Stage 3: [Layer 24-31]  → GPU 3
```

### 1.2 为什么需要 micro-batch

如果直接用一整个 batch 做流水线：

```
时间 →
GPU 0: [Forward Stage 0] [等待...等待...等待...] [Backward Stage 0]
GPU 1: [空闲] [Forward Stage 1] [等待...] [Backward Stage 1] [空闲]
GPU 2: [空闲] [空闲] [Forward Stage 2] [Backward Stage 2] [空闲] [空闲]
GPU 3: [空闲] [空闲] [空闲] [F+B Stage 3] [空闲] [空闲] [空闲]

空闲比例极高！
```

解决方案：将一个 mini-batch 切分为 m 个 micro-batch，让多个 micro-batch 同时在不同 stage 中流动。

## 2. GPipe 调度

### 2.1 算法

GPipe 将所有前向和所有反向分成两个阶段：

```
Phase 1: 所有 micro-batch 做前向
Phase 2: 所有 micro-batch 做反向

m=4, p=4:
时间步:    1    2    3    4    5    6    7    8    9   10   11   12
GPU 0:  [F0] [F1] [F2] [F3] [##] [##] [##] [B3] [B2] [B1] [B0] [Update]
GPU 1:       [F0] [F1] [F2] [F3] [##] [##] [B3] [B2] [B1] [B0] [Update]
GPU 2:            [F0] [F1] [F2] [F3] [##] [B3] [B2] [B1] [B0] [Update]
GPU 3:                 [F0] [F1] [F2] [F3] [B3] [B2] [B1] [B0] [Update]

F = Forward, B = Backward, ## = Bubble
```

### 2.2 Bubble 分析

```
总时间步: Forward flush: (p-1) + m 步
          Backward flush: m + (p-1) 步
          但 Forward 和 Backward 之间有 (p-1) 步 Bubble

计算时间: m × (tF + tB) per GPU
Bubble 时间: (p-1) × (tF + tB)

Bubble ratio = (p-1) × (tF + tB) / [m × (tF + tB) + (p-1) × (tF + tB)]
             = (p-1) / (m + p - 1)

例: p=4, m=4:  Bubble = 3/7 ≈ 43%  → 很大！
    p=4, m=8:  Bubble = 3/11 ≈ 27%
    p=4, m=16: Bubble = 3/19 ≈ 16%
    p=4, m=32: Bubble = 3/35 ≈ 8.6%
```

### 2.3 GPipe 的显存问题

```
GPipe 需要保存所有 micro-batch 的激活值（从前向到反向）:

Peak activation memory = m × (单个 micro-batch 的激活值)

m=16 时，需要保存 16 份激活值 → 显存压力巨大
→ GPipe 通常需要 activation recomputation
```

## 3. 1F1B (One Forward One Backward) 调度

### 3.1 算法

1F1B 的核心思想：在 warmup 后，每做一次前向就做一次反向（稳态阶段）。

```
m=8, p=4:

Phase 1 (Warmup): 前 p-1 个 micro-batch 做前向填充流水线
Phase 2 (Steady): 交替执行 1F + 1B
Phase 3 (Cooldown): 最后 p-1 个 micro-batch 做反向清空流水线

时间步:
GPU 0: [F0][F1][F2][F3][B0][F4][B1][F5][B2][F6][B3][F7][B4][B5][B6][B7]
GPU 1:     [F0][F1][F2]     [B0][F3][B1][F4][B2][F5][B3][F6][B4][B5][B6][B7]
GPU 2:         [F0][F1]          [B0][F2][B1][F3][B2][F4][B3][F5][B4][B5][B6][B7]
GPU 3:             [F0]               [B0][F1][B1][F2][B2][F3][B3][F4][B4][B5][B6][B7]

     ↑ Warmup ↑          ↑ Steady State ↑              ↑ Cooldown ↑
```

### 3.2 显存优势

```
1F1B 的激活值峰值:

GPU 0 (第一个 stage): 需要同时保存 p 个 micro-batch 的激活值
  (warmup 了 p-1 个，加上 steady 中即将反向的 1 个)

GPU i (第 i 个 stage): 需要保存 p-i 个 micro-batch 的激活值

Peak activation = p × (单个 micro-batch 激活值)  (在第一个 stage)

对比 GPipe: m × (单个 micro-batch 激活值)
当 m >> p 时，1F1B 大幅节省显存！

例: m=32, p=4
  GPipe: 32 份激活值
  1F1B:  4 份激活值 → 节省 8x
```

### 3.3 Bubble 分析

```
1F1B 的 Bubble:
  Warmup 阶段: GPU p-1 等待 p-1 步
  Cooldown 阶段: GPU 0 等待 p-1 步

总 Bubble = (p-1) × (tF + tB)  ← 与 GPipe 相同！

Bubble ratio = (p-1) / (m + p - 1)  ← 公式相同

但 1F1B 的优势是显存，不是 Bubble 率。
```

### 3.4 非交错 1F1B 的完整推导

```
设: tF = 前向一个 micro-batch 的时间
    tB = 反向一个 micro-batch 的时间
    通常 tB ≈ 2×tF (反向约为前向的 2 倍)

GPU 0 的时间线:
  Warmup: (p-1) × tF
  Steady: m - (p-1) 个 (tF + tB) 对
  Cooldown: (p-1) × tB
  
  总时间 = (p-1)×tF + [m-(p-1)]×(tF+tB) + (p-1)×tB
         = (p-1)(tF+tB) + m(tF+tB) - (p-1)(tF+tB)
         = m(tF+tB) + (p-1)(tF+tB) ← 等等，让我重新算

  实际总时间 (pipeline flush):
  T = m×tF + m×tB + (p-1)×(tF+tB) ← 不对

正确分析:
  GPU 0 处理 m 个 micro-batch 的总计算时间 = m × (tF + tB)
  GPU 0 的 Bubble = (p-1) × tF  (warmup 等最后一个 stage 开始反向)
                  ← 实际上 (p-1) 步是因为 pipeline depth

  Bubble = (p-1) 个时间单位 × (tF + tB) ÷ 实际时间
  
简化: Bubble ratio = (p-1) / (m + p - 1)
```

## 4. Interleaved 1F1B 调度

### 4.1 核心思想

将每个 GPU 分配多个非连续的 stage（chunks），而不是一个连续的 stage：

```
Non-interleaved (p=4, 每 GPU 1 个 stage):
  GPU 0: Layers 0-7
  GPU 1: Layers 8-15
  GPU 2: Layers 16-23
  GPU 3: Layers 24-31

Interleaved (p=4, v=2 chunks per GPU):
  GPU 0: Layers 0-3, 16-19    (chunk 0, chunk 4)
  GPU 1: Layers 4-7, 20-23    (chunk 1, chunk 5)
  GPU 2: Layers 8-11, 24-27   (chunk 2, chunk 6)
  GPU 3: Layers 12-15, 28-31  (chunk 3, chunk 7)
```

### 4.2 Bubble 率改进

```
Interleaved with v chunks per GPU:

Virtual pipeline stages = p × v
有效 micro-batch 数 = m (不变)

Bubble ratio = (p-1) / (m×v + p - 1)

例: p=4, m=8, v=2:
  Non-interleaved: 3/(8+3) = 27%
  Interleaved:     3/(16+3) = 16%

例: p=4, m=8, v=4:
  Interleaved:     3/(32+3) = 8.6%
```

### 4.3 通信代价

```
Interleaved 的额外通信:
  Non-interleaved: 每个 micro-batch 经过 p-1 次 P2P (线性)
  Interleaved (v chunks): 每个 micro-batch 经过 p×v - 1 次 P2P

  通信量增加: v 倍
  
  但 P2P 通信量小 (B×S×H per micro-batch)，且可以与计算重叠
  → 当 NVLink 带宽足够时，额外通信代价可接受
```

### 4.4 调度示例

```
p=4, m=4, v=2 (8 virtual stages):

时间步:     1   2   3   4   5   6   7   8   9   10  11  12  13  14
GPU 0:   [F00][F01][F02][F03]   [F04][F05][F06][F07]   [B07][B06]...
GPU 1:       [F00][F01][F02][F03]   [F04][F05][F06][F07]   [B07]...
GPU 2:           [F00][F01][F02][F03]   [F04][F05][F06][F07]   ...
GPU 3:               [F00][F01][F02][F03]   [F04][F05][F06][F07]...

FXY = Forward, micro-batch X, chunk Y
```

## 5. 高级调度算法

### 5.1 Zero Bubble Pipeline (ZB-H1, ZB-H2)

核心思想：利用 Warmup 阶段的空闲做 backward-W（权重梯度）而不是完整 backward。

```
将 Backward 分为:
  B = backward-input (∂L/∂X) + backward-weight (∂L/∂W)
  
  B_input 必须立即做（下一个 stage 等着梯度）
  B_weight 可以延迟（只在 optimizer step 前需要）

Zero Bubble 调度:
  Warmup 期间: 前向 + 延迟的 B_weight
  Steady: 1F + 1B_input + 1B_weight (交替)
  
  理论上可以实现 Bubble → 0
```

### 5.2 Bubble 率对比

```
调度算法          Bubble 率                     显存峰值
─────────────────────────────────────────────────────────────
GPipe            (p-1)/(m+p-1)                 m × activation
1F1B             (p-1)/(m+p-1)                 p × activation
Interleaved      (p-1)/(m×v+p-1)              p × activation
Zero Bubble      ≈ 0                           p × activation (+ extra)
```

## 6. Pipeline Parallel 的通信分析

### 6.1 P2P 通信量

```
每个 micro-batch 在 stage 之间传递的数据:
  Activation: B_micro × S × H × dtype_bytes

  B_micro=1, S=2048, H=4096, BF16:
  = 1 × 2048 × 4096 × 2 = 16 MB per micro-batch

每步总 P2P 通信:
  Non-interleaved: m × 16 MB × 2 (fwd + bwd) = 32m MB
  例 m=8: 256 MB per step → 很小！
```

### 6.2 与 TP 通信量对比

```
TP (每层 4 次 AllReduce):
  4 × L × B × S × H × 2 × 2(N-1)/N
  = 4 × 32 × 4 × 2048 × 4096 × 2 × 1.5
  ≈ 12 GB per step

PP (P2P):
  ≈ 256 MB per step

PP 通信量比 TP 小 50x！这就是为什么 PP 可以走低带宽链路
```

## 7. 负载均衡

### 7.1 均匀切分的问题

```
Transformer 各层计算量相同，但:
  - 第一个 stage 有 Embedding 层（计算少但参数大）
  - 最后一个 stage 有 LM Head（参数大）
  - 第一个/最后一个 stage 可能比中间 stage 更重

不均衡导致:
  - 快的 stage 空闲等待慢的 stage → 额外 Bubble
```

### 7.2 解决方案

```
1. 手动调整层数分配:
   Stage 0: Embedding + 6 layers
   Stage 1: 8 layers
   Stage 2: 8 layers
   Stage 3: 8 layers + LM Head + Loss

2. 自动均衡 (基于 profiling):
   测量每层的前向/反向时间，贪心分配使各 stage 时间相近

3. Interleaved 天然更均衡:
   每个 GPU 有多个 chunk，分散了不均衡的影响
```

## 8. Pipeline 中的 Batch 语义

### 8.1 Gradient Accumulation

```
Pipeline Parallel 自然涉及 gradient accumulation:
  Global batch = m × micro_batch_size × DP_size

  每个 step:
  1. 处理 m 个 micro-batch
  2. 累积梯度
  3. 一次 optimizer step

  这等效于 gradient accumulation steps = m
```

### 8.2 对 Learning Rate 的影响

```
Effective batch size = m × B_micro × DP
  
  例: m=8, B_micro=2, DP=2
  Effective batch = 8 × 2 × 2 = 32

  Learning rate 需要按照 effective batch size 调整
  (linear scaling rule: lr ∝ batch_size)
```

## 9. 8×H20 上的 PP 配置

### 9.1 TP=4, PP=2

```
GPU [0,1,2,3] → TP Group → Stage 0 (Layers 0-15)
GPU [4,5,6,7] → TP Group → Stage 1 (Layers 16-31)

通信:
  TP: AllReduce within {0,1,2,3} 和 {4,5,6,7} (NVLink, 高带宽)
  PP: P2P between GPU 3→4 (或 NVSwitch, 低带宽足够)
  
推荐 micro-batches: m ≥ 8 (Bubble = 1/9 ≈ 11%)
```

### 9.2 为什么不用 PP=4, TP=2？

```
PP=4, TP=2:
  TP Groups: {0,1}, {2,3}, {4,5}, {6,7}
  PP Stages: Stage 0→1→2→3

  Bubble = (4-1)/(m+3) = 3/(m+3)
  m=8 时: 3/11 = 27%  vs  PP=2 的 11%

  TP=2 时 NVLink 利用率也不够高

  → PP 越少越好（减少 Bubble），TP 用尽 NVLink
```

## 10. 实践中的 Pipeline 优化

```
1. Micro-batch size 选择:
   - m 越大, Bubble 越小
   - 但 m 大 → 梯度累积多 → global batch 大 → 可能影响收敛
   - 经验: m ≥ 4p 使 Bubble < 20%

2. Activation Checkpointing:
   - PP 中每个 stage 只有 L/p 层，激活值相对少
   - 可以选择性地对部分层做 checkpoint

3. 通信重叠:
   - PP 的 P2P 可以与计算重叠
   - 在一个 micro-batch 做反向时，下一个 micro-batch 的激活值已经传过来了

4. 非均匀 micro-batch size:
   - 某些调度允许第一个/最后一个 micro-batch 用更小的 size
   - 减少 warmup/cooldown 时间
```

## 11. 下一步

- [05_3d_parallelism.md](05_3d_parallelism.md)：将 TP+PP+DP 组合起来
- [Lab 04](../labs/04_pipeline_parallelism/)：实现 GPipe 和 1F1B 调度
