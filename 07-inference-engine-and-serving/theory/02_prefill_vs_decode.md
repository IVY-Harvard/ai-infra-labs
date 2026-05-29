# 02 - Prefill vs Decode 深度分析

## 核心问题

> 为什么同一个模型，处理 prompt 和生成 token 的计算特征完全不同？

这个问题的答案决定了整个推理系统的架构设计。

## 计算模式对比

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│         Prefill (Prompt Processing)                              │
│         ═══════════════════════════                              │
│                                                                  │
│         Input: [tok₁, tok₂, ..., tokₙ]  (N tokens 并行)        │
│                                                                  │
│         ┌──────────────────────┐                                │
│         │  Matrix × Matrix     │  (N,d) × (d,d) = GEMM         │
│         │  ████████████████    │  GPU 核心全部忙碌              │
│         │  ████████████████    │  Compute Utilization: HIGH     │
│         └──────────────────────┘                                │
│                                                                  │
│─────────────────────────────────────────────────────────────────│
│                                                                  │
│         Decode (Token Generation)                                │
│         ═════════════════════════                                │
│                                                                  │
│         Input: [tok_new]  (1 token)                             │
│                                                                  │
│         ┌──────────────────────┐                                │
│         │  Vector × Matrix     │  (1,d) × (d,d) = GEMV         │
│         │  ─                   │  绝大部分核心在等数据          │
│         │                      │  Compute Utilization: LOW      │
│         └──────────────────────┘                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Roofline 模型分析

### 什么是 Roofline 模型？

```
Performance (FLOPS/s)
    │
    │         ╱─────────────── Peak Compute (148 TFLOPS)
    │        ╱
    │       ╱
    │      ╱
    │     ╱
    │    ╱   ← Memory-Bound    Compute-Bound →
    │   ╱         Region            Region
    │  ╱
    │ ╱
    │╱
    └──────────────────────────────── Arithmetic Intensity (FLOPs/Byte)
                    ^
                    │
              Ridge Point = Peak_Compute / Peak_Bandwidth
                         = 148 TFLOPS / 4 TB/s
                         = 37 FLOPs/Byte (H20)
```

### H20 GPU Roofline 参数
```
NVIDIA H20:
  - FP16 Peak: 148 TFLOPS
  - HBM3 Bandwidth: 4.0 TB/s
  - HBM Capacity: 96 GB
  - Ridge Point: 148T / 4T = 37 FLOPs/Byte
  
对比 H100:
  - FP16 Peak: 989 TFLOPS  
  - HBM3 Bandwidth: 3.35 TB/s
  - Ridge Point: 989T / 3.35T = 295 FLOPs/Byte
  
注意: H20 的 Ridge Point 更低，意味着更容易达到 Compute-Bound！
      H20 的优势在于大显存(96GB)和高带宽(4TB/s)
```

## Prefill 阶段 Roofline 分析

### Self-Attention 中的 Q×K^T

```python
# Q: (batch, heads, seq_len, head_dim) 
# K: (batch, heads, seq_len, head_dim)
# Q×K^T: (batch, heads, seq_len, head_dim) × (batch, heads, head_dim, seq_len)
#       → (batch, heads, seq_len, seq_len)

# 对于单个 head:
# FLOPs = 2 × seq_len × head_dim × seq_len = 2 × N² × d_h
# Bytes = (N × d_h + N × d_h + N × N) × 2  (FP16)
#       = (2N × d_h + N²) × 2

# Arithmetic Intensity:
# AI = 2N²d_h / ((2Nd_h + N²) × 2)
#    ≈ 2N²d_h / (2N² × 2)  (当 N >> d_h)
#    = d_h / 2 = 64 (head_dim=128)
```

### FFN 层的 GEMM

```python
# FFN: x @ W1 → intermediate → x @ W2
# W1: (d_model, 4×d_model), W2: (4×d_model, d_model)
# x: (seq_len, d_model)

# 对于 x @ W1:
# FLOPs = 2 × seq_len × d_model × 4×d_model
# Bytes = (seq_len × d_model + d_model × 4×d_model + seq_len × 4×d_model) × 2

# AI = 2 × N × d × 4d / ((Nd + 4d² + 4Nd) × 2)
#    ≈ 2 × N × 4d² / (4d² × 2)  (当 d >> N，权重主导)
#    = N (seq_len)

# 对于 seq_len=2048: AI = 2048 >> 37 → 完全 Compute-Bound ✓
```

### Prefill 总结

```
┌────────────────────────────────────────────────────────────┐
│ Prefill 是 Compute-Bound 的原因:                           │
│                                                            │
│ 1. 操作是 GEMM (矩阵×矩阵)                                │
│ 2. seq_len 维度提供了大量并行 (数千个 token)                │
│ 3. Arithmetic Intensity = seq_len 级别 (数百~数千)          │
│ 4. 远超 Ridge Point (37 FLOPs/Byte on H20)                │
│                                                            │
│ 优化方向:                                                   │
│ - 增加 GPU 算力 (FLOPS)                                    │
│ - FlashAttention (减少 O(N²) 的 HBM 读写)                 │
│ - Tensor Parallelism (多卡分摊计算)                        │
│ - Chunked Prefill (分块处理超长 prompt)                    │
└────────────────────────────────────────────────────────────┘
```

## Decode 阶段 Roofline 分析

### 为什么 Decode 每步只处理 1 个 token？

```
自回归生成的本质:
  
  P(y₁, y₂, ..., yₜ | x) = ∏ P(yᵢ | x, y₁, ..., yᵢ₋₁)

每个 token 依赖前面所有 token → 无法并行生成
每步只能生成 1 个新 token → 矩阵退化为向量
```

### Decode 的 GEMV

```python
# 线性层: x @ W
# x: (1, d_model)  ← 只有 1 个 token!
# W: (d_model, d_model)

# FLOPs = 2 × 1 × d × d = 2d²
# Bytes = (d + d² + d) × 2 ≈ 2d² bytes (权重主导)

# Arithmetic Intensity:
# AI = 2d² / (2d²) = 1 FLOPs/Byte  ← 极低！

# 1 << 37 (Ridge Point) → 完全 Memory-Bound ✓
```

### Decode Attention 部分

```python
# 新 token 的 q 需要和所有历史 KV 做 attention
# q: (1, d_h)
# K_cache: (context_len, d_h)  ← 从 KV Cache 读取
# V_cache: (context_len, d_h)

# FLOPs = 2 × 1 × context_len × d_h × 2 = 4 × L × d_h
# Bytes = (d_h + L×d_h + L×d_h + d_h) × 2 ≈ 4L×d_h bytes

# AI = 4Ld_h / (4Ld_h) = 1 FLOPs/Byte  ← 还是极低！

# 而且 KV Cache 会随生成长度增长，带宽压力越来越大
```

### Decode 总结

```
┌────────────────────────────────────────────────────────────┐
│ Decode 是 Memory-Bound 的原因:                             │
│                                                            │
│ 1. 操作是 GEMV (矩阵×向量)                                │
│ 2. 只有 1 个 token，无法利用 GPU 大规模并行                 │
│ 3. Arithmetic Intensity ≈ 1 FLOPs/Byte                    │
│ 4. 远低于 Ridge Point (37 FLOPs/Byte on H20)              │
│                                                            │
│ GPU 利用率: 实际计算力 / 峰值计算力                         │
│   = AI × Bandwidth / Peak_Compute                          │
│   = 1 × 4TB/s / 148T = 2.7% !!!                          │
│                                                            │
│ 97% 的时间 GPU 在等数据从 HBM 传到 SM！                    │
│                                                            │
│ 优化方向:                                                   │
│ - Batching (增加有效 AI)                                   │
│ - 量化 (减少需要读取的 bytes)                               │
│ - 减少 KV Cache 读取 (GQA/MQA)                            │
│ - 提高带宽 (HBM3E, 多卡)                                  │
└────────────────────────────────────────────────────────────┘
```

## Batching 如何改变 Decode 的性质

```
Batch Size = B 时:

x: (B, d_model)
W: (d_model, d_model)  ← 权重只需读一次！

FLOPs = 2 × B × d² 
Bytes = (B×d + d² + B×d) × 2 ≈ 2d² (当 d >> B)

AI = 2Bd² / 2d² = B FLOPs/Byte

┌─────────────────────────────────────────────────────┐
│  Batch Size vs Arithmetic Intensity                  │
│                                                      │
│  B=1:   AI=1    → 2.7% GPU utilization              │
│  B=4:   AI=4    → 10.8% GPU utilization             │
│  B=16:  AI=16   → 43.2% GPU utilization             │
│  B=37:  AI=37   → ~100% GPU utilization (Ridge!)    │
│  B=64:  AI=64   → Compute-Bound (saturated)         │
│                                                      │
│  H20 的 Ridge Point 是 37                            │
│  所以 batch=37 就能让 GPU 满载                       │
│  (实际因为 KV Cache 显存限制，                        │
│   未必能到这么大的 batch)                             │
└─────────────────────────────────────────────────────┘
```

### 关键洞察

```
                    延迟 (每 token)        吞吐 (tokens/s)
                    ─────────────         ────────────────
Batch=1:           35 ms                  28
Batch=8:           36 ms (+3%)           222 (8x!)
Batch=32:          38 ms (+8%)           842 (30x!)
Batch=64:          45 ms (+28%)          1422 (50x!)

结论: Batch 增大时:
  - 延迟几乎不变 (因为额外计算被并行消化)
  - 吞吐线性增长 (直到 Compute-Bound 或显存不够)
  
这就是 Continuous Batching 的价值所在！
```

## Prefill-Decode Disaggregation（分离部署）

### 问题
```
在同一个 GPU 上混合运行 Prefill 和 Decode:

时间线:
  ──────────────────────────────────────────►
  │ Decode │ Prefill ████│ Decode │ Decode │

问题:
  1. Prefill 会阻塞 Decode (大 GEMM 抢占 GPU)
  2. TPOT 产生抖动 (Prefill 期间 Decode 停滞)
  3. Decode 的低利用率浪费了 Prefill 的高算力需求
```

### 解决方案: P/D 分离

```
┌────────────────────────────────────────────────────────┐
│                                                         │
│   Prefill Node (高算力 GPU)     Decode Node (高带宽 GPU) │
│   ┌─────────────────┐          ┌─────────────────┐     │
│   │ Compute-Optimized│   KV    │ Bandwidth-Optimized│   │
│   │ 全力做 Prefill   │ ──────▶ │ 专注做 Decode     │   │
│   │ H100/A100       │ Transfer │ H20 (大显存!)     │   │
│   └─────────────────┘          └─────────────────┘     │
│                                                         │
│   优势:                                                  │
│   - Prefill 不被 Decode 打断                            │
│   - Decode 不被 Prefill 阻塞                            │
│   - 各自优化自己的瓶颈                                   │
│   - H20 的 96GB 大显存适合 Decode (存更多 KV Cache)      │
│                                                         │
└────────────────────────────────────────────────────────┘
```

### H20 在 Decode 中的优势

```
H20 vs H100 for Decode:

H20:  HBM = 96GB, BW = 4.0 TB/s, FLOPS = 148T
H100: HBM = 80GB, BW = 3.35 TB/s, FLOPS = 989T

对于 Decode (Memory-Bound):
  性能 ∝ Bandwidth (不是 FLOPS!)
  
  H20 Decode 性能 / H100 Decode 性能 
  = 4.0 / 3.35 = 1.19x  (H20 反而更快!)
  
  而且 H20 显存更大: 96GB vs 80GB
  → 能服务更长上下文或更大 batch

这就是为什么 H20 适合推理工作负载！
```

## Chunked Prefill

### 问题
超长 prompt (如 128K) 的 Prefill 会:
1. 长时间霸占 GPU
2. Attention 的 O(N²) 内存需求爆炸
3. 阻塞其他请求的 Decode

### 解决方案
```
原始: Prefill 128K tokens 一次性处理 (可能需要 10+ 秒)
分块: 每次处理 chunk_size=8192 tokens

┌──────────────────────────────────────────────┐
│ 时间线对比:                                    │
│                                               │
│ 无分块:                                       │
│ │████████████████ Prefill 128K █████████████│ │
│ │                                           │ │
│ 有分块:                                       │
│ │██P1██│D│██P2██│D│██P3██│D│...│██P16█│D│   │
│                                               │
│ D = 穿插的 Decode steps                       │
│ → Decode 请求的 TPOT 不会被长 Prefill 阻塞    │
└──────────────────────────────────────────────┘
```

## 知识要点框架

### "请解释 Prefill 和 Decode 的区别"

```
答题框架:

1. 计算模式不同:
   - Prefill: GEMM (矩阵×矩阵), Compute-Bound
   - Decode: GEMV (矩阵×向量), Memory-Bound

2. 原因:
   - Prefill 处理 N 个 token, AI ≈ N >> Ridge Point
   - Decode 只处理 1 个 token, AI ≈ 1 << Ridge Point

3. 优化方向不同:
   - Prefill: 提高算力利用 (FlashAttention, TP)
   - Decode: 提高带宽利用 (Batching, 量化, GQA)

4. 实际影响:
   - Prefill → TTFT (首 token 延迟)
   - Decode → TPOT (后续 token 延迟)
   - Batching 对 Decode 效果显著 (线性提升吞吐)

5. 进阶: P/D 分离部署
   - 算力强的卡做 Prefill
   - 带宽大/显存大的卡 (如 H20) 做 Decode
```

### "为什么 Batch 能提升 Decode 吞吐但几乎不增加延迟？"

```
答题框架:

1. Decode 是 Memory-Bound:
   瓶颈在于读取模型权重 (140GB for 70B)
   
2. Batch 不增加读取量:
   权重只需读一次, 就能同时计算 B 个请求的结果
   
3. 有效 AI 线性增长:
   AI = B → batch=32 时 AI=32, 接近 Ridge Point
   
4. 延迟几乎不变:
   额外的计算被 GPU 并行单元消化
   (直到 Compute-Bound 或 KV Cache 显存不够)
```

## 量化数据验证 (H20 实测参考)

```
Model: LLaMA-2-70B on 8×H20 (TP=8)

┌────────────────────────────────────────────────────┐
│  Prompt Length  │  TTFT (ms)  │  理论下界 (ms)     │
├────────────────────────────────────────────────────┤
│      128        │     45      │     35             │
│      512        │     85      │     60             │
│     2048        │    250      │    180             │
│     8192        │    850      │    600             │
│    32768        │   3200      │   2400             │
└────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────┐
│  Batch Size  │  TPOT (ms)  │  Throughput (tok/s)  │
├────────────────────────────────────────────────────┤
│      1       │    42       │     24               │
│      4       │    43       │     93               │
│     16       │    45       │    356               │
│     32       │    48       │    667               │
│     64       │    55       │   1164               │
│    128       │    72       │   1778               │
└────────────────────────────────────────────────────┘

观察:
- TTFT 与 prompt 长度近似线性 (FlashAttention 避免了 O(N²))
- TPOT 随 batch 增加缓慢增长
- 吞吐与 batch 接近线性增长 (直到显存或计算饱和)
```

## 小结

| 维度 | Prefill | Decode |
|------|---------|--------|
| 输入 | N tokens (并行) | 1 token (串行) |
| 操作 | GEMM (矩阵×矩阵) | GEMV (矩阵×向量) |
| 瓶颈 | Compute (FLOPS) | Memory (Bandwidth) |
| AI | ~N (数百~数千) | ~1 |
| GPU 利用率 | 高 (>80%) | 极低 (~3% for B=1) |
| 延迟指标 | TTFT | TPOT |
| 优化 | FlashAttention, TP | Batching, 量化, GQA |
| 扩展 | Chunked Prefill | Continuous Batching |
