# 08 - 推理量化全景

## 核心问题

> 量化在推理中怎么工作？各种方案（GPTQ/AWQ/SmoothQuant/FP8/W4A16）有什么区别？
> 量化对性能和精度的影响是什么？

## 为什么推理需要量化

```
Decode 是 Memory-Bound → 瓶颈在 HBM 带宽

LLaMA-70B FP16:
  模型大小: 140 GB
  H20 带宽: 4 TB/s
  每步 Decode 最快: 140/4000 = 35ms
  
LLaMA-70B INT8:
  模型大小: 70 GB
  每步 Decode 最快: 70/4000 = 17.5ms → 快 2x!

LLaMA-70B INT4:
  模型大小: 35 GB
  每步 Decode 最快: 35/4000 = 8.75ms → 快 4x!

量化的核心价值:
  1. 减少模型大小 → 减少每步读取量 → 降低延迟
  2. 减少 KV Cache 大小 → 更大 batch size → 更高吞吐
  3. 减少显存需求 → 更少 GPU → 降低成本
```

## 量化基础

### 量化公式

```
均匀量化 (Uniform Quantization):

量化:   x_q = round(x / scale + zero_point)
反量化: x_dq = (x_q - zero_point) × scale

其中:
  scale = (x_max - x_min) / (2^bits - 1)
  zero_point = round(-x_min / scale)

示例 (INT8, 对称量化):
  x_float = [-1.5, -0.3, 0.0, 0.7, 1.2]
  scale = max(|x|) / 127 = 1.5 / 127 = 0.0118
  x_q = round(x / 0.0118) = [-127, -25, 0, 59, 102]
  
  反量化: x_dq = x_q × 0.0118 = [-1.499, -0.295, 0, 0.696, 1.204]
  误差很小! (但在极端分布下可能有问题)
```

### 量化粒度

```
┌───────────────────────────────────────────────────────┐
│  量化粒度                                              │
├───────────────────────────────────────────────────────┤
│                                                        │
│  Per-Tensor: 整个 tensor 共享一组 scale/zero_point    │
│    优点: 最少 overhead                                │
│    缺点: 精度最差 (不同通道 range 差异大)              │
│                                                        │
│  Per-Channel: 每个输出通道一组 scale/zero_point       │
│    优点: 精度好 (适应不同通道的分布)                   │
│    缺点: 需要更多存储 scale                           │
│                                                        │
│  Per-Group: 每 group_size 个元素一组 (如 128)         │
│    优点: 精度和效率的最佳平衡                          │
│    缺点: group 越小, overhead 越大                     │
│                                                        │
│  常见选择:                                             │
│    Weight: Per-Channel 或 Per-Group(128)               │
│    Activation: Per-Tensor 或 Per-Token                 │
│    KV Cache: Per-Tensor 或 Per-Channel                 │
│                                                        │
└───────────────────────────────────────────────────────┘
```

## 主流量化方案

### 1. GPTQ — Weight-Only INT4

```
GPTQ (GPT Quantization):
  类型: Weight-Only, Post-Training Quantization (PTQ)
  精度: W4A16 (权重 INT4, 激活 FP16)
  
原理:
  基于 OBQ (Optimal Brain Quantization):
  逐列量化权重, 用 Hessian 信息最小化量化误差
  量化第 i 列时, 调整剩余列来补偿误差

  min ||WX - Q(W)X||²   (最小化输出误差)
  其中 Q(W) 是量化后的权重

特点:
  - 只量化权重 (W4), 激活保持 FP16
  - 需要校准数据 (128-256 samples)
  - 量化过程慢 (几小时), 但一次性
  - Group size 通常 128

性能:
  模型大小: 70B → ~35 GB (INT4) + ~2 GB (scales)
  推理速度: 约 1.5-2x (Decode 阶段)
  精度损失: 较小 (PPL 增加 < 0.5%)

适用:
  - 显存有限, 想跑更大模型
  - 低并发, Decode 密集场景
  
vLLM 使用:
  vllm serve TheBloke/Llama-2-70B-GPTQ \
    --quantization gptq
```

### 2. AWQ — Activation-Aware Weight Quantization

```
AWQ:
  类型: Weight-Only, PTQ
  精度: W4A16
  
核心思想:
  并非所有权重同等重要!
  激活值大的通道 → 对应权重更重要 → 不量化 (或用更高精度)
  
  "保护 1% 的显著权重, 就能大幅减少量化误差"

  步骤:
  1. 用校准数据统计每个通道的激活幅度
  2. 激活大的通道 → scale up 权重 (放大后量化, 减少相对误差)
  3. 然后做标准 INT4 量化
  
  技巧: 不真的 "保留" FP16 权重 (那样需要混合精度 kernel)
       而是 scale up → 量化 → scale down activation
       等价效果, 但 kernel 更简单

性能:
  与 GPTQ 类似: 模型 ~37 GB, 速度 ~1.5-2x
  精度: 通常比 GPTQ 好一点 (尤其在 W3 等极端量化)
  量化速度: 比 GPTQ 快很多 (不需要逐列优化)

vLLM 使用:
  vllm serve TheBloke/Llama-2-70B-AWQ \
    --quantization awq
```

### 3. SmoothQuant — W8A8

```
SmoothQuant:
  类型: Weight + Activation 量化, PTQ
  精度: W8A8 (权重和激活都 INT8)

问题:
  权重分布平滑 → 量化友好
  激活分布有 outlier (异常大的值) → 量化困难
  
  典型: 激活的 99% 在 [-5, 5], 但 1% 在 [-100, 100]
  直接 INT8: scale = 100/127 → 大部分值被量化到 0 → 精度崩塌

SmoothQuant 的解决:
  把激活的"难度"迁移到权重上!
  
  Y = (X × diag(s)^{-1}) × (diag(s) × W)
    = X_smooth × W_scaled
  
  s 是逐通道的平滑因子:
  s_j = max(|X_j|)^α / max(|W_j|)^(1-α)
  α ∈ [0, 1] 控制迁移程度 (通常 0.5)
  
  X_smooth 的 outlier 被缩小 → 量化友好
  W_scaled 的 outlier 被放大 → 但权重本身 range 小, 还是量化友好

性能:
  模型大小: 70B → ~70 GB (INT8)
  推理速度: 可利用 INT8 Tensor Core → 约 1.5x
  精度: 几乎无损 (< 0.1% PPL 增加)
  
适用:
  - 需要高精度保证
  - 硬件支持 INT8 GEMM (所有现代 GPU)
```

### 4. FP8 (E4M3/E5M2)

```
FP8 量化:
  类型: Weight + Activation 量化
  精度: W8A8 (FP8 格式)

FP8 格式:
  E4M3: 4 bit 指数 + 3 bit 尾数 → 范围大, 精度一般
  E5M2: 5 bit 指数 + 2 bit 尾数 → 范围更大, 精度更低
  
  H20 支持 FP8 (E4M3) Tensor Core!

优势 (vs INT8):
  - 不需要校准数据 (可以 dynamic quantize)
  - 不需要 SmoothQuant 等额外处理
  - 浮点格式自然处理 outlier (指数部分)
  - 硬件原生支持 (H100/H20 FP8 Tensor Core)

性能 (H20):
  FP8 FLOPS: 与 INT8 相当
  模型大小: 70B → ~70 GB
  精度: 几乎无损
  额外好处: KV Cache 也可以 FP8 → batch size 翻倍!

vLLM 使用:
  vllm serve meta-llama/Llama-2-70b-hf \
    --quantization fp8 \
    --kv-cache-dtype fp8
```

### 5. W4A16 vs W8A8 vs FP8 对比

```
┌──────────────────────────────────────────────────────────────┐
│            量化方案对比 (LLaMA-70B, H20)                      │
├────────┬────────┬──────────┬──────────┬──────────────────────┤
│ 方案   │ 模型大小│ Decode速度│ 精度损失  │ 适用场景            │
├────────┼────────┼──────────┼──────────┼──────────────────────┤
│ FP16   │ 140 GB │ 基准     │ 无       │ 精度优先             │
│ FP8    │ 70 GB  │ ~1.7x   │ 极小     │ ⭐ 推荐! H20 原生    │
│ INT8   │ 70 GB  │ ~1.5x   │ 极小     │ 通用, 兼容好         │
│ GPTQ   │ 37 GB  │ ~2.0x   │ 小       │ 显存受限             │
│ AWQ    │ 37 GB  │ ~2.0x   │ 较小     │ 显存受限, 精度优先   │
│ INT4   │ 35 GB  │ ~2.5x   │ 中等     │ 极端显存优化         │
├────────┼────────┼──────────┼──────────┼──────────────────────┤
│ 推荐   │ FP8    │ 最佳性价比│ 几乎无损 │ 8×H20 生产环境      │
└────────┴────────┴──────────┴──────────┴──────────────────────┘

对于你的 8×H20:
  FP16: 140GB / 8 = 17.5 GB/GPU → KV Cache 空间充足
  FP8:  70GB / 8 = 8.75 GB/GPU → KV Cache 空间巨大 → batch 更大!
  INT4: 35GB / 8 = 4.4 GB/GPU → 单卡就能跑! 或用更大 batch
```

## KV Cache 量化

```
KV Cache 量化是独立于模型量化的优化:

KV Cache FP16 → FP8:
  - 大小减半
  - 可服务 batch 翻倍
  - 精度影响极小 (< 0.1% PPL)

KV Cache FP16 → INT8:
  - 同上, 但需要量化/反量化 overhead

KV Cache FP16 → INT4:
  - 大小减 75%
  - 有精度损失, 需要评估

vLLM: --kv-cache-dtype fp8  (推荐)
```

## 知识要点框架

### "推理量化有哪些方案？怎么选？"

```
"推理量化主要分两类:

1. Weight-Only (W4A16): GPTQ, AWQ
   - 只量化权重到 INT4, 激活保持 FP16
   - 模型缩小 4x, Decode 加速 2x
   - 适合显存受限或低并发场景

2. Weight + Activation (W8A8): SmoothQuant, FP8
   - 权重和激活都量化到 8bit
   - 可以利用 INT8/FP8 Tensor Core
   - 精度损失更小

选型建议 (8×H20):
   首选 FP8: H20 原生支持, 几乎无损, 性能好
   其次 AWQ INT4: 如果想最大化 batch size
   
另外 KV Cache 也应该用 FP8, 可以让 batch 翻倍"
```

## 小结

| 方案 | 类型 | 精度 | 速度 | 难度 | 推荐度 |
|------|------|------|------|------|--------|
| FP8 | W8A8 | 极好 | 1.7x | 简单 | ⭐⭐⭐⭐⭐ |
| AWQ | W4A16 | 好 | 2.0x | 中等 | ⭐⭐⭐⭐ |
| GPTQ | W4A16 | 好 | 2.0x | 中等 | ⭐⭐⭐ |
| SmoothQuant | W8A8 | 极好 | 1.5x | 复杂 | ⭐⭐⭐ |
