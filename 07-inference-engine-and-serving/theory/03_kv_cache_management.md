# 03 - KV Cache 管理：推理系统的核心瓶颈

## 核心问题

> KV Cache 是什么？为什么它是推理系统的核心瓶颈？它的显存占用怎么算？

## KV Cache 的本质

### 为什么需要 KV Cache？

```
自回归生成的朴素实现:

Step 1: input = [A, B, C]      → 计算 attention → output D
Step 2: input = [A, B, C, D]   → 重新计算 attention → output E  
Step 3: input = [A, B, C, D, E] → 又重新计算 attention → output F

问题: 每一步都重复计算历史 token 的 K 和 V！
     Step t 的计算量 = O(t × d²)  (每步都算所有历史的 K,V)
     总计算量 = O(T² × d²)  ← 灾难！

有了 KV Cache:

Step 1: 计算 A,B,C 的 K,V → 存入 Cache → output D
Step 2: 只计算 D 的 K,V → 拼接 Cache → output E
Step 3: 只计算 E 的 K,V → 拼接 Cache → output F

每步只做 1 个 token 的 K,V 计算: O(d²)
总计算量 = O(T × d²)  ← 线性！节省 T 倍计算
```

### KV Cache 的结构

```
┌─────────────────────────────────────────────────────────────┐
│                     KV Cache 结构                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Layer 0:  K: [seq_len, num_kv_heads, head_dim]             │
│            V: [seq_len, num_kv_heads, head_dim]             │
│                                                              │
│  Layer 1:  K: [seq_len, num_kv_heads, head_dim]             │
│            V: [seq_len, num_kv_heads, head_dim]             │
│                                                              │
│  ...                                                         │
│                                                              │
│  Layer L:  K: [seq_len, num_kv_heads, head_dim]             │
│            V: [seq_len, num_kv_heads, head_dim]             │
│                                                              │
│  每生成一个新 token → 每层的 K,V 各增加一行                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## KV Cache 大小计算公式

### 通用公式

```
KV_Cache_Size (per token, per request) = 
    2 × num_layers × num_kv_heads × head_dim × dtype_bytes

KV_Cache_Size (total) = 
    2 × L × n_kv × d_h × dtype × seq_len × batch_size

其中:
  2         = K 和 V 各一份
  L         = 层数
  n_kv      = KV head 数量 (GQA/MQA 时 < num_heads)
  d_h       = head_dim (通常 128)
  dtype     = 数据类型字节数 (FP16=2, FP8=1)
  seq_len   = 序列长度 (prompt + generated)
  batch_size = 并发请求数
```

### 实际模型计算

```
┌──────────────────────────────────────────────────────────────────┐
│              常见模型 KV Cache 计算                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  LLaMA-2-7B:                                                     │
│    L=32, n_kv=32, d_h=128, dtype=2 (FP16)                       │
│    Per token: 2×32×32×128×2 = 524,288 bytes = 0.5 MB            │
│    Seq_len=2048: 0.5MB × 2048 = 1 GB                            │
│    Batch=32: 1GB × 32 = 32 GB !!!                               │
│                                                                   │
│  LLaMA-2-70B:                                                    │
│    L=80, n_kv=8 (GQA), d_h=128, dtype=2 (FP16)                  │
│    Per token: 2×80×8×128×2 = 327,680 bytes = 0.3 MB             │
│    Seq_len=4096: 0.3MB × 4096 = 1.25 GB                         │
│    Batch=32: 1.25GB × 32 = 40 GB                                │
│                                                                   │
│  LLaMA-3-70B:                                                    │
│    L=80, n_kv=8 (GQA), d_h=128, dtype=2                         │
│    与 LLaMA-2-70B 相同 KV 结构                                    │
│    Seq_len=8192: 0.3MB × 8192 = 2.5 GB per request              │
│    Batch=64: 2.5GB × 64 = 160 GB !!!                            │
│                                                                   │
│  Qwen-2.5-72B:                                                   │
│    L=80, n_kv=8, d_h=128, dtype=2                                │
│    Per token: 0.3 MB (同上)                                      │
│    Seq_len=131072 (128K): 0.3MB × 131072 = 40 GB per request!  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### GQA/MQA 的显存节省

```
┌─────────────────────────────────────────────────┐
│  Attention 头分组对 KV Cache 的影响               │
├─────────────────────────────────────────────────┤
│                                                  │
│  MHA (Multi-Head Attention):                     │
│    n_kv = n_heads = 64 (如 LLaMA-2-7B)          │
│    KV Cache 最大                                 │
│                                                  │
│  GQA (Grouped-Query Attention):                  │
│    n_kv = n_heads / group_size                   │
│    LLaMA-2-70B: 64 heads / 8 groups = 8 kv_heads│
│    KV Cache 减少 8x !                            │
│                                                  │
│  MQA (Multi-Query Attention):                    │
│    n_kv = 1                                      │
│    KV Cache 最小 (减少 n_heads 倍)               │
│    但精度有损                                    │
│                                                  │
│  对比 (LLaMA-70B, seq=4096, FP16):              │
│    MHA: 2×80×64×128×2×4096 = 10.0 GB/request    │
│    GQA: 2×80×8×128×2×4096  = 1.25 GB/request    │
│    MQA: 2×80×1×128×2×4096  = 0.16 GB/request    │
│                                                  │
└─────────────────────────────────────────────────┘
```

## 显存布局分析

### 推理时的显存分配

```
┌─────────────────────────────── GPU Memory (96 GB H20) ──────┐
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Model Weights                           │    │
│  │         (固定大小, 加载后不变)                        │    │
│  │         LLaMA-70B FP16 = ~140 GB / 8 GPUs = 17.5 GB │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              KV Cache                                │    │
│  │         (动态增长, 运行时最大的显存消耗者)             │    │
│  │         最大可用 = 96 - 17.5 - overhead ≈ 75 GB      │    │
│  │                                                      │    │
│  │         可服务的最大 batch:                           │    │
│  │         75GB / (1.25GB/req @ seq=4096) = 60 请求     │    │
│  │                                                      │    │
│  │         ⚠️ 如果 seq=32K:                             │    │
│  │         75GB / (10GB/req) = 7 请求  ← 大幅减少!      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌──────────────────┐                                       │
│  │  Activation Mem  │  (临时, 前向传播中间结果)              │
│  │  (较小)          │                                       │
│  └──────────────────┘                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### KV Cache 是核心瓶颈的原因

```
为什么 KV Cache 是推理系统的核心瓶颈？

1. 显存占比最大:
   - Model Weights: 固定 (17.5 GB/GPU for 70B TP=8)
   - KV Cache: 动态, 可占 70%+ 的可用显存
   - Activations: 相对较小

2. 直接限制吞吐:
   - 可用 KV Cache 显存 → 决定最大 batch size
   - 最大 batch size → 决定最大吞吐
   - 吞吐 = batch_size × tokens/s_per_request

3. 动态变化:
   - 每个请求的 KV Cache 随生成不断增长
   - 不同请求长度不同 → 显存碎片化
   - 请求完成后释放 → 需要管理空闲空间

4. 长上下文放大问题:
   - 128K 上下文: 单请求就占 40 GB KV Cache
   - 几乎无法做 batching → 吞吐暴降
```

## 朴素 KV Cache 管理的问题

### 预分配方案

```
朴素方案: 为每个请求预分配 max_seq_len 的 KV Cache 空间

请求 A: 实际用 500 tokens, 预分配 4096 tokens
请求 B: 实际用 200 tokens, 预分配 4096 tokens
请求 C: 实际用 3000 tokens, 预分配 4096 tokens

┌─────────────────────── KV Cache Memory ───────────────────────┐
│                                                                │
│  请求 A: [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]           │
│          ↑ 实际用 12%              浪费 88%! ↑                 │
│                                                                │
│  请求 B: [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]           │
│          ↑ 实际用 5%               浪费 95%! ↑                 │
│                                                                │
│  请求 C: [██████████████████████████░░░░░░░░░░░░░░]           │
│          ↑ 实际用 73%              浪费 27% ↑                  │
│                                                                │
│  总显存利用率: (500+200+3000) / (4096×3) = 30%                 │
│                                                                │
│  问题:                                                         │
│  1. 内部碎片 (Internal Fragmentation): 预分配但没用到的空间     │
│  2. 外部碎片 (External Fragmentation): 请求间的空闲块不连续    │
│  3. 显存利用率低 → batch size 小 → 吞吐低                     │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

### 实际浪费量化

```
统计数据 (典型推理场景):

实际序列长度分布 (来自 ShareGPT 数据集):
  - 平均 prompt: ~200 tokens
  - 平均 output: ~300 tokens
  - 实际总长: ~500 tokens
  - max_seq_len: 4096 tokens (配置值)

浪费率 = 1 - 500/4096 = 87.8%

如果有 60 个 slot (预分配):
  理论可用显存: 60 × 4096 × 0.3MB = 75 GB
  实际使用: 60 × 500 × 0.3MB = 9 GB
  浪费: 66 GB !!!

如果用 PagedAttention:
  只分配实际使用的空间: 60 × 500 × 0.3MB = 9 GB
  剩余 66 GB 可以服务更多请求！
  → 潜在吞吐提升: 75GB/9GB = 8.3x batch size
```

## KV Cache 的生命周期

```
一个请求的 KV Cache 生命周期:

┌────────────────────────────────────────────────────────┐
│                                                         │
│  1. 请求到达 → 分配 KV Cache 空间                       │
│     ┌───┐                                              │
│     │   │ (空的, 等待 Prefill)                          │
│     └───┘                                              │
│                                                         │
│  2. Prefill → 一次性填充 prompt 的 KV                   │
│     ┌████████████████─────────────────────┐            │
│     │prompt KV Cache │  (预留生成空间)     │            │
│     └████████████████─────────────────────┘            │
│                                                         │
│  3. Decode → 每步增加一个 token 的 KV                   │
│     ┌████████████████████─────────────────┐            │
│     │                    │                 │            │
│     └████████████████████─────────────────┘            │
│     ┌█████████████████████████────────────┐            │
│     │                         │            │            │
│     └█████████████████████████────────────┘            │
│                                                         │
│  4. 生成完成 (EOS 或 max_len) → 释放 KV Cache           │
│     ┌─────────────────────────────────────┐            │
│     │          (已释放, 可复用)             │            │
│     └─────────────────────────────────────┘            │
│                                                         │
│  5. 被抢占 (Preemption) → KV Cache 可能需要 swap       │
│     GPU → CPU (swap out) 或 完全丢弃 (recompute)       │
│                                                         │
└────────────────────────────────────────────────────────┘
```

## KV Cache 优化技术全景

```
┌─────────────────────────────────────────────────────────────┐
│              KV Cache 优化技术                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 架构级优化 (训练时决定):                                  │
│     ├─ GQA: 减少 KV head 数 (8x 节省)                       │
│     ├─ MQA: 极限减少 (全部共享)                              │
│     └─ Sliding Window Attention (Mistral): 限制 KV 长度     │
│                                                              │
│  2. 显存管理优化 (系统级):                                    │
│     ├─ PagedAttention: 按需分页分配 → 下一章详解             │
│     ├─ Prefix Caching: 共享公共前缀的 KV                     │
│     └─ KV Cache Offloading: GPU↔CPU 交换                    │
│                                                              │
│  3. 压缩优化:                                                │
│     ├─ KV Cache Quantization: FP16→INT8/FP8                 │
│     ├─ Token Pruning/Eviction: 丢弃不重要的历史 token        │
│     ├─ StreamingLLM: 只保留 sink tokens + recent window     │
│     └─ KV Cache Compression: 动态合并/压缩                   │
│                                                              │
│  4. 共享优化:                                                │
│     ├─ Beam Search 的 Copy-on-Write                         │
│     ├─ Parallel Sampling 的 KV 共享                          │
│     └─ 前缀共享 (System Prompt 复用)                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## KV Cache Quantization

```
KV Cache 量化 — 进一步减少显存占用:

原始 (FP16): 2 bytes/element
INT8:        1 byte/element  → 减少 50%
FP8 (E4M3):  1 byte/element → 减少 50%
INT4:        0.5 byte/element → 减少 75%

精度影响:
  - KV Cache INT8: 几乎无损 (perplexity 增加 < 0.1%)
  - KV Cache FP8: 同上, 硬件友好
  - KV Cache INT4: 有一定精度损失, 需要 fine-tuning 适配

vLLM 已支持: FP8 KV Cache (--kv-cache-dtype fp8)
效果: batch size 几乎翻倍!

示例 (LLaMA-70B, seq=4096, 8×H20):
  FP16 KV Cache: max batch = 60
  FP8 KV Cache:  max batch = 120  (2x!)
  → 吞吐接近翻倍
```

## 核心知识点

### 必问: "KV Cache 的显存占用怎么算？"

```
回答模板:

"KV Cache 的大小 = 2 × 层数 × KV头数 × head_dim × 数据类型字节 × 序列长度 × batch

以 LLaMA-70B 为例:
- 80 层, GQA 8 个 KV head, head_dim=128, FP16
- 每 token: 2×80×8×128×2 = 327KB
- 序列长度 4096: 1.25 GB per request
- Batch=64: 80 GB KV Cache

这就是为什么 KV Cache 是推理的核心瓶颈：
1. 它是最大的动态显存消耗者
2. 它直接限制了 max batch size → 限制了吞吐
3. 它的管理效率决定了显存利用率

解决方案:
- PagedAttention: 按需分配, 避免预分配浪费
- GQA/MQA: 从模型架构上减少 KV head 数
- KV Cache Quantization: FP16→FP8, 容量翻倍
- Prefix Caching: 共享公共前缀"
```

### 追问: "如何估算一台 H20 能服务多少并发？"

```
计算步骤:

总显存: 96 GB (H20)
模型权重 (TP=8, 70B FP16): 140/8 = 17.5 GB
可用 KV Cache: 96 - 17.5 - 5 (overhead) = 73.5 GB

KV Cache per request:
  假设平均 seq_len = 2000 (prompt + output)
  Per request: 2×80×8×128×2×2000 / 8(TP) = 0.16 GB

最大并发: 73.5 / 0.16 ≈ 460 requests (per GPU)

但这是理论最大值, 实际:
- 需要预留 headroom (80%): 460 × 0.8 = 368
- 序列长度有波动: 再打个折 → ~250-300 并发

8×H20 整体: ~2000-2400 并发 (假设 TP=8 是一个实例)
```

## 小结

| 要点 | 关键数字 |
|------|----------|
| KV Cache 计算 | 2 × L × n_kv × d_h × dtype × seq_len × batch |
| LLaMA-70B per token | 327 KB (FP16, GQA=8) |
| 典型浪费率 | 60-90% (朴素预分配) |
| GQA 节省 | 8x (64 heads → 8 KV heads) |
| FP8 量化节省 | 2x |
| 长上下文的放大 | 128K 序列 → 单请求 40GB! |
| 核心矛盾 | KV Cache 显存 vs Batch Size vs 吞吐 |
