# 01 - LLM 推理全流程深度解析

## 概述

LLM 推理不是简单的"输入文本→输出文本"，而是一个精密的多阶段流水线。
理解每个阶段的计算特征和资源消耗模式，是做推理优化的前提。

## 推理全流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LLM Inference Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input Text                                                          │
│      │                                                               │
│      ▼                                                               │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────┐      │
│  │Tokenize  │──▶│ Prefill  │──▶│  Decode  │──▶│Detokenize  │      │
│  │          │   │(Prompt)  │   │(Generate)│   │            │      │
│  └──────────┘   └──────────┘   └──────────┘   └────────────┘      │
│       │               │              │               │              │
│       ▼               ▼              ▼               ▼              │
│  Token IDs      KV Cache +     New Tokens      Output Text          │
│               First Token      (one by one)                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Stage 1: Tokenize（分词）

### 做了什么
将输入文本转换为 token ID 序列。

### 计算特征
- **CPU 密集** — 不需要 GPU
- **延迟极低** — 通常 < 1ms
- **不是瓶颈** — 但 tokenizer 的选择影响后续所有阶段

### 关键细节
```python
# BPE tokenizer 示例
text = "Hello, how are you?"
tokens = tokenizer.encode(text)  # [15496, 11, 703, 389, 499, 30]

# 不同 tokenizer 的效率差异很大
# GPT-4: ~4 chars/token (English)
# LLaMA: ~3.5 chars/token
# 中文: 通常 1-2 chars/token
```

### 对推理性能的影响
- Token 数量直接决定 Prefill 的计算量
- 更高效的 tokenizer → 更少的 token → 更快的推理
- Prompt 模板的 token 效率影响 TTFT

## Stage 2: Prefill（预填充）

### 做了什么
一次性处理所有输入 token，生成初始 KV Cache 和第一个输出 token。

### 计算特征

```
┌─────────────────────────────────────────┐
│           Prefill 阶段特征               │
├─────────────────────────────────────────┤
│ 类型:     Compute-Bound (计算密集)       │
│ 操作:     大矩阵乘法 (GEMM)             │
│ 并行度:   高 (所有 input token 并行处理)  │
│ GPU 利用: 高 (接近峰值 FLOPS)            │
│ 瓶颈:     算力 (TFLOPS)                 │
│ 时间:     与 prompt 长度成正比            │
└─────────────────────────────────────────┘
```

### 为什么是计算密集？

Prefill 对 N 个 token 做 Self-Attention + FFN：
```
对于 prompt 长度 N:
  - Attention: Q×K^T 是 (N, d) × (d, N) = O(N² × d) FLOPs
  - FFN: 两个线性层 = O(N × d × 4d) FLOPs
  - 总计: O(N² × d + N × d²) per layer

示例 (LLaMA-70B, prompt=2048):
  - d = 8192, layers = 80, heads = 64
  - Attention FLOPs ≈ 2048² × 8192 × 80 ≈ 2.75T FLOPs
  - FFN FLOPs ≈ 2048 × 8192 × 4 × 8192 × 2 × 80 ≈ 70T FLOPs
  - 总计 ≈ 73T FLOPs
  - H20 峰值: 148 TFLOPS (FP16) → 理论最快 ~0.5s
```

### Arithmetic Intensity（算术强度）
```
AI = FLOPs / Bytes_accessed

Prefill 的 AI:
  GEMM (N×d) × (d×d): AI = 2Nd²/(N×d + d²) ≈ 2N (当 N >> 1)
  
  对于 N=2048: AI ≈ 4096 FLOPs/Byte
  H20 HBM 带宽: 4TB/s, 峰值计算: 148 TFLOPS
  Roofline 拐点: 148T/4T = 37 FLOPs/Byte
  
  AI=4096 >> 37 → 完全 Compute-Bound ✓
```

## Stage 3: Decode（解码/生成）

### 做了什么
自回归生成：每一步生成一个 token，作为下一步的输入。

### 计算特征

```
┌─────────────────────────────────────────┐
│           Decode 阶段特征                │
├─────────────────────────────────────────┤
│ 类型:     Memory-Bound (访存密集)        │
│ 操作:     矩阵-向量乘法 (GEMV)          │
│ 并行度:   低 (每次只处理 1 个 token)     │
│ GPU 利用: 低 (远低于峰值 FLOPS)          │
│ 瓶颈:     显存带宽 (TB/s)               │
│ 时间:     与生成长度成正比               │
└─────────────────────────────────────────┘
```

### 为什么是访存密集？

每一步 Decode 只处理 1 个新 token：
```
对于 batch_size=1, 每步 Decode:
  - 需要读取: 整个模型权重 + 整个 KV Cache
  - 计算量: 一次 GEMV = O(d²) per layer
  - 数据量: O(d²) bytes per layer (权重)

Arithmetic Intensity:
  AI = 2d² / (d² × 2) = 1 FLOPs/Byte  (FP16 下)
  
  AI=1 << 37 (Roofline 拐点) → 完全 Memory-Bound ✓
```

### Decode 的时间开销
```
每步 Decode 时间 ≈ Model_Size / HBM_Bandwidth

LLaMA-70B (FP16 = 140GB):
  H20 带宽: 4TB/s
  理论最快: 140GB / 4TB/s = 35ms per token
  实际约: 40-50ms per token (batch=1)
  
这就是为什么 Decode 要用 batching 来分摊带宽成本！
Batch=32 时: 还是读 140GB 权重，但处理 32 个 token
  → 有效 AI = 32 FLOPs/Byte → 接近 Roofline 拐点
```

## Stage 4: Detokenize（反分词）

### 做了什么
将生成的 token ID 转回文本。

### 计算特征
- CPU 操作，延迟极低
- 流式输出时需要处理 token 边界问题
- 某些 tokenizer 的 decode 需要上下文（如 BPE 的 byte fallback）

### 流式输出的细节
```python
# 流式 detokenize 的陷阱
# 有些 token 不能单独 decode（如 UTF-8 多字节字符被拆分）
buffer = []
for token_id in generated_tokens:
    buffer.append(token_id)
    text = tokenizer.decode(buffer)
    if text and not text.endswith("�"):  # 验证完整性
        yield text
        buffer = []
```

## 各阶段资源消耗对比

```
┌──────────────┬──────────────┬───────────────┬──────────────┬──────────┐
│    指标       │   Tokenize   │    Prefill    │    Decode    │Detokenize│
├──────────────┼──────────────┼───────────────┼──────────────┼──────────┤
│ 设备         │     CPU      │     GPU       │     GPU      │   CPU    │
│ 瓶颈资源     │   CPU cycles │    FLOPS      │  HBM BW      │ CPU cycles│
│ 计算类型     │   字符串处理  │    GEMM       │    GEMV      │ 字符串处理│
│ 并行度       │     低       │     高        │     低       │    低     │
│ 持续时间     │    <1ms      │  10ms-10s     │  生成时间    │   <1ms   │
│ 优化方向     │  选好tokenizer│  FlashAttention│  Batching   │   流式   │
│              │              │  Tensor Para  │  KV Cache    │          │
│              │              │               │  量化        │          │
└──────────────┴──────────────┴───────────────┴──────────────┴──────────┘
```

## 推理延迟分解

### TTFT (Time To First Token)
```
TTFT = Tokenize + Prefill + 1st_Decode_Step
     ≈ Prefill_Time  (Tokenize 可忽略)
     
影响因素:
  - Prompt 长度 (N² 或 N×logN with FlashAttention)
  - 模型大小
  - GPU 算力
```

### TPOT (Time Per Output Token)
```
TPOT = Single_Decode_Step_Time

影响因素:
  - 模型大小 / HBM 带宽
  - Batch size (更大 batch → 更好的计算/带宽比)
  - KV Cache 长度 (Attention 部分会随上下文增长)
```

### 端到端延迟
```
E2E_Latency = TTFT + TPOT × (output_length - 1)

示例 (LLaMA-70B on 8×H20, batch=1):
  TTFT (prompt=1000 tokens) ≈ 200ms
  TPOT ≈ 45ms
  生成 200 tokens: 200 + 45×199 ≈ 9.2s
```

## 与 Batching 的关系

```
                    Batch=1           Batch=32
                    ───────           ────────
Prefill 时间:      不变 (compute-bound, GPU 已饱和)
                   除非显存不够 → 需要分 chunk

Decode 时间:       35ms              36ms (几乎不变！)
                   (bandwidth-bound, 多 batch 分摊带宽)

吞吐:             22 tok/s           22×32 ≈ 700 tok/s
                   (单请求)           (32 请求并行)
```

**关键洞察**: Decode 阶段增加 batch size 几乎不增加延迟，但线性提升吞吐！
这就是为什么 Continuous Batching 如此重要。

## 推理优化的三大方向

```
┌─────────────────────────────────────────────────┐
│          推理优化全景                             │
├─────────────────────────────────────────────────┤
│                                                  │
│  1. 减少计算量                                   │
│     ├─ 量化 (INT8/INT4/FP8)                     │
│     ├─ 投机解码 (Speculative Decoding)           │
│     ├─ 稀疏注意力 (Sparse Attention)             │
│     └─ 模型蒸馏/剪枝                            │
│                                                  │
│  2. 提高并行度                                   │
│     ├─ Tensor Parallelism (模型并行)             │
│     ├─ Pipeline Parallelism (流水线并行)          │
│     ├─ Sequence Parallelism                      │
│     └─ FlashAttention (算子融合)                 │
│                                                  │
│  3. 提高资源利用率                               │
│     ├─ Continuous Batching (连续批处理)           │
│     ├─ PagedAttention (减少显存浪费)             │
│     ├─ KV Cache 优化 (GQA/MQA/压缩)            │
│     └─ Prefix Caching (前缀共享)                │
│                                                  │
└─────────────────────────────────────────────────┘
```

## 核心知识点

### 关键问题
1. "描述一下 LLM 推理的完整流程" → 四阶段 + 各自特征
2. "Prefill 和 Decode 的区别" → 见下一章详解
3. "TTFT 和 TPOT 分别受什么影响" → 上面的分析
4. "如何优化推理延迟" → 三大方向 + 具体技术

### 要点总结
```
"LLM 推理分为四个阶段：
1. Tokenize: CPU 上将文本转 token ID
2. Prefill: GPU 上一次性处理所有输入 token，是 compute-bound 的
3. Decode: GPU 上逐 token 生成，是 memory-bound 的  
4. Detokenize: CPU 上转回文本

其中 Prefill 决定了 TTFT，Decode 决定了 TPOT。
优化方向不同：Prefill 优化算力利用（FlashAttention、TP），
Decode 优化带宽利用（batching、量化、KV Cache 管理）。"
```

## 小结

| 要点 | 记住 |
|------|------|
| Prefill 是计算密集 | 大 GEMM，AI 高，瓶颈在 FLOPS |
| Decode 是访存密集 | GEMV，AI 低，瓶颈在 HBM 带宽 |
| Batching 主要帮 Decode | 分摊权重加载的带宽成本 |
| TTFT 由 Prefill 决定 | 优化方向：FlashAttention、TP |
| TPOT 由 Decode 决定 | 优化方向：Batching、量化、KV Cache |
