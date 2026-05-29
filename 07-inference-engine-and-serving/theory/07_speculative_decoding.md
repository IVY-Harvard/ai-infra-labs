# 07 - 投机解码 (Speculative Decoding)

## 核心问题

> 投机解码为什么能加速推理？为什么不影响输出质量？

## 动机：Decode 的根本瓶颈

```
Decode 阶段的问题:
  - 每步只生成 1 个 token
  - 但每步都要读取整个模型权重 (140GB for 70B)
  - GPU 利用率 < 5% (Memory-Bound)
  - 即使优化到极致, 也被 HBM 带宽限制

理想: 每步生成多个 token → 分摊带宽成本
问题: 自回归性质 → 下一个 token 依赖上一个, 无法并行

投机解码的思路:
  用小模型"猜测"多个 token → 大模型"验证" → 一步验证多个
  → 有效地一步生成多个 token!
```

## 投机解码原理

### 基本流程

```
┌─────────────────────────────────────────────────────────────┐
│              Speculative Decoding 流程                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: Draft (投机/草稿)                                   │
│  ─────────────────────────                                   │
│  小模型 (Draft Model) 自回归生成 K 个候选 token:             │
│                                                              │
│  Context: "The cat sat on the"                               │
│  Draft:   → "mat" → "and" → "looked" → "at" → "me"        │
│           (K=5 个候选 token, 用小模型快速生成)                │
│                                                              │
│  Step 2: Verify (验证)                                       │
│  ─────────────────────                                       │
│  大模型一次性验证这 K 个 token (Prefill 模式, 并行!):       │
│                                                              │
│  Input: "The cat sat on the mat and looked at me"            │
│  大模型前向传播 → 得到每个位置的概率分布                      │
│                                                              │
│  验证结果:                                                   │
│  Position 1: "mat"    → P_large("mat"|context) > threshold ✓ │
│  Position 2: "and"    → P_large("and"|...) > threshold ✓     │
│  Position 3: "looked" → P_large("looked"|...) > threshold ✓  │
│  Position 4: "at"     → P_large("at"|...) < threshold ✗      │
│  Position 5: "me"     → 被拒绝 (position 4 已失败)          │
│                                                              │
│  Step 3: Accept + Resample                                   │
│  ─────────────────────────                                   │
│  接受前 3 个 token + 用大模型重新采样第 4 个 token            │
│  → 一步生成了 4 个 token! (正常只能生成 1 个)                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 为什么不影响输出质量？

```
关键保证: 投机解码的输出分布 = 纯大模型的输出分布

验证算法 (Rejection Sampling):

对于每个候选 token x_i (draft model 生成):
  p = P_large(x_i | context)   # 大模型给这个 token 的概率
  q = P_draft(x_i | context)   # 小模型给这个 token 的概率
  
  if p >= q:
    accept x_i (概率 1)
  else:
    accept x_i with probability p/q
    reject with probability 1 - p/q
    → 如果 reject, 从修正分布重新采样:
      P_resample = normalize(max(0, P_large - P_draft))

数学证明:
  最终采样的 token 分布:
  P(accept x) = q(x) × min(1, p(x)/q(x))           (接受小模型的)
  P(resample x) = (1 - Σ q(x)min(1,p(x)/q(x))) × P_resample(x)  (重采样的)
  
  两者之和 = P_large(x)  ← 等价于直接从大模型采样!

结论: 无论 draft model 多差, 输出质量都等价于大模型
     draft model 差 → acceptance rate 低 → 加速比低 (但不影响质量!)
```

## 加速比分析

### 理论加速比

```
设:
  K = 每轮投机的 token 数
  α = 平均接受率 (acceptance rate)
  t_draft = draft model 生成 K 个 token 的时间
  t_verify = 大模型验证 K 个 token 的时间
  t_decode = 大模型正常 decode 一步的时间

投机解码一轮:
  时间: t_draft + t_verify
  产出: α×K + 1 个 token (平均)
  
  (α×K 个被接受 + 1 个重采样的)

正常解码产出相同 token:
  时间: (α×K + 1) × t_decode

加速比:
  Speedup = (α×K + 1) × t_decode / (t_draft + t_verify)

简化 (假设 t_verify ≈ t_decode, t_draft << t_verify):
  Speedup ≈ α×K + 1

示例:
  K=5, α=0.7: Speedup ≈ 0.7×5 + 1 = 4.5x (理论上限)
  K=5, α=0.5: Speedup ≈ 0.5×5 + 1 = 3.5x
  K=3, α=0.8: Speedup ≈ 0.8×3 + 1 = 3.4x

实际 (考虑 overhead):
  通常 1.5x - 2.5x 加速
```

### 影响接受率的因素

```
接受率 α 取决于 draft model 和 target model 的分布匹配度:

高接受率场景:
  - Draft model 是 target 的蒸馏版 → 分布接近
  - 生成确定性内容 (代码, 固定格式)
  - temperature = 0 (greedy) → 两个模型大概率选同一个 token

低接受率场景:
  - Draft model 太小, 能力差距大
  - 创意写作 (high temperature) → 分布分散
  - 需要世界知识的内容 (小模型不具备)

典型接受率:
  LLaMA-7B draft + LLaMA-70B target: α ≈ 0.6-0.8
  LLaMA-1B draft + LLaMA-70B target: α ≈ 0.4-0.6
  Medusa heads: α ≈ 0.5-0.7 (per head)
```

## 投机解码变体

### 1. Draft Model (经典方案)

```
使用独立的小模型作为 Draft Model:

Target: LLaMA-70B
Draft:  LLaMA-7B (或同系列小模型)

优点:
  - 实现简单
  - Draft model 可以独立优化

缺点:
  - 需要额外显存存 draft model
  - Draft model 需要和 target 是同系列 (tokenizer 相同)
  - 两次模型加载
  
适用: 有足够显存放两个模型 (你的 8×H20 够!)
```

### 2. Medusa (多头投机)

```
在 target model 的最后一层加多个 prediction heads:

┌────────────────────────────────────────┐
│  Target Model (LLaMA-70B)               │
│  ┌─────────────┐                        │
│  │ Last Hidden  │                        │
│  │   State      │                        │
│  └──┬──┬──┬──┬─┘                        │
│     │  │  │  │                           │
│     ▼  ▼  ▼  ▼                          │
│  ┌──┐┌──┐┌──┐┌──┐                      │
│  │H0││H1││H2││H3│  ← Medusa Heads       │
│  └──┘└──┘└──┘└──┘                      │
│   │   │   │   │                          │
│   ▼   ▼   ▼   ▼                         │
│  t+1 t+2 t+3 t+4  ← 预测后续 token     │
│                                          │
│  Head 0: 预测下一个 token (和原始 LM head 相同)│
│  Head 1: 预测第 2 个 token                │
│  Head 2: 预测第 3 个 token                │
│  Head 3: 预测第 4 个 token                │
│                                          │
└────────────────────────────────────────┘

优点:
  - 不需要额外的 draft model
  - 额外参数量很小 (几个 MLP head)
  - 和 target model 共享计算

缺点:
  - 需要额外训练 Medusa heads
  - 接受率可能低于独立 draft model
  
Tree-based verification:
  Medusa 生成 token 树 (每个 head top-k) → 一次验证所有路径
```

### 3. Eagle (特征级投机)

```
EAGLE: 用 target model 的特征预测未来 token

┌────────────────────────────────────────┐
│  Target Model 前向传播:                  │
│  Input → ... → last_hidden (feature)    │
│                     │                    │
│  EAGLE Draft Head:  │                    │
│  feature → lightweight_model → K tokens │
│  (只用 1-2 层 transformer)              │
│                                          │
│  然后用 target model 验证               │
│                                          │
└────────────────────────────────────────┘

优势:
  - 利用 target model 的高质量特征
  - Draft head 很轻量 (1-2 层)
  - 接受率通常比 Medusa 高
  
EAGLE-2:
  - 动态调整投机长度 (confidence-based)
  - 接受率高时多猜, 低时少猜
```

### 4. Self-Speculative (自投机)

```
不用额外模型, 用 target model 自身的"跳层":

正常: Layer 0 → Layer 1 → ... → Layer 79 (80 layers)
Draft: Layer 0 → Layer 20 → Layer 40 → Layer 79 (跳层, 快 4x)

或者用 early exit:
  前 20 层的输出 → 直接采样 → 作为 draft

优点: 不需要额外显存
缺点: 接受率较低, 加速有限
```

## 在 vLLM 中使用投机解码

```bash
# Draft Model 方式
vllm serve meta-llama/Llama-2-70b-hf \
  --tensor-parallel-size 8 \
  --speculative-model meta-llama/Llama-2-7b-hf \
  --num-speculative-tokens 5 \
  --speculative-max-model-len 4096

# Medusa 方式 (需要 medusa heads 权重)
vllm serve meta-llama/Llama-2-70b-hf \
  --tensor-parallel-size 8 \
  --speculative-model "[medusa]" \
  --num-speculative-tokens 3

# 关键参数:
# --num-speculative-tokens K: 每轮投机几个 token
# --speculative-draft-tensor-parallel-size: draft model 的 TP
# --spec-decoding-acceptance-method: rejection_sampler / typical_acceptance
```

## 投机解码的适用场景

```
┌─────────────────────────────────────────────────────────────┐
│  适合投机解码的场景                                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ✓ Batch size 小 (1-4):                                     │
│    - 本身 Memory-Bound → 投机利用空闲算力                    │
│    - 大 batch 时 GPU 已经 Compute-Bound → 投机反而拖慢       │
│                                                              │
│  ✓ 确定性内容生成:                                           │
│    - 代码补全 (高接受率)                                     │
│    - 格式化输出 (JSON, 固定模板)                              │
│    - 翻译 (source-target 对齐度高)                            │
│                                                              │
│  ✓ 延迟敏感场景:                                             │
│    - 实时对话 (低并发, 要快)                                  │
│    - 单请求推理                                              │
│                                                              │
│  ✗ 不适合的场景:                                             │
│    - 大 batch 推理 (GPU 已满载)                               │
│    - 高并发服务 (吞吐已足够, 优先级是 batch)                  │
│    - 创意写作 (高 temperature → 低接受率)                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 知识要点框架

### "请解释投机解码的原理"

```
"投机解码的核心思想:

1. 用小模型快速生成 K 个候选 token (Draft)
2. 用大模型一次性验证这 K 个 token (Verify, 利用 Prefill 并行性)
3. 通过 rejection sampling 决定接受哪些

为什么能加速:
- 验证 K 个 token (Prefill 模式) ≈ 生成 1 个 token (Decode 模式) 的时间
  (因为 Prefill 是 Compute-Bound, 增加 token 数不线性增加时间)
- 所以一次验证就能"免费"确认多个 token

为什么不影响质量:
- Rejection Sampling 保证最终分布 = 大模型分布
- 数学可证明: 无论 draft model 多差, 输出等价于纯大模型
- draft 差 → acceptance rate 低 → 加速少, 但质量不受影响

加速比:
- 理论: α×K + 1 (α=接受率, K=投机长度)
- 实际: 1.5-2.5x (考虑 overhead)
- 适合低 batch, 延迟敏感场景"
```

## 小结

| 方案 | 原理 | 优势 | 劣势 | 加速比 |
|------|------|------|------|--------|
| Draft Model | 独立小模型猜测 | 简单可靠 | 额外显存 | 1.5-2.5x |
| Medusa | 多头并行预测 | 无额外模型 | 需训练 heads | 1.3-2.0x |
| Eagle | 特征级预测 | 高接受率 | 需训练 | 1.5-2.5x |
| Self-Spec | 跳层推理 | 零额外开销 | 接受率低 | 1.2-1.5x |
