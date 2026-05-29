# 03 — 张量并行详解：Megatron-style Column/Row Parallel

## 1. 张量并行的核心思想

张量并行将单个矩阵运算切分到多张 GPU 上并行执行。对于 Transformer 的核心计算 `Y = XW`，如果 W 太大（如 hidden=12288, FFN=49152），可以将 W 切分到多张 GPU。

### 关键约束

$$Y = X \cdot W = X \cdot [W_1 | W_2 | ... | W_t]$$

矩阵乘法可以按**列**或**行**切分，但不同切分方式需要不同的通信模式。

## 2. 列并行 (Column Parallel Linear)

### 2.1 原理

将权重矩阵 W ∈ R^{h×k} 按列切分为 t 个分片（t = TP size）：

```
W = [W₁, W₂, ..., Wₜ]    where Wᵢ ∈ R^{h × k/t}

输入 X ∈ R^{b×h} (每张 GPU 各有一份完整 X)

GPU i: Yᵢ = X @ Wᵢ       →  Yᵢ ∈ R^{b × k/t}
```

**前向计算**:
```
输入: X (在所有 GPU 上完整存在)
计算: Yᵢ = X @ Wᵢ (本地计算，无需通信)
输出: Yᵢ 是 Y 的第 i 个列分片

注: 前向不需要通信！
```

**反向传播**:
```
给定 ∂L/∂Yᵢ (loss 对本地输出的梯度):

∂L/∂Wᵢ = Xᵀ @ (∂L/∂Yᵢ)    → 本地计算，不需要通信
∂L/∂X = (∂L/∂Yᵢ) @ Wᵢᵀ     → 这只是 ∂L/∂X 的部分和

需要 AllReduce 来得到完整的 ∂L/∂X:
∂L/∂X = Σᵢ (∂L/∂Yᵢ) @ Wᵢᵀ = AllReduce(∂L/∂Xᵢ)
```

### 2.2 GeLU 的处理

列并行的一个关键优势：**GeLU 可以本地计算**

```
Y = GeLU(X @ W) 

列并行: Yᵢ = GeLU(X @ Wᵢ)  ✓ (GeLU 是逐元素操作，可在列分片上独立做)

注意: 这对行并行不成立！
GeLU(X₁@W₁ + X₂@W₂) ≠ GeLU(X₁@W₁) + GeLU(X₂@W₂)  (非线性不可分)
```

### 2.3 实现要点

```python
class ColumnParallelLinear(nn.Module):
    """
    将 Linear(in_features, out_features) 按列切分
    每张 GPU 持有 Linear(in_features, out_features // tp_size)
    """
    def __init__(self, in_features, out_features, tp_size, tp_rank):
        super().__init__()
        assert out_features % tp_size == 0
        self.out_features_per_partition = out_features // tp_size
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        self.bias = nn.Parameter(
            torch.empty(self.out_features_per_partition)
        )
    
    def forward(self, x):
        # x: [batch, seq_len, in_features] — 完整输入
        # 本地计算，不需要通信
        output = F.linear(x, self.weight, self.bias)
        # output: [batch, seq_len, out_features // tp_size]
        return output
```

## 3. 行并行 (Row Parallel Linear)

### 3.1 原理

将权重矩阵 W ∈ R^{k×h} 按行切分为 t 个分片：

```
W = [W₁; W₂; ...; Wₜ]ᵀ 实际上按行:  Wᵢ ∈ R^{k/t × h}

对应地，输入 X 也需要按列切分: Xᵢ ∈ R^{b × k/t}

GPU i: Yᵢ = Xᵢ @ Wᵢ     →  Yᵢ ∈ R^{b × h}

最终结果: Y = Σᵢ Yᵢ = AllReduce([Y₁, Y₂, ..., Yₜ])
```

**前向计算**:
```
输入: Xᵢ (每张 GPU 持有输入的第 i 个分片 — 来自上一层列并行的输出)
计算: Yᵢ = Xᵢ @ Wᵢ (本地计算)
通信: Y = AllReduce(Yᵢ)  ← 前向需要一次 AllReduce！
输出: Y (在所有 GPU 上完整存在)
```

**反向传播**:
```
给定 ∂L/∂Y (完整):

∂L/∂Wᵢ = Xᵢᵀ @ (∂L/∂Y)   → 本地计算
∂L/∂Xᵢ = (∂L/∂Y) @ Wᵢᵀ    → 本地计算，不需要通信

反向不需要额外通信！（因为 ∂L/∂Xᵢ 就是送回列并行层的梯度）
```

### 3.2 实现要点

```python
class RowParallelLinear(nn.Module):
    """
    将 Linear(in_features, out_features) 按行切分
    每张 GPU 持有 Linear(in_features // tp_size, out_features)
    """
    def __init__(self, in_features, out_features, tp_size, tp_rank):
        super().__init__()
        assert in_features % tp_size == 0
        self.in_features_per_partition = in_features // tp_size
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )
        # bias 只在 rank 0 上，或者分散到 AllReduce 后加
        self.bias = nn.Parameter(torch.empty(out_features))
    
    def forward(self, x):
        # x: [batch, seq_len, in_features // tp_size] — 本地分片
        output_local = F.linear(x, self.weight)
        # AllReduce 求和得到完整结果
        dist.all_reduce(output_local, op=dist.ReduceOp.SUM)
        output = output_local + self.bias
        return output
```

## 4. Transformer 中的 TP 布局

### 4.1 MLP 块

```
标准 MLP:
  h₁ = GeLU(x @ W₁ + b₁)    # W₁ ∈ R^{H × 4H}  (up projection)
  h₂ = h₁ @ W₂ + b₂          # W₂ ∈ R^{4H × H}  (down projection)

TP 布局:
  W₁ 用列并行: 每 GPU 持有 R^{H × 4H/t}，GeLU 本地做
  W₂ 用行并行: 每 GPU 持有 R^{4H/t × H}，输出 AllReduce

通信: 前向 1 次 AllReduce (在 W₂ 之后)
       反向 1 次 AllReduce (W₁ 的输入梯度)
```

```
┌─────────────────────────────────────────────────┐
│ MLP Block (TP=2)                                │
│                                                 │
│  Input x (replicated on both GPUs)              │
│     │                                           │
│     ├─── GPU 0: x @ W₁₀ → GeLU → h₀           │
│     └─── GPU 1: x @ W₁₁ → GeLU → h₁           │
│                                                 │
│     ├─── GPU 0: h₀ @ W₂₀ → y₀                 │
│     └─── GPU 1: h₁ @ W₂₁ → y₁                 │
│                                                 │
│  AllReduce: y = y₀ + y₁  (on both GPUs)        │
│                                                 │
│  Output y (replicated)                          │
└─────────────────────────────────────────────────┘
```

### 4.2 Self-Attention 块

```
标准 Multi-Head Attention:
  Q = x @ Wq,  K = x @ Wk,  V = x @ Wv    # 投影
  attn = softmax(Q @ Kᵀ / √d) @ V           # Attention
  out = attn @ Wo                            # 输出投影

TP 布局 (t 个 GPU, h 个 attention heads):
  Wq, Wk, Wv 用列并行: 每 GPU 负责 h/t 个 heads
    GPU i: Qᵢ = x @ Wqᵢ,  Kᵢ = x @ Wkᵢ,  Vᵢ = x @ Wvᵢ
  
  Attention 本地计算:
    GPU i: attnᵢ = softmax(Qᵢ @ Kᵢᵀ / √d) @ Vᵢ
    # 每个 head 的 attention 是独立的！→ 可以本地做
  
  Wo 用行并行:
    GPU i: outᵢ = attnᵢ @ Woᵢ
    AllReduce: out = Σᵢ outᵢ

通信: 前向 1 次 AllReduce (Wo 之后)
       反向 1 次 AllReduce (Wq/Wk/Wv 的输入梯度)
```

### 4.3 完整 Transformer 层的通信

```
一个 Transformer 层:
  [LayerNorm] → [Attention (Column→Row)] → [Residual] → [LayerNorm] → [MLP (Column→Row)] → [Residual]
                          ↑                                                    ↑
                     AllReduce (fwd)                                     AllReduce (fwd)
                     AllReduce (bwd)                                     AllReduce (bwd)

前向: 2 次 AllReduce
反向: 2 次 AllReduce
总计: 每层 4 次 AllReduce
```

## 5. 通信量详细分析

### 5.1 单次 AllReduce 数据量

```
AllReduce 数据量 = tensor size = B × S × H × dtype_bytes

B = micro_batch_size (per GPU)
S = sequence_length
H = hidden_size
dtype = BF16 → 2 bytes

例: B=4, S=2048, H=4096, BF16
  单次 = 4 × 2048 × 4096 × 2 = 64 MB
```

### 5.2 每步总通信量

```
L = 层数, 每层 4 次 AllReduce:

总通信量 = 4L × B × S × H × 2  (BF16)
Ring AllReduce 实际传输 = 4L × 2(t-1)/t × B×S×H×2 ≈ 4L × 2 × B×S×H×2

例: L=32, B=4, S=2048, H=4096, t=4, BF16
  总通信量 = 4 × 32 × 2 × (3/4) × 4 × 2048 × 4096 × 2
           = 4 × 32 × 1.5 × 64 MB
           = 12,288 MB ≈ 12 GB per step
```

### 5.3 通信时间 vs 计算时间

```
NVLink 带宽 (H20): ~450 GB/s bidirectional (~225 GB/s unidirectional effective)

通信时间 = 12 GB / 225 GB/s ≈ 53 ms

计算时间 (H20, BF16):
  MLP 计算量 = 2 × B × S × H × 4H × L × 2 (fwd + bwd)
             = 2 × 4 × 2048 × 4096 × 16384 × 32 × 2
             ≈ 2.75 × 10^13 FLOPs per GPU (每 GPU 只算 1/t)
             = 2.75e13 / 4 = 6.87e12 FLOPs
  
  H20 BF16 算力: 148 TFLOPS
  计算时间 ≈ 6.87e12 / 148e12 ≈ 46 ms

通信/计算比 ≈ 53/46 ≈ 1.15
→ 通信是瓶颈！这就是为什么 TP 必须用 NVLink
```

### 5.4 为什么 TP 受限于 NVLink

```
如果走 PCIe (64 GB/s):
  通信时间 = 12 GB / 64 GB/s = 187 ms
  通信/计算比 = 187/46 = 4.1x → 严重瓶颈

如果走网络 (25 GB/s InfiniBand):
  通信时间 = 12 GB / 25 GB/s = 480 ms
  通信/计算比 = 480/46 = 10.4x → 完全不可接受

结论:
  TP 的通信频率极高（每层都要通信），只能放在 NVLink 互联的 GPU 组内。
  H20 的 8 卡全 NVLink 互联，所以 TP ≤ 8。
  典型配置: TP=4 或 TP=8。
```

## 6. TP 对显存的影响

### 6.1 参数显存

```
TP=t 时每卡参数量:

Attention:
  Wq, Wk, Wv: H × H/t × 3 = 3H²/t
  Wo: H/t × H = H²/t
  小计: 4H²/t

MLP:
  W₁: H × 4H/t = 4H²/t
  W₂: 4H/t × H = 4H²/t
  小计: 8H²/t

每层每卡: 12H²/t (相比不切分的 12H²)
→ 参数显存线性减少 t 倍
```

### 6.2 激活值显存

```
不使用 SP 时:
  LayerNorm 和 Dropout 的输入仍是完整的 B×S×H
  只有 Attention 和 MLP 内部的激活值被切分

使用 SP (Sequence Parallelism) 时:
  LayerNorm 和 Dropout 的输入也按 sequence 维度切分
  激活值进一步减少
```

## 7. f/g 操作符（Megatron 的优雅抽象）

Megatron-LM 定义了 `f` 和 `g` 两个通信操作符：

```
f: 前向 identity，反向 AllReduce
g: 前向 AllReduce，反向 identity

使用方式:
  Column Parallel:
    输入经过 f (前向: 直接传递; 反向: AllReduce 梯度)
    输出直接使用（已经是分片的）
  
  Row Parallel:
    输入直接使用（已经是分片的）
    输出经过 g (前向: AllReduce 结果; 反向: 直接传递梯度)
```

```python
class _CopyToModelParallelRegion(torch.autograd.Function):
    """f: identity in forward, all-reduce in backward"""
    @staticmethod
    def forward(ctx, input_):
        return input_
    
    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)  # AllReduce


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """g: all-reduce in forward, identity in backward"""
    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)  # AllReduce
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
```

## 8. TP 的扩展：GQA/MQA 下的处理

### Grouped Query Attention (GQA)

```
GQA: num_kv_heads < num_heads

例: num_heads=32, num_kv_heads=8, TP=4
  每 GPU: 32/4=8 个 Q heads, 8/4=2 个 KV heads
  
  要求: num_kv_heads 必须能被 TP_size 整除
  如果 num_kv_heads=8, TP_size=4: ✓ (每卡 2 个 KV head)
  如果 num_kv_heads=8, TP_size=8: ✓ (每卡 1 个 KV head)
  如果 num_kv_heads=4, TP_size=8: ✗ (需要复制 KV heads)
```

## 9. 实践建议

### 9.1 TP size 选择

```
规则:
1. TP size ≤ 单机 NVLink GPU 数量
2. TP size 必须能整除 num_attention_heads 和 num_kv_heads
3. TP size 越大，通信开销越大（但每次通信量 ∝ (t-1)/t）
4. TP size 太大会导致每卡 tensor 太小，矩阵乘法效率下降

8×H20 建议:
  - 7B 模型: TP=4 (32 heads / 4 = 8 heads per GPU)
  - 13B 模型: TP=4 或 TP=8
  - 70B 模型: TP=8 (需多机 PP)
```

### 9.2 常见 Bug

```
1. 参数初始化不一致:
   TP 切分后每个 rank 的参数必须是原始参数的对应分片
   → 先初始化完整参数，再切分分配

2. Dropout 种子不同步:
   如果 Dropout 用在 AllReduce 之前，不同 rank 的 dropout mask 不同
   → 导致 AllReduce 后结果不等于全量计算
   → 解决: 在列并行区域使用相同的 dropout seed

3. 随机数状态管理:
   数据采样随机数: 每个 TP rank 不同
   模型 dropout 随机数: 同一 TP group 内相同
   → Megatron 使用 CudaRNGStatesTracker 管理
```

## 10. 下一步

- [04_pipeline_parallelism.md](04_pipeline_parallelism.md)：流水线并行调度算法
- [Lab 03](../labs/03_tensor_parallelism/)：手写列并行和行并行
