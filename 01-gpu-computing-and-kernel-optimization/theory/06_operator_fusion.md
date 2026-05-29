# 06 - 算子融合原理

## 为什么算子融合是最重要的优化？

在 AI 推理和训练中，算子融合（Operator/Kernel Fusion）通常能带来 **2-5x** 的性能提升，是投入产出比最高的优化手段。

### 未融合的执行流程

```python
# PyTorch eager mode 下执行 LayerNorm + ReLU + Linear
x = layer_norm(x)    # Kernel 1: 读 HBM → 计算 → 写 HBM
x = relu(x)          # Kernel 2: 读 HBM → 计算 → 写 HBM  
x = linear(x)        # Kernel 3: 读 HBM → 计算 → 写 HBM
```

每个算子独立启动 kernel，中间结果必须写回 HBM 再读出：

```
时间线：
[Kernel 1: LN] → [写HBM] → [读HBM] → [Kernel 2: ReLU] → [写HBM] → [读HBM] → [Kernel 3: Linear]
                  ~~~~~~~~   ~~~~~~~~                      ~~~~~~~~   ~~~~~~~~
                  这些 HBM 读写完全可以避免！
```

### 融合后的执行流程

```
[Fused Kernel: LN + ReLU] → [写HBM] → [读HBM] → [Kernel: Linear]
                                        只有一次中间读写
```

甚至可以进一步融合（如果 Linear 的计算模式允许）。

## 融合加速的三个原因

### 1. 减少 HBM 访问（最主要）

```
以 x = relu(layer_norm(x)) 为例，x shape = [batch=64, hidden=4096], FP16

未融合：
- LN: 读 x (512KB) + 写 out (512KB) = 1MB HBM 访问
- ReLU: 读 x (512KB) + 写 out (512KB) = 1MB HBM 访问
- 总共: 2MB HBM 访问

融合后：
- Fused: 读 x (512KB) + 写 out (512KB) = 1MB HBM 访问
- 节省: 50% HBM 访问

对于 N 个可融合的 pointwise 算子：
- 未融合: N × 2 × tensor_size 的 HBM 访问
- 融合后: 1 × 2 × tensor_size 的 HBM 访问
- 节省比例: (N-1)/N → N 越大节省越多
```

### 2. 减少 Kernel Launch Overhead

```
每次 kernel 启动的固定开销：
- CPU → GPU 命令传输: ~5-10 μs
- GPU 调度 overhead: ~2-5 μs
- 总计: ~7-15 μs / kernel

看起来很小？但 LLM 推理中：
- 一个 Transformer layer 有 ~20-30 个算子
- 一个 70B 模型有 80 层 → ~2000 个 kernel
- 总 overhead: 2000 × 10μs = 20ms
- 而单 token 推理总时间可能就 30-50ms
- overhead 占比高达 40-60%！

融合后：
- 2000 个 kernel → ~200 个 kernel
- overhead 降低 ~10x
```

### 3. 提高数据局部性

```
融合 kernel 中，中间数据可以保持在：
- 寄存器（最快）
- 共享内存（次快）
而不是写回 HBM 再读出

数据在寄存器中的带宽: ~20 TB/s
数据在 HBM 中的带宽: 4 TB/s
差距: 5x
```

## 融合的类型

### 1. Pointwise Fusion（逐元素融合）

最简单也最常见的融合类型：

```
条件：每个输出元素只依赖对应位置的输入元素
适用：ReLU, GELU, Add, Mul, Dropout, 类型转换...

示例：y = dropout(gelu(x * w + b))
未融合：5 个 kernel
融合后：1 个 kernel，中间结果全在寄存器
```

### 2. Reduction Fusion（归约融合）

```
条件：将 reduction 与其前后的 pointwise ops 融合
适用：LayerNorm, Softmax, BatchNorm

示例：LayerNorm = mean + variance + normalize + scale + shift
未融合：5 个 kernel（包含多次全局读写）
融合后：1 个 kernel，利用 shared memory 做 reduction
```

### 3. GEMM Epilogue Fusion（矩阵乘后处理融合）

```
条件：GEMM 后跟 pointwise 操作
适用：Linear + ReLU, Linear + Add + LayerNorm

示例：y = relu(x @ W + b)
- cuBLAS 做 GEMM
- 在 GEMM 写结果时顺便做 ReLU（在寄存器中）
- 不需要先写 HBM 再启动 ReLU kernel

cuBLAS 支持的 epilogue:
- RELU, GELU
- Bias Add
- Scale
```

### 4. Memory-bound Kernel Fusion（访存密集算子融合）

```
条件：多个访存密集算子处理同一数据
示例：Residual Add + LayerNorm + Dropout

关键insight：
这些算子各自都是 memory-bound
融合后，总的内存访问量减少
但计算量不变 → 变得"更计算密集"
→ 离 roofline 的屋顶更近 → GPU 利用率提高
```

## FlashAttention：融合的巅峰之作

### 标准 Attention 的问题

```python
# 标准实现（PyTorch）
def standard_attention(Q, K, V):
    # Q, K, V: [batch, heads, seq_len, head_dim]
    # seq_len=2048, head_dim=128
    
    S = Q @ K.T          # [batch, heads, 2048, 2048] — 存入 HBM!
                         # 大小: batch×heads×2048×2048×2bytes = 很大
    P = softmax(S)       # [batch, heads, 2048, 2048] — 读+写 HBM
    O = P @ V            # [batch, heads, 2048, 128]  — 读 HBM
    return O
```

问题：中间矩阵 S 和 P 的大小是 O(N²)，序列越长越爆：
- seq_len=2048: S 大小 = 2048² × 2 = 8 MB/head
- seq_len=8192: S 大小 = 8192² × 2 = 128 MB/head
- seq_len=32768: S 大小 = 32768² × 2 = 2 GB/head → 显存溢出！

### FlashAttention 的核心思想

**不把 N×N 的注意力矩阵完整存入 HBM**，而是分块计算。

```
传统方法：
Q[N×d] × K[N×d]ᵀ → S[N×N] → softmax → P[N×N] × V[N×d] → O[N×d]
                     ↑ 这个 N×N 矩阵太大

FlashAttention：
for each block of Q (大小 Br×d):
    for each block of K, V (大小 Bc×d):
        - 计算局部 S_block = Q_block × K_blockᵀ (Br×Bc) ← 在 SRAM 中!
        - 计算局部 softmax（在线算法）
        - 用局部 P × V_block 更新输出
```

### 在线 Softmax（Online Softmax）

FlashAttention 的技术难点：softmax 需要全局 max 和 sum，但我们分块处理。

```
标准 softmax:
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
需要先扫描完所有 x 才能算 max，再扫描一遍算 sum

在线算法（递推更新）：
处理第 1 块时: m₁ = max(block₁), l₁ = sum(exp(block₁ - m₁))
处理第 2 块时: 
  m₂ = max(m₁, max(block₂))           ← 更新全局 max
  l₂ = l₁ × exp(m₁ - m₂) + sum(exp(block₂ - m₂))  ← 更新全局 sum
  O₂ = O₁ × exp(m₁ - m₂)/l₂ + P₂ × V₂ / l₂       ← 修正之前的输出

这样每块数据只需读一次 HBM！
```

### FlashAttention 的 IO 复杂度

```
标准 Attention:
- HBM 读写: O(N² + Nd) ← 主要是 N×N 矩阵的读写

FlashAttention:
- HBM 读写: O(N²d / M) ← M 是 SRAM 大小
- 当 M 足够大时，远小于 O(N²)

具体数字（seq_len=2048, d=128, SRAM=228KB）:
- 标准: ~16 MB HBM 读写 / head
- Flash: ~4 MB HBM 读写 / head
- 加速: ~4x（纯 IO 角度）
- 实际加速: 2-4x（加上 kernel launch 等因素）
```

### FlashAttention-2 的进一步优化

```
FlashAttention-1 → FlashAttention-2 的改进：

1. 减少非矩阵乘计算的占比
   - FA1: 有很多 rescaling 操作消耗 CUDA Core
   - FA2: 重新组织算法，减少 rescaling 次数

2. 更好的并行度
   - FA1: 外层循环在 batch×heads 维度并行
   - FA2: 同时在 seq_len 维度并行 → 更多 SM 被利用

3. 优化 Warp 级别的工作分配
   - FA1: 所有 warp 做相同的工作
   - FA2: 不同 warp 处理输出矩阵的不同部分

结果: FA2 比 FA1 快 ~2x
```

## Kernel Fusion 策略

### 自动融合（编译器）

```python
# torch.compile 自动识别融合机会
@torch.compile
def transformer_block(x, weight, bias):
    # Inductor 会自动融合以下 pointwise ops:
    y = x @ weight + bias    # GEMM + bias → GEMM epilogue fusion
    y = F.gelu(y)            # ← 融合到 bias add
    y = F.dropout(y, 0.1)   # ← 可能融合
    return y
```

编译器能自动融合的：
- 连续的 pointwise 操作
- GEMM + epilogue（如果 cuBLAS 支持）
- 简单的 reduction + pointwise

编译器**不能**自动融合的：
- 复杂的算法重构（如 FlashAttention 的分块策略）
- 跨 kernel 的数据 reuse pattern
- 需要算法层面重新设计的优化

### 手动融合（Triton/CUDA）

```python
# 手动融合 Residual + LayerNorm + Dropout
@triton.jit
def fused_residual_ln_dropout(
    x_ptr, residual_ptr, weight_ptr, bias_ptr, output_ptr,
    N, eps, dropout_p, seed,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # 读取 x 和 residual（只读一次 HBM）
    x = tl.load(x_ptr + row * N + offsets, mask=offsets < N)
    res = tl.load(residual_ptr + row * N + offsets, mask=offsets < N)
    
    # Residual add（在寄存器中）
    x = x + res
    
    # LayerNorm（在寄存器/shared memory 中）
    mean = tl.sum(x, axis=0) / N
    var = tl.sum((x - mean) ** 2, axis=0) / N
    x_norm = (x - mean) / tl.sqrt(var + eps)
    
    # Scale + Shift
    w = tl.load(weight_ptr + offsets, mask=offsets < N)
    b = tl.load(bias_ptr + offsets, mask=offsets < N)
    x_norm = x_norm * w + b
    
    # Dropout（在寄存器中）
    random = tl.rand(seed, row * N + offsets)
    mask_drop = random > dropout_p
    x_norm = tl.where(mask_drop, x_norm / (1 - dropout_p), 0.0)
    
    # 只写一次 HBM
    tl.store(output_ptr + row * N + offsets, x_norm, mask=offsets < N)
```

### 融合决策框架

```
是否应该融合两个算子 A 和 B？

1. 数据依赖检查
   └── B 的输入是 A 的输出吗？ → 是：融合有价值
   
2. 计算类型匹配
   ├── A, B 都是 pointwise → 必须融合
   ├── A 是 GEMM, B 是 pointwise → epilogue fusion
   ├── A 是 reduction, B 是 pointwise → 可以融合
   └── A, B 都是 GEMM → 通常不融合（除非特殊情况）

3. 资源约束
   ├── 融合后 shared memory 够吗？
   ├── 融合后 register 够吗？（会不会 spill）
   └── 融合后 kernel 会不会太大（编译慢、occupancy 低）

4. 收益评估
   ├── 节省多少 HBM 访问？
   ├── 减少多少 kernel launch？
   └── 融合后 arithmetic intensity 提升多少？
```

## 实际工程中的融合案例

### vLLM 中的 PagedAttention

```
融合了：
1. KV Cache 的分页读取
2. Attention Score 计算
3. Softmax
4. Value 加权求和

为什么融合：
- 分页 KV Cache 的读取模式不规则
- 单独做每步会有大量 HBM 随机访问
- 融合后在 shared memory 中完成所有计算
```

### DeepSpeed 的 Fused Optimizer

```
融合了 Adam 优化器的所有操作：
1. 梯度读取
2. 一阶矩更新 (m = β₁m + (1-β₁)g)
3. 二阶矩更新 (v = β₂v + (1-β₂)g²)
4. 偏差修正
5. 权重更新 (w = w - lr * m̂ / (√v̂ + ε))

未融合：5个 kernel，每个都要读写所有参数（bandwidth-bound）
融合后：1个 kernel，参数只读写一次

对于 70B 模型（FP16 参数 = 140GB）：
- 未融合：5 × 2 × 140GB = 1.4TB HBM 访问
- 融合后：1 × 2 × 140GB = 280GB HBM 访问
- 节省：5x
```

## 本章要点总结

1. **算子融合**减少 HBM 访问、kernel launch overhead、提高数据局部性
2. **Pointwise fusion** 是最简单有效的融合，编译器可自动完成
3. **FlashAttention** 是融合的巅峰——通过算法重构避免 O(N²) 的 HBM 访问
4. **在线 Softmax** 是 FlashAttention 的核心技术难点
5. 编译器（torch.compile/Inductor）能自动做简单融合，复杂融合需要手动设计
6. 融合决策需要综合考虑数据依赖、资源约束和收益

## 延伸阅读

- [FlashAttention: Fast and Memory-Efficient Exact Attention (Paper)](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)
- [Online Softmax (Milakov & Gimelshein)](https://arxiv.org/abs/1805.02867)
- [PyTorch Inductor: A Compiler Backend for PyTorch](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-target-agnostic-backends/747)
