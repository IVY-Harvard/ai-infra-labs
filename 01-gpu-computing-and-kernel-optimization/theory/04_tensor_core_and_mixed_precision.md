# 04 - Tensor Core 与混合精度

## 为什么 Tensor Core 改变了 AI 计算？

在 Tensor Core 出现之前（Pascal 架构及更早），GPU 做矩阵乘全靠 CUDA Core：
- 一个 CUDA Core 每周期做 1 次 FP32 FMA（Fused Multiply-Add）
- 4096×4096 的矩阵乘需要 ~137 GFLOPs
- 靠堆 CUDA Core 数量来提升算力

Tensor Core 改变了游戏规则：
- 一次操作完成 4×4 矩阵乘加（64 次 FMA）
- 同样的面积，算力提升 8-16 倍
- 但代价是：精度受限，只支持特定数据类型

**对 AI Infra 工程师的含义**：你的代码能不能利用 Tensor Core，直接决定了 8-16x 的性能差距。

## Tensor Core 工作原理

### 基本操作：MMA (Matrix Multiply-Accumulate)

```
D = A × B + C

其中：
A: M×K 矩阵（输入类型）
B: K×N 矩阵（输入类型）
C: M×N 矩阵（累加类型）
D: M×N 矩阵（输出类型）
```

### 各代 Tensor Core

| 代次 | 架构 | 基本操作形状 | 支持类型 |
|------|------|-------------|----------|
| 1st | Volta (V100) | 4×4×4 | FP16→FP32 |
| 2nd | Turing (T4) | 8×8×4 | FP16, INT8, INT4 |
| 3rd | Ampere (A100) | 8×8×4 | FP16, BF16, TF32, INT8, INT4 |
| 4th | Hopper (H100/H20) | 16×16×16 | FP8, FP16, BF16, TF32, INT8 |

### 第四代 Tensor Core (Hopper)

```
单次操作：
D[16×16] = A[16×16] × B[16×16] + C[16×16]

实际执行（在 warp 级别）：
- 一个 Warp（32 线程）协作完成整个 MMA
- 每个线程持有 A, B, C, D 矩阵的一部分片段（fragment）
- 通过 Warp 内 shuffle 交换数据
- Tensor Core 硬件执行矩阵乘加
```

### WMMA API

```cuda
#include <mma.h>
using namespace nvcuda::wmma;

// 声明矩阵片段
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;

// 初始化累加器
fill_fragment(c_frag, 0.0f);

// 从全局内存加载 A 和 B
load_matrix_sync(a_frag, A_ptr, lda);
load_matrix_sync(b_frag, B_ptr, ldb);

// 执行矩阵乘加
mma_sync(c_frag, a_frag, b_frag, c_frag);

// 存储结果
store_matrix_sync(D_ptr, c_frag, ldd, mem_row_major);
```

## 数据类型深度解析

### 浮点格式对比

```
FP32 (32 bits): [1 sign][8 exponent][23 mantissa]
  范围: ±3.4×10³⁸, 精度: ~7 位十进制
  
FP16 (16 bits): [1 sign][5 exponent][10 mantissa]
  范围: ±65504, 精度: ~3.3 位十进制
  问题: 动态范围小，容易 overflow/underflow

BF16 (16 bits): [1 sign][8 exponent][7 mantissa]
  范围: ±3.4×10³⁸ (同FP32!), 精度: ~2.4 位十进制
  优势: 动态范围与FP32相同，训练稳定性好

TF32 (19 bits, 但存储为32bit): [1 sign][8 exponent][10 mantissa]
  范围: ±3.4×10³⁸, 精度: ~3.3 位十进制
  特殊: 只在 Tensor Core 内部使用，外部看是FP32

FP8 (E4M3): [1 sign][4 exponent][3 mantissa]
  范围: ±448, 精度: ~1.7 位十进制
  用于: 推理中的权重和激活值

FP8 (E5M2): [1 sign][5 exponent][2 mantissa]  
  范围: ±57344, 精度: ~1.2 位十进制
  用于: 训练中的梯度（需要更大动态范围）

INT8 (8 bits): [-128, 127]
  用于: 量化推理
```

### 精度-性能权衡

```
H20 上各精度的理论算力：
FP64:            ~22 TFLOPS
FP32 (CUDA):    ~44 TFLOPS
TF32 (Tensor):  ~74 TFLOPS
FP16 (Tensor):  ~148 TFLOPS
INT8 (Tensor):  ~296 TOPS
FP8 (Tensor):   ~296 TFLOPS

性能倍率（相对FP32 CUDA Core）：
FP32  →  1x (基准)
TF32  →  ~1.7x (透明加速，无需改代码)
FP16  →  ~3.4x
INT8  →  ~6.7x
FP8   →  ~6.7x
```

## 混合精度训练 (Mixed Precision Training)

### 核心思想

```
前向传播: FP16 计算（快）
反向传播: FP16 计算（快）
权重更新: FP32 累加（精确）
Loss Scaling: 防止 FP16 梯度 underflow
```

### 为什么需要混合？

纯 FP16 训练的问题：
1. **梯度下溢**：小梯度（如 1e-8）在 FP16 中变成 0
2. **权重更新消失**：当 weight >> gradient 时，FP16 加法精度不够
3. **累加误差**：大量小数相加，FP16 累计误差显著

### PyTorch AMP (Automatic Mixed Precision)

```python
import torch
from torch.cuda.amp import autocast, GradScaler

model = model.cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()  # Loss Scaling

for data, target in dataloader:
    optimizer.zero_grad()
    
    # 自动选择 FP16/FP32
    with autocast():
        output = model(data)    # FP16 forward
        loss = criterion(output, target)  # FP32 loss
    
    # Scaled backward
    scaler.scale(loss).backward()  # FP16 gradients, scaled
    scaler.step(optimizer)          # Unscale → FP32 update
    scaler.update()                 # 调整 scale factor
```

### Loss Scaling 原理

```
问题：梯度太小 (e.g., 1e-8)，FP16 表示为 0
解决：
1. Loss × scale_factor (e.g., 1024)
2. 所有梯度自动 × 1024（chain rule）
3. 更新前 ÷ 1024 恢复真实梯度

动态 Loss Scaling：
- 如果没有 overflow → 尝试增大 scale
- 如果出现 overflow → 减小 scale，跳过这步更新
```

## BF16 vs FP16：工程选择

### BF16 的优势

```
场景: 训练 LLM（如 GPT-3/4, LLaMA）

FP16 问题:
- 动态范围只有 ±65504
- 大模型中 activation 经常超范围
- 需要 Loss Scaling 才能稳定训练

BF16 优势:
- 动态范围 = FP32 (±3.4×10³⁸)
- 不需要 Loss Scaling
- 训练更稳定，超参数更容易调

BF16 劣势:
- 精度比 FP16 更低（7 vs 10 mantissa bits）
- 某些精度敏感的操作可能有问题
```

### 工程实践建议

```
训练:
├── 大模型 (>1B params)     → BF16 (稳定，不需 loss scaling)
├── 小模型 (<100M params)   → FP16 + AMP (精度够用)
└── 学术复现                → 按原论文选择

推理:
├── 首选 INT8 量化          → 最高性价比
├── FP16                   → 精度要求高时
├── FP8 (Hopper)           → 极致性能
└── 动态量化               → 针对不同层选不同精度
```

## TF32：透明加速

### 什么是 TF32？

TF32 是 Ampere/Hopper 架构的独特设计：
- 输入以 FP32 格式存储
- Tensor Core 自动截取高 19 位（1+8+10）用于计算
- 输出仍是 FP32
- **完全透明**：不需要改任何代码

```
FP32:  [1 sign][8 exp][23 mantissa]
TF32:  [1 sign][8 exp][10 mantissa] ← 截断低13位

精度损失: 从 ~7 位有效数字降到 ~3.3 位
速度提升: ~2x (A100), ~1.7x (H20)
```

### 使用方式

```python
# PyTorch 中默认开启 TF32 (Ampere+)
torch.backends.cuda.matmul.allow_tf32 = True   # 默认 True
torch.backends.cudnn.allow_tf32 = True          # 默认 True

# 如果需要精确 FP32（如数值验证）
torch.backends.cuda.matmul.allow_tf32 = False
```

## FP8 推理（Hopper 专属）

### FP8 的两种格式

```
E4M3: 4-bit exponent, 3-bit mantissa
├── 范围: ±448
├── 精度: 较高（相对E5M2）
└── 用途: 权重和前向激活值

E5M2: 5-bit exponent, 2-bit mantissa
├── 范围: ±57344
├── 精度: 较低
└── 用途: 梯度（训练时）
```

### FP8 量化推理流程

```python
# 使用 NVIDIA TensorRT-LLM 的 FP8 推理
# 1. 校准：用少量数据确定每层的 scale factor
# 2. 量化：权重和激活值转为 FP8
# 3. 推理：Tensor Core 直接做 FP8 计算

# 概念流程
for layer in model.layers:
    # FP8 GEMM: 输入FP8, 权重FP8, 输出FP16/FP32
    output = fp8_gemm(input_fp8, weight_fp8, scale_a, scale_b)
    # 非线性操作在 FP16/FP32 下进行
    output = activation(output.to(fp16))
    # 重新量化为 FP8
    input_fp8 = quantize_to_fp8(output, compute_scale(output))
```

## INT8 量化推理

### 量化基础

```
量化公式:
x_int8 = round(x_float / scale)
x_dequant = x_int8 * scale

其中: scale = max(|x_float|) / 127

对称量化: zero_point = 0
非对称量化: x_int8 = round(x / scale) + zero_point
```

### Per-Tensor vs Per-Channel vs Per-Token

```
Per-Tensor: 整个张量一个 scale
├── 简单，但精度损失大
└── 适合权重分布均匀的场景

Per-Channel: 每个 output channel 一个 scale
├── 精度好，权重量化常用
└── GEMM: W[out, in] 按 out 维度各有 scale

Per-Token: 每个 token 一个 scale
├── 动态量化，适合激活值
└── 每次推理重新计算 scale
```

### SmoothQuant 技术

```
问题: 激活值中有 outlier（异常大的值），量化误差大
解决: 将 outlier 的"困难"从激活值转移到权重

Y = X × W
  = (X / s) × (s × W)    # s 是平滑因子
  = X̃ × W̃

X̃ = X / s  → 激活值范围变小，量化友好
W̃ = s × W  → 权重吸收了 scale，offline 处理
```

## 对齐要求与性能

### 为什么维度要对齐？

Tensor Core 操作固定的矩阵尺寸。如果你的矩阵维度不对齐，需要 padding，浪费计算。

```
Tensor Core 对齐要求:
├── FP16: M, N, K 必须是 16 的倍数
├── INT8:  M, N, K 必须是 16 的倍数
├── FP8:   M, N, K 必须是 16 的倍数
└── TF32:  M, N, K 必须是 8 的倍数

工程含义:
- hidden_size 选择 128/256/512/1024/2048/4096...
- head_dim 通常是 64 或 128
- vocab_size 补齐到 128 的倍数
- batch_size 最好是 8 的倍数
```

### 性能 cliff 示例

```
矩阵乘 M×N×K = 4096×4096×4096 → 100% Tensor Core 利用
矩阵乘 M×N×K = 4097×4097×4097 → 可能退化到 CUDA Core!

解决: Padding 到对齐边界
padding = (align - dim % align) % align
```

## 本章要点总结

1. **Tensor Core** 是 AI 计算的核心加速器，提供 8-16x 加速
2. **数据类型选择**直接决定算力上限：FP32 < TF32 < FP16/BF16 < INT8/FP8
3. **混合精度训练**用 FP16 计算 + FP32 累加 + Loss Scaling
4. **BF16** 是大模型训练的首选（动态范围大，无需 loss scaling）
5. **TF32** 是免费的 ~2x 加速（默认开启，透明替换 FP32）
6. **INT8/FP8** 是推理的性能-精度最优解
7. **维度对齐**到 8/16 的倍数，否则 Tensor Core 无法利用

## 延伸阅读

- [NVIDIA Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)
- [FP8 Formats for Deep Learning (NVIDIA)](https://arxiv.org/abs/2209.05433)
- [SmoothQuant Paper](https://arxiv.org/abs/2211.10438)
- [NVIDIA Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/)
