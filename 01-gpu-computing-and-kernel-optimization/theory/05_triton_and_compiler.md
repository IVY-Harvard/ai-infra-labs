# 05 - Triton 与编译器生态

## 为什么需要 Triton？

### CUDA 的门槛问题

写高性能 CUDA kernel 需要：
- 理解硬件细节（shared memory bank conflict, register spilling, warp scheduling）
- 手动管理内存层次（global → shared → register）
- 处理大量低级细节（索引计算、边界检查、同步）
- 调优大量参数（block size, tile size, unroll factor, vectorize width）

一个优化良好的 GEMM kernel 通常需要 500-1000 行 CUDA 代码，而 cuBLAS 的实现更是数万行。

### Triton 的定位

```
抽象层次:
High    ┌──────────────────┐
        │ PyTorch/TF       │  ← 框架用户
        ├──────────────────┤
        │ Triton           │  ← AI Infra 工程师 ★
        ├──────────────────┤
        │ CUDA C++         │  ← GPU 专家
        ├──────────────────┤
        │ PTX/SASS         │  ← 硬件工程师
Low     └──────────────────┘
```

Triton 的核心价值：
- **用 Python 写 GPU kernel**：大幅降低门槛
- **自动处理**：内存 coalescing、shared memory 管理、同步
- **块级编程**：你思考的粒度是 "数据块" 而非 "单个线程"
- **接近 CUDA 的性能**：大多数场景达到手写 CUDA 的 90%+

## Triton 编程模型

### 核心概念：Block-Level Programming

CUDA 中你思考的是"每个线程做什么"，Triton 中你思考的是"每个程序实例处理哪个数据块"。

```python
# CUDA 思维：
# "线程 i 处理元素 i"
# __global__ void add(float* a, float* b, float* c, int n) {
#     int i = blockIdx.x * blockDim.x + threadIdx.x;
#     if (i < n) c[i] = a[i] + b[i];
# }

# Triton 思维：
# "程序实例 pid 处理从 offset 开始的 BLOCK_SIZE 个元素"
@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)
```

### 关键区别

| 维度 | CUDA | Triton |
|------|------|--------|
| 编程粒度 | 线程 | 数据块 |
| 内存管理 | 手动 shared memory | 自动 (编译器决定) |
| 同步 | 手动 `__syncthreads()` | 自动 |
| 向量化 | 手动 `float4` | 自动 |
| 边界处理 | 手动 if 判断 | mask 机制 |
| 调优 | 手动选参数 | Autotuning |

### tl.constexpr：编译期常量

```python
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr):
    # BLOCK_SIZE 在编译时确定
    # 编译器可以据此做循环展开、寄存器分配等优化
    pass

# 不同的 BLOCK_SIZE 会编译出不同的 kernel binary
# autotune 就是尝试多个值，选最快的
```

### Auto-tuning

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],  # 当这些值变化时重新 tune
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...):
    ...
```

## Triton 编译流程

```
Python AST (Triton-decorated function)
       ↓
Triton IR (Triton 中间表示)
       ↓
TTIR → TTGIR (GPU-specific IR)
       ↓
LLVM IR (通过 MLIR)
       ↓
PTX (NVIDIA 汇编)
       ↓
CUBIN (GPU binary)
```

每一步都有优化 pass：
- **Triton IR → TTGIR**：tile 大小决定、shared memory 分配
- **TTGIR → LLVM**：循环展开、向量化、指令调度
- **LLVM → PTX**：寄存器分配、指令选择

## torch.compile / PyTorch 2.0 编译栈

### 动机

PyTorch 的 eager mode（即时执行）有性能问题：
1. 每个算子单独启动一个 kernel → kernel launch overhead
2. 中间结果都存在 HBM → 带宽浪费
3. 无法跨算子优化

### 编译栈架构

```
用户代码: model(x)
       ↓
┌─── TorchDynamo ───┐
│ Python Bytecode    │  ← 拦截 Python 字节码
│ → FX Graph         │  ← 生成计算图
└────────────────────┘
       ↓
┌─── AOTAutograd ───┐
│ 自动微分展开       │  ← 前向+反向都变成图
│ → Aten IR          │  ← 底层算子表示
└────────────────────┘
       ↓
┌─── Inductor ──────┐
│ 算子融合           │  ← 识别可融合的算子
│ 内存规划           │  ← 优化内存分配
│ 代码生成           │  ← 生成 Triton kernel
└────────────────────┘
       ↓
Triton Kernel (自动生成)
       ↓
PTX/CUBIN
```

### TorchDynamo 工作原理

```python
# TorchDynamo 通过 Python frame evaluation hook 拦截执行

@torch.compile
def fn(x, y):
    z = x + y          # → 记录到计算图
    if z.sum() > 0:    # → graph break! (动态控制流)
        return z * 2   # → 新的子图
    return z

# Dynamo 会将代码分成多个子图（subgraph）
# 每个子图内没有 Python 副作用，可以编译优化
# graph break 时回退到 eager mode
```

### Inductor 后端

Inductor 是 PyTorch 默认的编译后端，它：
1. 对计算图做**算子融合**（point-wise ops 融合成一个 kernel）
2. 生成**Triton 代码**（GPU）或 **C++/OpenMP 代码**（CPU）
3. 利用 Triton 的 autotune 选择最佳参数

```python
# 用户代码
def fn(x):
    y = x.relu()
    z = y * 2
    w = z + 1
    return w

# Inductor 融合后生成的 Triton kernel（示意）:
@triton.jit
def fused_kernel(x_ptr, w_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets)
    # relu + mul + add 融合在一个 kernel 中
    y = tl.maximum(x, 0.0)
    z = y * 2.0
    w = z + 1.0
    tl.store(w_ptr + offsets, w)
```

### torch.compile 使用

```python
import torch

model = MyModel().cuda()

# 基本用法
compiled_model = torch.compile(model)

# 指定模式
compiled_model = torch.compile(model, mode="reduce-overhead")  # 减少开销
compiled_model = torch.compile(model, mode="max-autotune")     # 最大性能

# 指定后端
compiled_model = torch.compile(model, backend="inductor")      # 默认
compiled_model = torch.compile(model, backend="eager")         # 不编译（debug用）

# 查看生成的代码
import torch._inductor.config
torch._inductor.config.debug = True  # 会打印生成的 Triton 代码
```

## 更广泛的编译器生态

### TVM (Apache TVM)

```
定位：跨硬件的深度学习编译器

用户模型 (PyTorch/TF/ONNX)
       ↓
Relay IR (高级图 IR)
       ↓
TE (Tensor Expression) / TIR (低级 IR)
       ↓
AutoTVM / Ansor (自动调优)
       ↓
目标代码 (CUDA/OpenCL/Metal/LLVM)
```

TVM 的优势：
- 跨平台：同一模型部署到 GPU/CPU/ARM/FPGA
- 自动调优：Ansor 能搜索出接近手写的性能
- 适合边缘部署和国产芯片

### XLA (Accelerated Linear Algebra)

```
定位：Google 的 ML 编译器，主要用于 TPU

JAX/TensorFlow 代码
       ↓
HLO IR (High Level Operations)
       ↓
优化 Pass (算子融合、内存优化)
       ↓
目标代码 (TPU/GPU/CPU)
```

XLA 的特点：
- 与 JAX 紧密集成
- TPU 的唯一编译路径
- 全图编译（需要静态 shape）

### MLIR (Multi-Level Intermediate Representation)

```
定位：编译器基础设施框架（不是编译器本身）

MLIR 是一个"建造编译器的工具"：
├── 提供多层 IR 的框架
├── 各团队可以定义自己的 Dialect
├── 提供通用的 Pass 基础设施
└── 被多个项目使用：
    ├── TensorFlow (MLIR-HLO)
    ├── Triton (Triton Dialect → LLVM Dialect)
    ├── IREE (Google)
    └── Torch-MLIR
```

### 编译器选择指南

| 场景 | 推荐 | 原因 |
|------|------|------|
| PyTorch 模型加速 | torch.compile | 生态集成最好 |
| 自定义算子 | Triton | 开发效率高 |
| 极致性能 | CUDA + cuBLAS | 上限最高 |
| 跨硬件部署 | TVM | 支持硬件最多 |
| TPU 场景 | XLA/JAX | 唯一选择 |
| 国产芯片适配 | TVM / 厂商编译器 | 需要具体情况分析 |

## Triton vs CUDA：什么时候用哪个？

### 用 Triton 的场景

```
1. Pointwise 融合 kernel（如 fused LayerNorm + Dropout + Residual）
2. 自定义 Attention 变体
3. 需要快速迭代验证想法
4. 性能要求 90%+ of CUDA
5. 团队 CUDA 经验有限
```

### 仍需 CUDA 的场景

```
1. 需要极致性能（如 FlashAttention 的正式实现）
2. 复杂的 warp-level 原语操作
3. 需要精确控制共享内存 layout
4. 非标准内存访问模式
5. 需要 inline PTX
```

### 性能对比（大致）

```
场景                    Triton vs 手写CUDA
向量加法                 ~100% (编译器很擅长)
矩阵乘法 (大)           ~95% (接近cuBLAS)
矩阵乘法 (小/不规则)     ~85-95%
Softmax                 ~95-100%
FlashAttention          ~85-90% (复杂kernel差距大)
自定义融合 kernel        ~90-100%
```

## 本章要点总结

1. **Triton** 让你用 Python 写出接近 CUDA 性能的 GPU kernel
2. 核心思想是**块级编程**：你处理数据块，编译器处理线程细节
3. **torch.compile = Dynamo + AOTAutograd + Inductor**，自动将 PyTorch 代码编译优化
4. Inductor 生成 **Triton kernel**，自动做算子融合
5. 更广泛的编译器生态：**TVM**（跨平台）、**XLA**（TPU）、**MLIR**（基础设施）
6. 选择标准：开发效率 vs 极致性能 vs 硬件覆盖

## 延伸阅读

- [Triton Official Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [PyTorch 2.0 torch.compile Tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [TVM Documentation](https://tvm.apache.org/docs/)
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations (Paper)](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
