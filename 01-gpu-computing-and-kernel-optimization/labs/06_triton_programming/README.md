# Lab 06: Triton 编程入门

## 实验目的

用 OpenAI Triton 实现三个经典 GPU kernel，体验"用 Python 写 GPU 内核"的编程模型：
1. 向量加法 — 最简单的 Triton kernel
2. 矩阵乘法 — 展示 tiling 和 autotune
3. Fused Softmax — 展示算子融合

## 前置要求

- PyTorch 2.0+
- Triton: `pip install triton`
- 已读 theory/05（Triton 编程模型）

## 运行

```bash
python vector_add_triton.py
python matmul_triton.py
python fused_softmax.py
```

## 核心概念

- `@triton.jit`: 声明 Triton kernel
- `tl.program_id(axis)`: 获取当前程序实例 ID（类似 blockIdx）
- `tl.arange(0, BLOCK)`: 生成偏移量数组
- `tl.load / tl.store`: 内存读写（支持 mask 边界处理）
- `tl.constexpr`: 编译期常量（用于 autotune）
