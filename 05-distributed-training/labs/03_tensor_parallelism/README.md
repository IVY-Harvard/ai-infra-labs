# Lab 03: 张量并行手写实现

## 目标

1. 从零手写列并行 Linear 和行并行 Linear，理解 TP 的数学原理
2. 用列并行 + 行并行组合搭建一个 Transformer MLP 块和 Attention 块
3. 分析 TP 的通信量和带宽需求，理解为什么 TP 必须用 NVLink

## 前置知识

- 完成 Lab 01-02
- 对应理论：[theory/03_tensor_parallelism.md](../../theory/03_tensor_parallelism.md)

## 文件说明

| 文件 | 说明 |
|------|------|
| `column_parallel_linear.py` | 手写列并行 Linear，包含 `f` 操作符 |
| `row_parallel_linear.py` | 手写行并行 Linear，包含 `g` 操作符 |
| `tp_transformer_block.py` | 用 TP 版本搭建完整 Transformer 块 |
| `communication_analysis.py` | 测量和分析 TP 的实际通信开销 |

## 实验步骤

### 实验 1：列并行

```bash
torchrun --nproc_per_node=4 column_parallel_linear.py
```

### 实验 2：行并行

```bash
torchrun --nproc_per_node=4 row_parallel_linear.py
```

### 实验 3：完整 Transformer 块

```bash
torchrun --nproc_per_node=4 tp_transformer_block.py
```

### 实验 4：通信分析

```bash
torchrun --nproc_per_node=4 communication_analysis.py
```

## 核心公式

```
列并行: Y_i = X @ W_i        (前向无通信，反向 AllReduce 输入梯度)
行并行: Y = sum(X_i @ W_i)   (前向 AllReduce 输出，反向无额外通信)

每层通信: 前向 2 次 AllReduce + 反向 2 次 AllReduce = 4 次 AllReduce
每次数据量: B × S × H × dtype_bytes
```

## 思考题

1. 为什么 MLP 中 W1 用列并行而 W2 用行并行？能反过来吗？
2. GeLU 为什么可以在列并行后本地计算？如果换成 Sigmoid 呢？
3. TP=4 和 TP=8 哪个通信开销更大？直觉上是 8，但实际呢？
