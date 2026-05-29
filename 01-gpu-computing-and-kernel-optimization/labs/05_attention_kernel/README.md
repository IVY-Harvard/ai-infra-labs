# Lab 05: Attention 内核分析

## 实验目的

1. 实现朴素 Attention，理解其计算流程和内存访问模式
2. 分析 FlashAttention 为什么快 —— 不是因为减少了 FLOPs，而是减少了 HBM 访问
3. 量化 O(N^2) 中间矩阵的显存开销

## 前置要求

- PyTorch
- `pip install flash-attn`（可选，用于实际性能对比）
- 已读 theory/06（FlashAttention 部分）

## 运行

```bash
python naive_attention.py
python flash_attention_analysis.py
```

## 关键问题

1. 朴素 Attention 中 S = Q @ K^T 的大小是多少？序列长度翻倍后呢？
2. 为什么 FlashAttention 不需要存储完整的 N×N 矩阵？
3. Online Softmax 的核心思想是什么？
