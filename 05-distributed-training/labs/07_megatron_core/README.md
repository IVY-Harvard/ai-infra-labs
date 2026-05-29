# Lab 07: Megatron-Core 入门

## 目标

1. 了解 Megatron-Core 的架构和核心概念
2. 理解 Sequence Parallelism (SP) 如何配合 TP 节省激活值显存
3. 体验 Megatron-Core 的使用方式

## 文件说明

| 文件 | 说明 |
|------|------|
| `megatron_setup_guide.md` | Megatron-Core 安装和配置指南 |
| `sequence_parallel_demo.py` | Sequence Parallelism 原理演示 |

## 核心概念

Megatron-Core 在 TP 基础上增加了 Sequence Parallelism:
- TP 切分了权重和 GEMM 中的激活值
- 但 LayerNorm、Dropout 的输入仍是完整的 [B, S, H]
- SP 将这些操作按 sequence 维度切分
- 用 ReduceScatter + AllGather 替代 AllReduce
