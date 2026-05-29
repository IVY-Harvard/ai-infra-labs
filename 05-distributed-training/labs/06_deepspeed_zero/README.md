# Lab 06: DeepSpeed ZeRO — Stage 1/2/3 实战

## 目标

1. 使用 DeepSpeed ZeRO Stage 1/2/3 训练模型
2. 对比不同 Stage 的显存占用和通信量
3. 理解 ZeRO 的配置参数调优

## 前置知识

- 对应理论：[theory/07_deepspeed_zero.md](../../theory/07_deepspeed_zero.md)

## 文件说明

| 文件 | 说明 |
|------|------|
| `zero1_train.py` | ZeRO Stage 1: 优化器状态切分 |
| `zero2_train.py` | ZeRO Stage 2: 梯度 + 优化器切分 |
| `zero3_train.py` | ZeRO Stage 3: 全切分 |
| `memory_breakdown.py` | 各 Stage 显存拆解分析 |

## 运行

```bash
deepspeed --num_gpus=8 zero1_train.py
deepspeed --num_gpus=8 zero2_train.py
deepspeed --num_gpus=8 zero3_train.py
deepspeed --num_gpus=8 memory_breakdown.py
```

## 核心对比

| Stage | 每 GPU 显存 (7B) | 通信量 | 编程复杂度 |
|-------|----------------|--------|-----------|
| DDP | 112 GB | 2M | 低 |
| Stage 1 | 77 GB | 3M | 低 |
| Stage 2 | 52.5 GB | 2M | 低 |
| Stage 3 | 28 GB (峰值 42) | 3M | 中 |
