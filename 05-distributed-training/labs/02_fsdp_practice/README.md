# Lab 02: FSDP 实践 — 全切片数据并行

## 目标

1. 掌握 PyTorch FSDP 的配置和使用
2. 对比 DDP 和 FSDP 的显存占用，理解切分带来的收益
3. 理解 FSDP 的 ShardingStrategy 选择逻辑

## 前置知识

- 完成 Lab 01（DDP）
- 对应理论：[theory/02_data_parallelism.md](../../theory/02_data_parallelism.md)

## 环境要求

- 8 × H20 GPU
- PyTorch >= 2.1（支持 FSDP2）

## 文件说明

| 文件 | 说明 |
|------|------|
| `fsdp_train.py` | FSDP 训练脚本，支持多种 ShardingStrategy |
| `memory_comparison.py` | DDP vs FSDP 显存对比实验 |

## 实验步骤

### 实验 1：FSDP 基础训练

```bash
# FULL_SHARD (ZeRO-3 等效)
torchrun --nproc_per_node=8 fsdp_train.py --sharding full

# SHARD_GRAD_OP (ZeRO-2 等效)
torchrun --nproc_per_node=8 fsdp_train.py --sharding grad_op

# NO_SHARD (等效 DDP)
torchrun --nproc_per_node=8 fsdp_train.py --sharding no_shard
```

### 实验 2：显存对比

```bash
torchrun --nproc_per_node=8 memory_comparison.py
```

## 思考题

1. FSDP FULL_SHARD 的通信量为 3M（比 DDP 的 2M 多 50%），多出来的是什么？
2. 为什么 FSDP 前向传播要 AllGather 参数？反向传播为什么还要再 AllGather 一次？
3. `auto_wrap_policy` 如何影响显存和通信效率？
