# Lab 01: DDP 基础 — 从单卡到多卡

## 目标

1. 建立单卡训练基线，理解训练循环各环节的显存和时间开销
2. 将单卡代码改造为 DDP 多卡训练，观察线性扩展效果
3. 深入观察梯度同步过程，理解 Gradient Bucketing 和 AllReduce

## 前置知识

- PyTorch 基础训练循环（forward / backward / optimizer.step）
- 对应理论：[theory/01_parallelism_overview.md](../../theory/01_parallelism_overview.md)、[theory/02_data_parallelism.md](../../theory/02_data_parallelism.md)

## 环境要求

- 8 × H20 GPU（NVLink 互联）
- PyTorch >= 2.1
- NCCL backend

## 文件说明

| 文件 | 说明 |
|------|------|
| `single_gpu.py` | 单卡训练基线：一个简单的 Transformer 模型在单 GPU 上训练 |
| `ddp_train.py` | DDP 多卡训练：最少改动将单卡代码迁移到 DDP |
| `gradient_sync_demo.py` | 梯度同步可视化：观察 AllReduce 前后的梯度差异、Bucketing 行为 |

## 实验步骤

### 实验 1：单卡基线

```bash
python single_gpu.py --epochs 3 --batch-size 32
```

**观察**：
- 每步训练时间
- GPU 显存占用（nvidia-smi）
- 训练 loss 收敛曲线

### 实验 2：DDP 多卡

```bash
# 2 卡
torchrun --nproc_per_node=2 ddp_train.py --epochs 3 --batch-size 32

# 4 卡
torchrun --nproc_per_node=4 ddp_train.py --epochs 3 --batch-size 32

# 8 卡
torchrun --nproc_per_node=8 ddp_train.py --epochs 3 --batch-size 32
```

**观察**：
- 吞吐量是否线性扩展（tokens/sec）
- 每张卡的显存占用是否与单卡一致
- 全局 batch size = per_gpu_batch × num_gpus，对 loss 的影响

### 实验 3：梯度同步

```bash
torchrun --nproc_per_node=4 gradient_sync_demo.py
```

**观察**：
- AllReduce 前各 rank 的梯度是否不同
- AllReduce 后各 rank 的梯度是否一致
- Bucket 大小对通信时间的影响

## 思考题

1. DDP 的通信量为 `2 × model_size`，与 GPU 数量无关——为什么？（提示：Ring AllReduce）
2. 为什么 DDP 在反向传播**过程中**就开始通信，而不是等反向传播完成？
3. 如果全局 batch size 随卡数线性增长，可能带来什么问题？如何解决？
4. 如果某张卡上的数据比其他卡多一条（数据不均匀），DDP 会怎样？

## 预期输出

```
=== Single GPU Baseline ===
Throughput: ~2500 tokens/sec
Memory: ~8.2 GB

=== DDP 8 GPU ===
Throughput: ~18000 tokens/sec (7.2x speedup)
Memory per GPU: ~8.2 GB (unchanged)
```
