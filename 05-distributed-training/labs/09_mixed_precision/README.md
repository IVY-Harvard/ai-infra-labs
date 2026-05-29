# Lab 09: 混合精度训练

## 目标

1. 理解 FP16 和 BF16 的区别，以及各自的适用场景
2. 实现 Loss Scaling 机制，理解为什么 FP16 需要而 BF16 不需要
3. 分析混合精度对显存和吞吐量的影响

## 文件说明

| 文件 | 说明 |
|------|------|
| `fp16_training.py` | FP16 混合精度训练（带 Loss Scaling）|
| `bf16_training.py` | BF16 混合精度训练 |
| `loss_scaling_demo.py` | Loss Scaling 机制原理演示 |
| `memory_saving_analysis.py` | 混合精度显存节省分析 |

## 运行

```bash
torchrun --nproc_per_node=4 fp16_training.py
torchrun --nproc_per_node=4 bf16_training.py
python loss_scaling_demo.py
python memory_saving_analysis.py
```

## 核心知识

```
FP32: 1 sign + 8 exp + 23 mantissa  →  范围大，精度高
FP16: 1 sign + 5 exp + 10 mantissa  →  范围小 (6.5e4)，需要 loss scaling
BF16: 1 sign + 8 exp + 7 mantissa   →  范围同 FP32，精度低

H20 推荐: BF16（范围够大，不需要 loss scaling，简单可靠）
```
