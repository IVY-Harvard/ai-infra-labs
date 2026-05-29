# Lab 03: Unsloth 加速微调

## 目标
- 使用 Unsloth 实现 2-4x 训练加速
- 对比 Unsloth 与标准 HuggingFace 的速度差异
- 理解 Unsloth 的优化原理

## Unsloth 原理
Unsloth 通过手写 Triton CUDA kernel 优化:
- 融合 LoRA 前向/反向计算
- 优化注意力计算
- 减少内存拷贝和碎片

## 运行方式
```bash
# Unsloth 加速微调
python unsloth_finetune.py

# 速度对比实验
python speed_comparison.py
```
