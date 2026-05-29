# Lab 02: LoRA 与 QLoRA 微调

## 目标
- 掌握 LoRA 微调的完整流程
- 对比 LoRA 和 QLoRA 的效果与资源差异
- 理解 rank 选择对效果的影响
- 学会合并 LoRA 权重并导出

## 硬件要求
- LoRA: 1 × H20 (7B 模型约 30GB)
- QLoRA: 1 × H20 (70B 模型约 50-80GB)

## 实验内容
1. `lora_train.py` — 标准 LoRA 微调 Qwen2-7B
2. `qlora_train.py` — QLoRA 4-bit 微调
3. `rank_experiment.py` — 不同 rank 的对比实验
4. `merge_and_export.py` — LoRA 合并与导出

## 运行方式
```bash
# LoRA 微调
python lora_train.py

# QLoRA 微调
python qlora_train.py

# Rank 对比实验
python rank_experiment.py --ranks 8 16 32 64 128

# 合并并导出
python merge_and_export.py --base_model Qwen/Qwen2-7B --lora_path ./output/lora
```
