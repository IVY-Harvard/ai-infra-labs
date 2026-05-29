# Lab 07: DPO Training

## 目标
- 掌握 DPO 训练流程
- 学会构造偏好数据
- 对比 DPO 和 SFT 效果差异

## 硬件
- 1-2 × H20

## 运行方式
```bash
# DPO 训练
python dpo_train.py

# 构造偏好数据
python preference_data_builder.py --sft_model ./sft_output --num 1000

# DPO vs SFT 对比
python dpo_vs_sft_comparison.py --sft_model ./sft_output --dpo_model ./dpo_output
```
