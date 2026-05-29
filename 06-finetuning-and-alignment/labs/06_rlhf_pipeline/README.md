# Lab 06: RLHF Pipeline

## 目标
- 理解 Reward Model 的训练流程
- 实践 PPO 训练
- 学会分析 Reward 分布

## 硬件要求
- 2-4 × H20：PPO 需要同时加载 policy + ref + RM

## 实验内容
1. `reward_model_train.py` — Reward Model 训练
2. `ppo_train.py` — PPO 训练
3. `reward_analysis.py` — Reward 分布分析

## 运行方式
```bash
# 训练 Reward Model
python reward_model_train.py

# PPO 训练（需要先有 SFT 模型和 RM）
python ppo_train.py --sft_model ./sft_output --reward_model ./rm_output

# 分析 Reward 分布
python reward_analysis.py --model ./rm_output --data test_prompts.jsonl
```
