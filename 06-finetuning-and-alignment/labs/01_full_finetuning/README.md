# Lab 01: 全量微调 Baseline

## 目标
- 理解全量微调的完整流程
- 分析全量微调的显存占用
- 建立效果 baseline 用于与 LoRA/QLoRA 对比

## 硬件要求
- 1-2 × H20 (96GB)：可全量微调 7B 模型
- 8 × H20：可全量微调 13B+ 模型

## 实验内容
1. `full_ft_demo.py` — 使用 Qwen2-1.5B 进行全量微调（单卡即可）
2. `memory_analysis.py` — 详细分析各组件的显存占用

## 运行方式
```bash
# 全量微调 demo (1.5B 模型，单卡)
python full_ft_demo.py

# 显存分析
python memory_analysis.py --model_name Qwen/Qwen2-1.5B
python memory_analysis.py --model_name Qwen/Qwen2-7B
```
