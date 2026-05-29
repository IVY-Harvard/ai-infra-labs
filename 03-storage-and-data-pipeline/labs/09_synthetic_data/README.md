# Lab 09：合成数据生成

## 实验目标

实现 Self-Instruct 和 Evol-Instruct 两种合成数据生成方法，并配合质量过滤生成高质量训练数据。

## 实验内容

### 实验 1：Self-Instruct
从种子指令集出发，让 LLM 自动生成新的指令和回复。

### 实验 2：Evol-Instruct
通过进化策略逐步提升指令的复杂度和多样性。

### 实验 3：质量过滤
对合成数据进行多维度过滤，确保数据质量。

## 运行方式

```bash
pip install openai numpy

# Self-Instruct（需要 LLM API）
python self_instruct.py --seed-file seeds.jsonl --output generated_data.jsonl --num-generate 100

# Evol-Instruct
python evol_instruct.py --input instructions.jsonl --output evolved_data.jsonl --rounds 3

# 质量过滤
python quality_filter.py --input generated_data.jsonl --output filtered_data.jsonl --min-score 0.7
```

## 文件列表

- `self_instruct.py` — Self-Instruct 数据生成
- `evol_instruct.py` — Evol-Instruct 进化生成
- `quality_filter.py` — 合成数据质量过滤
