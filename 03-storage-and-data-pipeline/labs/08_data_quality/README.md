# Lab 08：数据质量工具

## 实验目标

实现数据去重、质量打分和数据画像三大数据质量工具。

## 实验内容

### 实验 1：MinHash 近似去重
使用 MinHash + LSH 对文本数据做大规模近似去重。

### 实验 2：质量打分器
多维度评估文本质量（语言质量、信息密度、重复度）。

### 实验 3：数据画像
统计数据集的分布特征，生成数据画像报告。

## 运行方式

```bash
pip install numpy datasketch

# MinHash 去重
python dedup_minhash.py --input-dir /path/to/texts --threshold 0.8

# 质量打分
python quality_scorer.py --input-file /path/to/data.jsonl --output scored_data.jsonl

# 数据画像
python data_profiler.py --input-dir /path/to/data
```

## 文件列表

- `dedup_minhash.py` — MinHash 近似去重
- `quality_scorer.py` — 多维度质量打分
- `data_profiler.py` — 数据集画像工具
