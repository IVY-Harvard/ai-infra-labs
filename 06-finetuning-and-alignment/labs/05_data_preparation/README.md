# Lab 05: 数据准备工具集

## 目标
- 掌握多种数据格式的转换
- 学会数据质量检查和过滤
- 了解合成数据生成方法

## 工具
1. `data_converter.py` — 多格式转换（Alpaca/ShareGPT/Messages 互转）
2. `data_quality_checker.py` — 数据质量自动检查
3. `synthetic_data_gen.py` — 使用 LLM 生成合成数据

## 运行方式
```bash
# 格式转换
python data_converter.py --input data.json --from alpaca --to sharegpt --output converted.json

# 质量检查
python data_quality_checker.py --input data.jsonl --report quality_report.json

# 合成数据生成
python synthetic_data_gen.py --seed_file seeds.json --num_samples 1000 --output synthetic.jsonl
```
