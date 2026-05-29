# Lab 07: 推理量化实践

## 目标
- 实测 GPTQ/AWQ/FP8 量化效果
- 对比量化前后的精度和速度
- 为 H20 选择最佳量化方案

## 实验内容
1. **gptq_quant.py** — GPTQ INT4 量化
2. **awq_quant.py** — AWQ INT4 量化
3. **fp8_quant.py** — FP8 量化 (H20 原生支持)
4. **quality_benchmark.py** — 精度对比
5. **speed_benchmark.py** — 速度对比

## 运行方式
```bash
python gptq_quant.py --model meta-llama/Llama-2-7b-hf
python awq_quant.py --model meta-llama/Llama-2-7b-hf
python fp8_quant.py --model meta-llama/Llama-2-7b-hf
python quality_benchmark.py  # 对比精度
python speed_benchmark.py    # 对比速度
```
