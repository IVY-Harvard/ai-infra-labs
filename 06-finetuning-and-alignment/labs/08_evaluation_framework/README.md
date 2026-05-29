# Lab 08: Evaluation Framework

## 目标
- 搭建统一的模型评估框架
- 运行多种基准测试
- 实现人工评估辅助工具

## 运行方式
```bash
# 运行评估
python eval_runner.py --model ./my_model --benchmarks mmlu ceval gsm8k

# 多基准测试
python benchmark_suite.py --model ./my_model --full

# 人工评估工具
python human_eval_tool.py --model_a ./sft_model --model_b ./dpo_model --num 20
```
