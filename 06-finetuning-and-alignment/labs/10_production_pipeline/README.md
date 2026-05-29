# Lab 10: 生产流水线

## 目标
- 构建端到端训练流水线
- 实现模型版本管理
- 搭建 A/B 测试部署

## 运行方式
```bash
# 端到端训练流水线
python training_pipeline.py --config pipeline_config.yaml

# 模型版本管理
python model_versioning.py register --model_path ./output/lora --metadata '{"task": "cs"}'
python model_versioning.py list
python model_versioning.py promote --version v1.0 --status deployed

# A/B 测试部署
python ab_test_deploy.py --model_a ./model_v1 --model_b ./model_v2 --split 0.5
```
