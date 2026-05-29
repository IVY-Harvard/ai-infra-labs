# Lab 04: LLaMA-Factory 使用指南

## 概述
LLaMA-Factory 是一个开箱即用的 LLM 微调框架，支持 100+ 模型和多种微调方法。
无需写代码，通过配置文件或 WebUI 即可完成训练。

## 安装
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[metrics]"
```

## 快速开始

### 命令行方式
```bash
# 使用 YAML 配置文件训练
llamafactory-cli train configs/lora_qwen2_7b.yaml

# 推理测试
llamafactory-cli chat configs/inference_qwen2_7b.yaml

# 导出模型
llamafactory-cli export configs/export_qwen2_7b.yaml
```

### WebUI 方式
```bash
# 启动 WebUI
llamafactory-cli webui

# 然后浏览器访问 http://localhost:7860
```

## 配置示例
参考 `config_examples/` 目录下的各种配置。

## 自定义数据集
1. 准备 JSON/JSONL 格式的数据
2. 在 `data/dataset_info.json` 中注册数据集
3. 在配置中引用数据集名称
