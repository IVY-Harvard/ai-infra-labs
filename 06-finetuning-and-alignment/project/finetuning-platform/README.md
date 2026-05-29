# 微调平台 (Finetuning Platform)

## 概述

一个简化版的企业级 LLM 微调平台，提供统一的训练、评估、部署流水线。

## 架构

```
┌──────────────────────────────────────────────┐
│                FastAPI Server                │
│              /api/train  /api/eval           │
├──────────────────────────────────────────────┤
│                                              │
│  ┌──────────┐  ┌───────────┐  ┌───────────┐ │
│  │ Trainer  │  │ Evaluator │  │  Serving  │ │
│  │ LoRA/DPO │  │ Benchmark │  │  Export   │ │
│  │ Distrib. │  │  Report   │  │  Deploy   │ │
│  └────┬─────┘  └─────┬─────┘  └─────┬─────┘ │
│       │              │              │        │
│  ┌────┴──────────────┴──────────────┴─────┐  │
│  │           Data Layer                   │  │
│  │  Loader / Processor / Quality Filter   │  │
│  └────────────────────────────────────────┘  │
│                                              │
└──────────────────────────────────────────────┘
```

## 目录结构

```
finetuning-platform/
├── src/
│   ├── trainer/
│   │   ├── base_trainer.py        # 训练器基类
│   │   ├── lora_trainer.py        # LoRA 训练
│   │   ├── dpo_trainer.py         # DPO 训练
│   │   └── distributed_trainer.py # 分布式封装
│   ├── data/
│   │   ├── data_loader.py         # 数据加载
│   │   ├── data_processor.py      # 数据预处理
│   │   └── quality_filter.py      # 质量过滤
│   ├── evaluation/
│   │   ├── evaluator.py           # 统一评估器
│   │   ├── benchmarks.py          # 基准测试
│   │   └── report_generator.py    # 评估报告
│   ├── serving/
│   │   ├── model_exporter.py      # 模型导出
│   │   └── deploy_manager.py      # 部署管理
│   └── api/
│       └── server.py              # FastAPI 服务
├── configs/
│   ├── lora_qwen_7b.yaml         # Qwen 7B LoRA 配置
│   └── dpo_llama_8b.yaml         # Llama 8B DPO 配置
├── tests/
│   └── test_trainer.py
├── Dockerfile
└── requirements.txt
```

## 快速开始

```bash
# 安装
pip install -r requirements.txt

# 启动 API 服务
uvicorn src.api.server:app --host 0.0.0.0 --port 8080

# 提交训练任务
curl -X POST http://localhost:8080/api/train \
  -H "Content-Type: application/json" \
  -d '{"config_path": "configs/lora_qwen_7b.yaml"}'

# 查看任务状态
curl http://localhost:8080/api/train/{job_id}/status
```

## 硬件要求

- 最少: 1 × H20 (96GB) — 单卡 LoRA/QLoRA
- 推荐: 8 × H20 (768GB) — 多卡大模型训练
