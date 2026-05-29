# AI Observability Platform — GPU 推理全栈可观测平台

## 项目概述

整合 Module 09 所有 Lab 的技术, 构建一个完整的 GPU 推理可观测平台。

## 架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AI Observability Platform                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────── Data Collection ────────────┐                            │
│  │  gpu_collector.py        GPU/DCGM 指标   │                            │
│  │  inference_collector.py  vLLM 推理指标    │                            │
│  │  training_collector.py   训练任务指标     │                            │
│  │  network_collector.py    NVLink/网络指标  │                            │
│  └─────────────────────┬───────────────────┘                            │
│                        │                                                │
│  ┌─────────── Analytics Engine ────────────┐                            │
│  │  capacity_planner.py    容量规划与预测    │                            │
│  │  cost_analyzer.py       成本分析与归因    │                            │
│  │  performance_advisor.py 性能优化建议      │                            │
│  │  anomaly_engine.py      异常检测引擎      │                            │
│  └─────────────────────┬───────────────────┘                            │
│                        │                                                │
│  ┌──────────── Alerting System ────────────┐                            │
│  │  alert_manager.py      告警管理与路由     │                            │
│  │  escalation.py         告警升级策略       │                            │
│  │  notifier.py           多渠道通知         │                            │
│  └─────────────────────┬───────────────────┘                            │
│                        │                                                │
│  ┌──────────── API & Reporting ────────────┐                            │
│  │  api.py                REST API 服务      │                            │
│  │  daily_report.py       日报生成           │                            │
│  │  cost_report.py        成本报告           │                            │
│  └─────────────────────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## 快速启动

```bash
# 构建
docker-compose -f deploy/docker-compose.yaml build

# 启动
docker-compose -f deploy/docker-compose.yaml up -d

# 查看状态
curl http://localhost:8080/api/v1/health
```

## 项目结构

```
ai-observability-platform/
├── README.md
├── requirements.txt
├── Dockerfile
├── src/
│   ├── collectors/          # 数据采集层
│   │   ├── gpu_collector.py
│   │   ├── inference_collector.py
│   │   ├── training_collector.py
│   │   └── network_collector.py
│   ├── analytics/           # 分析引擎
│   │   ├── capacity_planner.py
│   │   ├── cost_analyzer.py
│   │   ├── performance_advisor.py
│   │   └── anomaly_engine.py
│   ├── alerting/            # 告警系统
│   │   ├── alert_manager.py
│   │   ├── escalation.py
│   │   └── notifier.py
│   ├── dashboard/           # API 服务
│   │   └── api.py
│   └── reporting/           # 报告生成
│       ├── daily_report.py
│       └── cost_report.py
├── tests/
│   └── test_collectors.py
└── deploy/
    └── docker-compose.yaml
```
