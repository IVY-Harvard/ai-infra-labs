# Agent 服务平台

## 概述

企业级 Agent 服务平台，集成多 Agent 编排、RAG 知识问答、在线评估和灰度发布。

## 架构

```
                    ┌──────────────────┐
                    │    API Gateway   │
                    │  (FastAPI)       │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │  Router  │  │Rate Limit│  │Load Bal. │
        └────┬─────┘  └──────────┘  └──────────┘
             │
     ┌───────┼───────┐
     ▼       ▼       ▼
  ┌──────┐┌──────┐┌──────┐
  │Agent ││Agent ││Agent │  ← Agent Registry
  │  A   ││  B   ││  C   │
  └──┬───┘└──┬───┘└──┬───┘
     │       │       │
     ▼       ▼       ▼
  ┌─────────────────────┐
  │   Shared Services   │
  │ ┌───────┐ ┌───────┐ │
  │ │Memory │ │ Tools │ │
  │ └───────┘ └───────┘ │
  │ ┌───────┐ ┌───────┐ │
  │ │ Eval  │ │  MCP  │ │
  │ └───────┘ └───────┘ │
  └─────────────────────┘
```

## 目录结构

```
src/
├── api/server.py          # FastAPI 入口
├── gateway/
│   ├── router.py          # 请求路由
│   ├── load_balancer.py   # 负载均衡
│   └── rate_limiter.py    # 限流
├── agent/
│   ├── agent_registry.py  # Agent 注册中心
│   ├── orchestrator.py    # 多 Agent 编排器
│   └── executor.py        # Agent 执行器
├── memory/
│   ├── short_term.py      # 短期记忆
│   ├── long_term.py       # 长期记忆（向量）
│   └── memory_manager.py  # 记忆管理
├── tools/
│   ├── tool_registry.py   # 工具注册
│   └── mcp_adapter.py     # MCP 适配器
└── evaluation/
    ├── online_evaluator.py  # 在线评估
    └── quality_gate.py      # 质量门禁
```

## 快速启动

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
uvicorn src.api.server:app --host 0.0.0.0 --port 8080

# Docker 方式
docker-compose -f deploy/docker-compose.yaml up
```

## API

```
POST /v1/chat          — 单轮对话
POST /v1/chat/stream   — 流式对话
POST /v1/agents        — 注册 Agent
GET  /v1/agents        — 列出 Agent
POST /v1/evaluate      — 触发评估
GET  /v1/health        — 健康检查
GET  /v1/metrics       — Prometheus 指标
```
