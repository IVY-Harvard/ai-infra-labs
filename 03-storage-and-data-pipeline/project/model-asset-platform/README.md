# 模型资产管理平台

## 项目概述

一个企业级的模型资产管理平台，集成了存储后端管理、模型注册与版本控制、分发调度和 Checkpoint 生命周期管理。

## 核心功能

1. **存储管理** — 统一抽象 S3/MinIO/本地文件系统，多级缓存和自动复制
2. **模型注册** — 模型元信息管理、版本控制、格式转换和校验
3. **分发调度** — P2P 分发、多级缓存加载、预热调度
4. **Checkpoint 管理** — 生命周期管理、GC 策略、跨集群同步

## 架构图

```
┌─────────────────────────────────────────────────┐
│                  API Layer (FastAPI)              │
│           /models  /storage  /distribute         │
├─────────────────────────────────────────────────┤
│          ┌──────────────────────────┐            │
│          │    Model Registry        │            │
│          │  版本管理 | 格式转换      │            │
│          └──────────────────────────┘            │
│          ┌──────────────────────────┐            │
│          │    Distribution Engine   │            │
│          │  P2P | Cache | Prewarm   │            │
│          └──────────────────────────┘            │
│          ┌──────────────────────────┐            │
│          │   Checkpoint Manager     │            │
│          │  Lifecycle | GC          │            │
│          └──────────────────────────┘            │
├─────────────────────────────────────────────────┤
│                Storage Layer                      │
│    ┌─────────┐  ┌──────────┐  ┌──────────┐     │
│    │ Local   │  │  S3/MinIO │  │ JuiceFS  │     │
│    │ NVMe    │  │           │  │          │     │
│    └─────────┘  └──────────┘  └──────────┘     │
└─────────────────────────────────────────────────┘
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动依赖服务（MinIO + Redis）
docker-compose -f deploy/docker-compose.yaml up -d

# 3. 启动平台
python -m src.api.server

# 4. 访问 API 文档
open http://localhost:8000/docs
```

## 项目结构

```
model-asset-platform/
├── README.md
├── Dockerfile
├── requirements.txt
├── deploy/
│   └── docker-compose.yaml
├── src/
│   ├── storage/
│   │   ├── backend.py          # 存储后端抽象
│   │   ├── cache_manager.py    # 多级缓存管理
│   │   └── replication.py      # 数据复制
│   ├── model/
│   │   ├── registry.py         # 模型注册中心
│   │   ├── versioning.py       # 版本管理
│   │   ├── format_converter.py # 格式转换
│   │   └── validator.py        # 模型校验
│   ├── distribution/
│   │   ├── distributor.py      # 分发引擎
│   │   ├── p2p_transfer.py     # P2P 传输
│   │   └── prewarmer.py        # 预热管理
│   ├── checkpoint/
│   │   ├── ckpt_manager.py     # Checkpoint 管理器
│   │   └── gc_policy.py        # GC 策略
│   └── api/
│       └── server.py           # FastAPI 服务
└── tests/
    └── test_storage.py         # 存储层测试
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /models/register | 注册新模型 |
| GET | /models/{name}/versions | 获取模型版本列表 |
| POST | /models/{name}/distribute | 触发模型分发 |
| GET | /storage/status | 存储状态 |
| POST | /checkpoints/gc | 触发 Checkpoint GC |

## 开发指南

```bash
# 运行测试
pytest tests/ -v

# 代码格式化
black src/ tests/

# 类型检查
mypy src/
```
