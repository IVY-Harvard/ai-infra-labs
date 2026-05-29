# 模块 08：LLMOps 与 Agent 基础设施

## 模块概述

本模块面向有后端微服务经验但对 LLM 应用开发框架不熟悉的工程师，系统讲解从传统 MLOps 到 LLMOps 的演进，涵盖 RAG 架构、向量数据库、Agent 框架、Prompt 工程化、评估体系和部署流水线等核心主题。

**硬件环境**：8 张 NVIDIA H20 GPU（96GB HBM3 each, 总计 768GB 显存）

## 学习目标

完成本模块后，读者将能够：

1. 理解 LLMOps 全生命周期，掌握与传统 MLOps 的核心区别
2. 设计并实现生产级 RAG 系统（Advanced RAG + GraphRAG）
3. 选型并运维向量数据库，理解核心索引算法原理
4. 构建 Multi-Agent 系统，掌握主流编排框架
5. 实现 Prompt 工程化管理，包括版本控制和自动优化
6. 建立 LLM 评估体系，实现全链路可观测性
7. 搭建完整的实验追踪到灰度部署流水线

## 前置知识

- Python 后端开发（FastAPI/Flask）
- Docker & Kubernetes 基础
- 微服务架构设计经验
- 基本的机器学习概念

## 模块结构

```
08-llmops-and-agent-infra/
├── README.md                          # 本文件
├── theory/                            # 理论知识（7 篇）
│   ├── 01_llmops_overview.md          # LLMOps 全景
│   ├── 02_rag_architecture.md         # RAG 架构深度解析
│   ├── 03_vector_database.md          # 向量数据库
│   ├── 04_agent_architecture.md       # Agent 架构
│   ├── 05_prompt_engineering.md       # Prompt 工程化
│   ├── 06_evaluation_and_observability.md  # 评估与可观测
│   └── 07_experiment_and_deployment.md     # 实验到部署
├── labs/                              # 动手实验（10 个）
│   ├── 01_langchain_basics/           # LangChain 基础
│   ├── 02_rag_pipeline/              # 高级 RAG 流水线
│   ├── 03_vector_db_practice/        # 向量数据库实战
│   ├── 04_agent_frameworks/          # Agent 框架
│   ├── 05_multi_agent_orchestration/ # 多 Agent 编排
│   ├── 06_mcp_protocol/             # MCP 协议
│   ├── 07_prompt_engineering/        # Prompt 工程化
│   ├── 08_evaluation_pipeline/       # 评估流水线
│   ├── 09_experiment_tracking/       # 实验追踪
│   └── 10_deployment_pipeline/       # 部署流水线
└── project/                           # 企业级项目
    └── agent-serving-platform/        # Agent 服务平台
```

## 学习路径

### 第一周：基础概念与 RAG

| 天数 | 理论 | 实验 |
|------|------|------|
| Day 1 | 01_llmops_overview | Lab 01: LangChain 基础 |
| Day 2 | 02_rag_architecture | Lab 02: 高级 RAG 流水线 |
| Day 3 | 03_vector_database | Lab 03: 向量数据库实战 |

### 第二周：Agent 与工程化

| 天数 | 理论 | 实验 |
|------|------|------|
| Day 4 | 04_agent_architecture | Lab 04 + Lab 05: Agent 框架与多 Agent |
| Day 5 | 05_prompt_engineering | Lab 06 + Lab 07: MCP 与 Prompt 工程 |
| Day 6 | 06_evaluation_and_observability | Lab 08: 评估流水线 |
| Day 7 | 07_experiment_and_deployment | Lab 09 + Lab 10: 实验与部署 |

### 第三周：企业级项目

- 搭建 Agent 服务平台
- 集成 RAG、多 Agent 编排、在线评估
- 实现灰度发布与自动回滚

## 环境准备

```bash
# 创建 conda 环境
conda create -n llmops python=3.11 -y
conda activate llmops

# 核心依赖
pip install langchain langchain-community langchain-openai langgraph
pip install chromadb milvus-lite qdrant-client pgvector
pip install ragas deepeval trulens
pip install mlflow wandb
pip install fastapi uvicorn httpx
pip install dspy-ai autogen-agentchat crewai

# 可选：本地模型推理
pip install vllm transformers
```

## 关键技术栈

| 类别 | 技术 |
|------|------|
| LLM 框架 | LangChain, LlamaIndex, DSPy |
| Agent 编排 | LangGraph, AutoGen, CrewAI |
| 向量数据库 | Milvus, Qdrant, pgvector |
| 评估 | Ragas, DeepEval, TruLens |
| 可观测性 | LangSmith, Langfuse, Phoenix |
| 实验追踪 | MLflow, Weights & Biases |
| 部署 | vLLM, TGI, Docker, K8s |
| 协议标准 | MCP (Model Context Protocol) |

## 参考资源

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Anthropic MCP Specification](https://modelcontextprotocol.io/)
- [Ragas Documentation](https://docs.ragas.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [vLLM Documentation](https://docs.vllm.ai/)
