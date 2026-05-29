# Lab 01: LangChain 基础

## 目标

掌握 LangChain 核心组件：Chain、RAG、Memory，为后续高级实验打基础。

## 前置准备

```bash
pip install langchain langchain-openai langchain-community chromadb
```

## 实验内容

1. `rag_basic.py` — 基础 RAG 流水线（文档加载 → 分块 → Embedding → 检索 → 生成）
2. `chain_demo.py` — Chain 编排模式（顺序链/路由链/转换链）
3. `memory_demo.py` — 对话记忆机制（Buffer/Summary/Vector Memory）

## 验证标准

- RAG 能正确回答基于文档内容的问题
- Chain 能正确路由到不同处理逻辑
- Memory 能在多轮对话中保持上下文

## 预计用时

2-3 小时
