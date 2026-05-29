# Lab 02: 高级 RAG 流水线

## 目标

实现 Advanced RAG 的核心优化技术：HyDE、Reranker、Hybrid Search、多种 Chunking 策略对比。

## 前置准备

```bash
pip install langchain langchain-openai chromadb rank_bm25 sentence-transformers
pip install FlagEmbedding  # for bge-reranker
```

## 实验内容

1. `advanced_rag.py` — HyDE + Reranker 集成的完整 RAG
2. `hybrid_search.py` — 稠密检索 + 稀疏检索(BM25) 混合搜索
3. `chunk_strategy_comparison.py` — 不同分块策略的效果对比

## 验证标准

- HyDE 在模糊查询上的改善幅度 > 10%
- Hybrid Search 在专业术语查询上优于纯向量搜索
- 能说出不同 Chunking 策略的适用场景
