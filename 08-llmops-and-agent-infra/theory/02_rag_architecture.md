# RAG 架构深度解析

## 1. RAG 概述

### 1.1 为什么需要 RAG

LLM 的核心局限：
- **知识截止**：训练数据有截止日期，无法获取最新信息
- **幻觉问题**：在知识边界外倾向于编造看似合理的答案
- **领域知识缺乏**：企业私有数据未在训练集中
- **无法溯源**：回答缺乏可验证的引用来源

RAG (Retrieval-Augmented Generation) 通过"先检索、后生成"的方式解决这些问题：

```
用户查询 → 检索相关文档 → 将文档作为上下文 → LLM 基于上下文生成回答
```

### 1.2 RAG vs 微调 vs 长上下文

| 方法 | 适用场景 | 优势 | 劣势 |
|------|----------|------|------|
| RAG | 知识密集型问答 | 实时更新、可溯源 | 检索质量依赖 |
| 微调 | 风格/格式适配 | 内化知识、低延迟 | 更新成本高 |
| 长上下文 | 单文档深度理解 | 实现简单 | 成本高、"中间丢失" |
| RAG + 微调 | 生产最佳实践 | 兼顾两者 | 工程复杂度高 |

## 2. RAG 架构演进

### 2.1 Naive RAG

最基础的 RAG 实现：

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  用户查询 │ →  │ Embedding │ →  │ 向量检索  │ →  │ LLM 生成 │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                                      ↑
                              ┌──────────────┐
                              │  向量数据库   │
                              │  (预处理文档) │
                              └──────────────┘
```

**Indexing 阶段**：
```
原始文档 → 文档解析 → 文本分块(Chunking) → Embedding → 存入向量数据库
```

**Query 阶段**：
```
用户问题 → Embedding → 向量相似度检索 → Top-K 文档 → 拼接 Prompt → LLM 回答
```

**Naive RAG 的问题**：
1. 分块粗糙导致上下文不完整
2. 查询与文档语义空间不匹配
3. 检索结果缺乏排序优化
4. 无法处理需要多步推理的复杂问题

### 2.2 Advanced RAG

在 Naive RAG 基础上的系统性优化：

```
┌──────────────────────────────────────────────────────────────┐
│                    Advanced RAG Pipeline                       │
│                                                                │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐    │
│  │ Pre-Retrieval│     │  Retrieval  │     │Post-Retrieval│    │
│  │             │     │             │     │             │    │
│  │ • Query     │  →  │ • Hybrid    │  →  │ • Reranker  │    │
│  │   Rewriting │     │   Search    │     │ • Filter    │    │
│  │ • HyDE      │     │ • Multi-    │     │ • Compress  │    │
│  │ • Query     │     │   Index     │     │ • Dedup     │    │
│  │   Expansion │     │ • Recursive │     │             │    │
│  └─────────────┘     └─────────────┘     └──────┬──────┘    │
│                                                   │          │
│                                            ┌──────▼──────┐   │
│                                            │  Generation │   │
│                                            │  + Citation │   │
│                                            └─────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

#### Pre-Retrieval 优化

**Query Rewriting（查询改写）**：
```python
# 原始查询可能模糊或不完整
original_query = "H20 跑什么模型好"

# 改写后
rewritten_queries = [
    "NVIDIA H20 GPU 96GB 适合部署哪些大语言模型",
    "H20 GPU 推理性能对比：70B vs 7B 模型",
    "H20 显存适配的开源 LLM 模型推荐"
]
```

**HyDE（Hypothetical Document Embedding）**：
```
查询 → LLM 生成假设性答案 → 对假设答案做 Embedding → 用假设答案的 Embedding 检索
```

核心思想：假设答案与真实文档在 Embedding 空间更接近，而用户查询往往是疑问句式，与文档的陈述句式存在语义间隙。

#### Retrieval 优化

**Hybrid Search（混合搜索）**：
```python
# 结合稀疏检索（BM25）和稠密检索（向量）
def hybrid_search(query, alpha=0.7):
    # 稠密检索：语义理解强
    dense_results = vector_db.search(embed(query), top_k=20)
    
    # 稀疏检索：精确匹配强（术语、编号、人名）
    sparse_results = bm25_index.search(query, top_k=20)
    
    # 加权融合 (Reciprocal Rank Fusion)
    return rrf_merge(dense_results, sparse_results, alpha=alpha)
```

**Multi-Index 策略**：
```
文档类型  →  专用索引
  ├── 技术文档  →  按章节分块，保留层级结构
  ├── FAQ      →  问答对索引
  ├── 代码     →  函数/类级别索引
  └── 表格     →  结构化索引
```

#### Post-Retrieval 优化

**Reranker（重排序）**：
```python
# 第一阶段：召回 (高效但粗糙)
candidates = vector_db.search(query, top_k=50)

# 第二阶段：精排 (精确但昂贵)
# 使用 Cross-Encoder 对 (query, doc) 对做精细打分
reranked = reranker.rank(query, candidates)[:5]
```

常用 Reranker 模型：
- `bge-reranker-v2-m3`（开源，可本地部署在 H20 上）
- `Cohere Rerank`（API 服务）
- `Jina Reranker`（API 服务）

### 2.3 Modular RAG

将 RAG 各组件模块化，支持灵活组合：

```
┌─────────────────────────────────────────────────────┐
│                  Modular RAG                          │
│                                                       │
│  模块池：                                              │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │
│  │ Loader │ │Chunker │ │Embedder│ │Indexer │       │
│  └────────┘ └────────┘ └────────┘ └────────┘       │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │
│  │Rewriter│ │Retriever│ │Reranker│ │ Filter │       │
│  └────────┘ └────────┘ └────────┘ └────────┘       │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │
│  │Compress│ │Augment │ │Generate│ │  Judge  │       │
│  └────────┘ └────────┘ └────────┘ └────────┘       │
│                                                       │
│  编排模式：                                            │
│  • 线性管道：A → B → C → D                            │
│  • 条件分支：if complex → path_A else → path_B       │
│  • 循环迭代：Generate → Judge → (retry if bad)       │
│  • 自适应：根据查询类型动态组装管道                     │
└─────────────────────────────────────────────────────┘
```

## 3. Embedding 选型

### 3.1 Embedding 模型对比

| 模型 | 维度 | 中文支持 | MTEB 排名 | 部署方式 |
|------|------|----------|-----------|----------|
| text-embedding-3-large | 3072 | 良好 | Top 10 | API |
| bge-m3 | 1024 | 优秀 | Top 5 | 本地 (H20) |
| jina-embeddings-v3 | 1024 | 优秀 | Top 5 | API/本地 |
| e5-mistral-7b | 4096 | 良好 | Top 3 | 本地 (H20) |
| multilingual-e5-large | 1024 | 优秀 | Top 15 | 本地 |

### 3.2 H20 上的 Embedding 部署

```python
# 使用 sentence-transformers 在 H20 上部署 bge-m3
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3", device="cuda:0")
# 单张 H20 可同时服务 embedding 和 reranker
# 96GB 显存足够部署多个 embedding 模型
```

## 4. Chunking 策略

### 4.1 分块策略对比

```
策略 1：Fixed-Size Chunking（固定大小）
  ├── 简单高效，适合均匀文本
  ├── 参数：chunk_size=512, overlap=50
  └── 问题：可能截断语义单元

策略 2：Recursive Character Splitting（递归字符分割）
  ├── LangChain 默认策略
  ├── 按 ["\n\n", "\n", " ", ""] 优先级分割
  └── 尽量在自然边界处分块

策略 3：Semantic Chunking（语义分块）
  ├── 基于 Embedding 相似度判断分块边界
  ├── 相邻句子相似度突降处 = 分块边界
  └── 质量最好但计算成本高

策略 4：Document-Structure Chunking（结构化分块）
  ├── 利用文档结构（标题/段落/列表）
  ├── 适合有明确结构的技术文档
  └── 保留层级关系作为元数据

策略 5：Agentic Chunking（代理分块）
  ├── 用 LLM 判断每段文本的主题归属
  ├── 相同主题的段落合并为一个 chunk
  └── 质量最高但成本最贵
```

### 4.2 分块参数调优

```python
# 经验法则
chunk_configs = {
    "technical_docs": {"size": 1000, "overlap": 200},  # 技术文档需要更多上下文
    "faq": {"size": 300, "overlap": 0},                  # FAQ 天然成对
    "legal": {"size": 1500, "overlap": 300},              # 法律文本需要完整条款
    "code": {"size": 500, "overlap": 100},                # 代码按函数/类分块
    "chat_logs": {"size": 200, "overlap": 50},            # 对话按轮次
}
```

## 5. GraphRAG

### 5.1 传统 RAG 的局限

传统向量检索在以下场景表现不佳：
- **多跳推理**："谁是张三的老板的老板？"
- **全局问题**："公司今年的主要业务方向是什么？"
- **关系查询**："A 项目和 B 项目有什么关联？"

### 5.2 GraphRAG 架构

```
┌─────────────────────────────────────────────────────┐
│                    GraphRAG                            │
│                                                       │
│  索引阶段：                                            │
│  文档 → LLM 实体抽取 → 构建知识图谱 → 社区检测       │
│                 ↓              ↓            ↓         │
│           实体节点        关系边       社区摘要        │
│                                                       │
│  查询阶段：                                            │
│  ┌─────────┐                                         │
│  │ Local   │ → 实体匹配 → 子图检索 → 子图+LLM回答   │
│  │ Search  │   适合具体问题                           │
│  └─────────┘                                         │
│  ┌─────────┐                                         │
│  │ Global  │ → 社区摘要 → Map-Reduce → 综合回答      │
│  │ Search  │   适合全局问题                           │
│  └─────────┘                                         │
└─────────────────────────────────────────────────────┘
```

### 5.3 GraphRAG 实体抽取示例

```python
extraction_prompt = """
从以下文本中抽取实体和关系：

文本：{text}

请以 JSON 格式返回：
{
  "entities": [
    {"name": "...", "type": "Person/Org/Tech/...", "description": "..."}
  ],
  "relationships": [
    {"source": "...", "target": "...", "relation": "...", "description": "..."}
  ]
}
"""

# 实际生产中使用 microsoft/graphrag 或 nano-graphrag
# 注意：GraphRAG 索引阶段需要大量 LLM 调用，成本较高
```

## 6. 生产级 RAG 架构

### 6.1 完整架构图

```
                          ┌──────────────┐
                          │   用户查询    │
                          └──────┬───────┘
                                 │
                          ┌──────▼───────┐
                          │ Query Router │ ── 判断是否需要检索
                          └──────┬───────┘
                     ┌───────────┼───────────┐
                     ▼           ▼           ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │直接回答  │ │ RAG 路径 │ │ Agent路径│
              └──────────┘ └────┬─────┘ └──────────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
             ┌──────────┐ ┌──────────┐ ┌──────────┐
             │HyDE 改写 │ │多查询展开│ │Step-back │
             └────┬─────┘ └────┬─────┘ └────┬─────┘
                  └────────────┼────────────┘
                               ▼
                    ┌──────────────────────┐
                    │   Hybrid Retrieval   │
                    │  Dense + Sparse + KG │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │  Reranker + Filter   │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │  Context Compression │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │  LLM Generation      │
                    │  + Citation          │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │  Response Validator  │
                    │  (幻觉检测+质量评估) │
                    └──────────────────────┘
```

### 6.2 关键设计原则

1. **查询路由**：不是所有查询都需要 RAG，简单问候/闲聊直接回答
2. **多策略检索**：不同查询类型使用不同的检索策略
3. **质量门禁**：生成结果需通过幻觉检测才能返回给用户
4. **可观测性**：每个阶段都记录延迟、结果质量、Token 消耗
5. **优雅降级**：检索失败时 fallback 到纯 LLM 回答，并标注"未检索到相关文档"

## 7. 小结

RAG 架构的演进路线：

```
Naive RAG → Advanced RAG → Modular RAG → Agentic RAG
  简单拼接     系统优化       灵活编排      智能自适应
```

选择哪个阶段的架构取决于：
- 数据复杂度和规模
- 查询类型的多样性
- 对质量的要求程度
- 可接受的延迟和成本

对于大多数企业应用，Advanced RAG（HyDE + Hybrid Search + Reranker）是性价比最高的起点。
