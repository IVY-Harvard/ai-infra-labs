"""
Lab 02: Hybrid Search - 稠密检索 + 稀疏检索混合
结合向量搜索（语义理解）和 BM25（精确匹配）
"""
import os
import numpy as np
from typing import Optional
from dataclasses import dataclass

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# BM25 稀疏检索
from rank_bm25 import BM25Okapi
import jieba  # 中文分词


# =============================================================================
# 配置
# =============================================================================

EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:8001/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")


@dataclass
class SearchResult:
    document: Document
    score: float
    source: str  # "dense", "sparse", or "hybrid"


# =============================================================================
# BM25 稀疏检索器
# =============================================================================

class BM25Retriever:
    """BM25 稀疏检索器 - 基于词频统计的精确匹配"""

    def __init__(self, documents: list[Document]):
        self.documents = documents
        # 中文分词
        self.tokenized_corpus = [
            list(jieba.cut(doc.page_content)) for doc in documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """BM25 检索"""
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)

        # 获取 top_k
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(SearchResult(
                    document=self.documents[idx],
                    score=float(scores[idx]),
                    source="sparse",
                ))
        return results


# =============================================================================
# 稠密检索器
# =============================================================================

class DenseRetriever:
    """向量检索器 - 基于 Embedding 的语义检索"""

    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """向量相似度检索"""
        results = self.vectorstore.similarity_search_with_score(
            query, k=top_k
        )

        return [
            SearchResult(
                document=doc,
                score=1.0 / (1.0 + score),  # 将距离转为相似度
                source="dense",
            )
            for doc, score in results
        ]


# =============================================================================
# 混合检索器
# =============================================================================

class HybridRetriever:
    """
    混合检索器 - 结合稠密和稀疏检索
    使用 Reciprocal Rank Fusion (RRF) 融合排序
    """

    def __init__(self, dense_retriever: DenseRetriever,
                 sparse_retriever: BM25Retriever,
                 dense_weight: float = 0.7):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = 1.0 - dense_weight

    def search(self, query: str, top_k: int = 5,
               fusion_method: str = "rrf") -> list[SearchResult]:
        """
        混合检索
        fusion_method: "rrf" (Reciprocal Rank Fusion) 或 "weighted"
        """
        # 获取两路检索结果
        dense_results = self.dense.search(query, top_k=top_k * 2)
        sparse_results = self.sparse.search(query, top_k=top_k * 2)

        if fusion_method == "rrf":
            return self._rrf_fusion(dense_results, sparse_results, top_k)
        else:
            return self._weighted_fusion(dense_results, sparse_results, top_k)

    def _rrf_fusion(self, dense_results: list, sparse_results: list,
                    top_k: int, k: int = 60) -> list[SearchResult]:
        """
        Reciprocal Rank Fusion
        公式: RRF_score = sum(1 / (k + rank_i))
        k 是平滑常数（默认 60）
        """
        doc_scores = {}

        # 稠密检索的 RRF 分数
        for rank, result in enumerate(dense_results):
            doc_id = result.document.page_content[:50]  # 简化的文档 ID
            rrf_score = self.dense_weight / (k + rank + 1)
            if doc_id in doc_scores:
                doc_scores[doc_id]["score"] += rrf_score
            else:
                doc_scores[doc_id] = {
                    "document": result.document,
                    "score": rrf_score,
                }

        # 稀疏检索的 RRF 分数
        for rank, result in enumerate(sparse_results):
            doc_id = result.document.page_content[:50]
            rrf_score = self.sparse_weight / (k + rank + 1)
            if doc_id in doc_scores:
                doc_scores[doc_id]["score"] += rrf_score
            else:
                doc_scores[doc_id] = {
                    "document": result.document,
                    "score": rrf_score,
                }

        # 排序并返回
        sorted_docs = sorted(
            doc_scores.values(), key=lambda x: x["score"], reverse=True
        )

        return [
            SearchResult(
                document=d["document"], score=d["score"], source="hybrid"
            )
            for d in sorted_docs[:top_k]
        ]

    def _weighted_fusion(self, dense_results, sparse_results,
                         top_k) -> list[SearchResult]:
        """加权融合：归一化分数后加权求和"""
        # 归一化稠密分数
        if dense_results:
            max_dense = max(r.score for r in dense_results)
            for r in dense_results:
                r.score = r.score / max_dense if max_dense > 0 else 0

        # 归一化稀疏分数
        if sparse_results:
            max_sparse = max(r.score for r in sparse_results)
            for r in sparse_results:
                r.score = r.score / max_sparse if max_sparse > 0 else 0

        # 合并并加权
        doc_scores = {}
        for r in dense_results:
            doc_id = r.document.page_content[:50]
            doc_scores[doc_id] = {
                "document": r.document,
                "score": r.score * self.dense_weight,
            }

        for r in sparse_results:
            doc_id = r.document.page_content[:50]
            if doc_id in doc_scores:
                doc_scores[doc_id]["score"] += r.score * self.sparse_weight
            else:
                doc_scores[doc_id] = {
                    "document": r.document,
                    "score": r.score * self.sparse_weight,
                }

        sorted_docs = sorted(
            doc_scores.values(), key=lambda x: x["score"], reverse=True
        )

        return [
            SearchResult(
                document=d["document"], score=d["score"], source="hybrid"
            )
            for d in sorted_docs[:top_k]
        ]


# =============================================================================
# 对比实验
# =============================================================================

def run_hybrid_search_demo():
    """混合检索对比实验"""
    # 准备测试文档
    docs = [
        Document(page_content="HNSW (Hierarchical Navigable Small World) 是一种基于图的近似最近邻索引算法，"
                              "由 Yuri Malkov 等人在 2016 年提出。"),
        Document(page_content="IVF (Inverted File Index) 通过 K-Means 聚类将向量分组，"
                              "查询时只搜索最近的几个簇，大幅减少计算量。"),
        Document(page_content="Milvus 2.0 采用存算分离架构，支持 HNSW、IVF_FLAT、IVF_PQ 等多种索引。"
                              "最新版本支持 GPU 加速的 CAGRA 索引。"),
        Document(page_content="pgvector 是 PostgreSQL 的向量搜索扩展，支持 HNSW 和 IVF 索引。"
                              "适合已有 PG 基础设施的中小规模场景。"),
        Document(page_content="Qdrant 使用 Rust 语言实现，以高性能和丰富的过滤能力著称。"
                              "支持 scalar quantization 和 product quantization。"),
        Document(page_content="Product Quantization (PQ) 通过将向量分段量化来压缩存储。"
                              "可将 1024 维 float32 向量从 4KB 压缩到 128 字节。"),
        Document(page_content="向量数据库的选型需要考虑：数据规模、查询延迟、过滤需求、运维复杂度等因素。"),
        Document(page_content="在 RAG 系统中，向量检索的 recall@10 应该达到 90% 以上才能保证生成质量。"),
    ]

    # 构建检索器
    embeddings = OpenAIEmbeddings(
        base_url=EMBEDDING_BASE_URL, model=EMBEDDING_MODEL, api_key="not-needed"
    )
    vectorstore = Chroma.from_documents(docs, embeddings)

    dense_retriever = DenseRetriever(vectorstore)
    sparse_retriever = BM25Retriever(docs)
    hybrid_retriever = HybridRetriever(
        dense_retriever, sparse_retriever, dense_weight=0.6
    )

    # 测试查询
    test_queries = [
        "HNSW 算法是什么？",               # 精确术语 → BM25 有优势
        "哪个向量数据库适合小团队？",       # 语义理解 → Dense 有优势
        "怎么压缩向量节省内存？",           # 混合 → Hybrid 最佳
        "Milvus CAGRA GPU 索引",            # 多关键词 → BM25 有优势
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"查询: {query}")
        print(f"{'='*60}")

        # 稠密检索
        dense_results = dense_retriever.search(query, top_k=3)
        print(f"\n  [Dense] Top-3:")
        for r in dense_results:
            print(f"    score={r.score:.4f}: {r.document.page_content[:60]}...")

        # 稀疏检索
        sparse_results = sparse_retriever.search(query, top_k=3)
        print(f"\n  [Sparse/BM25] Top-3:")
        for r in sparse_results:
            print(f"    score={r.score:.4f}: {r.document.page_content[:60]}...")

        # 混合检索
        hybrid_results = hybrid_retriever.search(query, top_k=3)
        print(f"\n  [Hybrid/RRF] Top-3:")
        for r in hybrid_results:
            print(f"    score={r.score:.6f}: {r.document.page_content[:60]}...")


if __name__ == "__main__":
    run_hybrid_search_demo()
