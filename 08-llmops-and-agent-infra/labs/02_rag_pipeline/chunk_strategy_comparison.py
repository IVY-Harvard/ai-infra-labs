"""
Lab 02: Chunking 策略对比实验
对比不同分块策略对 RAG 质量的影响
"""
import os
import time
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate


# =============================================================================
# 配置
# =============================================================================

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-72b-instruct")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:8001/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")


# =============================================================================
# Chunking 策略
# =============================================================================

class FixedSizeChunker:
    """固定大小分块"""
    name = "Fixed-Size"

    def __init__(self, chunk_size: int = 500, overlap: int = 0):
        self.splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separator="",
        )

    def split(self, documents: list[Document]) -> list[Document]:
        return self.splitter.split_documents(documents)


class RecursiveChunker:
    """递归字符分块（LangChain 默认）"""
    name = "Recursive"

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

    def split(self, documents: list[Document]) -> list[Document]:
        return self.splitter.split_documents(documents)


class SemanticChunker:
    """
    语义分块：基于 Embedding 相似度判断分块边界
    当相邻句子的语义相似度突降时，在该处分块
    """
    name = "Semantic"

    def __init__(self, threshold: float = 0.5, min_chunk_size: int = 100):
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size
        self.embeddings = OpenAIEmbeddings(
            base_url=EMBEDDING_BASE_URL,
            model=EMBEDDING_MODEL,
            api_key="not-needed",
        )

    def split(self, documents: list[Document]) -> list[Document]:
        all_chunks = []
        for doc in documents:
            chunks = self._split_single(doc)
            all_chunks.extend(chunks)
        return all_chunks

    def _split_single(self, document: Document) -> list[Document]:
        """对单个文档进行语义分块"""
        import numpy as np

        # 按句子分割
        text = document.page_content
        sentences = [s.strip() for s in text.replace("\n", "。").split("。") if s.strip()]

        if len(sentences) <= 1:
            return [document]

        # 计算每个句子的 embedding
        sentence_embeddings = self.embeddings.embed_documents(sentences)

        # 计算相邻句子的余弦相似度
        similarities = []
        for i in range(len(sentence_embeddings) - 1):
            a = np.array(sentence_embeddings[i])
            b = np.array(sentence_embeddings[i + 1])
            sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            similarities.append(sim)

        # 在相似度低于阈值处分块
        chunks = []
        current_chunk = [sentences[0]]

        for i, sim in enumerate(similarities):
            if sim < self.threshold and len("。".join(current_chunk)) >= self.min_chunk_size:
                chunks.append(Document(
                    page_content="。".join(current_chunk),
                    metadata={**document.metadata, "chunk_method": "semantic"},
                ))
                current_chunk = []
            current_chunk.append(sentences[i + 1])

        if current_chunk:
            chunks.append(Document(
                page_content="。".join(current_chunk),
                metadata={**document.metadata, "chunk_method": "semantic"},
            ))

        return chunks


class StructureChunker:
    """结构化分块：利用 Markdown 标题结构"""
    name = "Structure"

    def split(self, documents: list[Document]) -> list[Document]:
        from langchain.text_splitter import MarkdownHeaderTextSplitter

        headers_to_split = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split
        )

        all_chunks = []
        for doc in documents:
            md_chunks = splitter.split_text(doc.page_content)
            for chunk in md_chunks:
                chunk.metadata.update(doc.metadata)
            all_chunks.extend(md_chunks)

        return all_chunks


# =============================================================================
# 评估框架
# =============================================================================

class ChunkingEvaluator:
    """分块策略评估器"""

    def __init__(self):
        self.llm = ChatOpenAI(
            base_url=LLM_BASE_URL, model=LLM_MODEL,
            api_key="not-needed", temperature=0,
        )
        self.embeddings = OpenAIEmbeddings(
            base_url=EMBEDDING_BASE_URL, model=EMBEDDING_MODEL,
            api_key="not-needed",
        )

    def evaluate_chunker(self, chunker, documents, eval_questions):
        """评估一个分块策略"""
        print(f"\n{'='*60}")
        print(f"评估分块策略: {chunker.name}")
        print(f"{'='*60}")

        # Step 1: 分块
        start_time = time.time()
        chunks = chunker.split(documents)
        chunk_time = time.time() - start_time

        # 统计信息
        chunk_sizes = [len(c.page_content) for c in chunks]
        print(f"  分块数: {len(chunks)}")
        print(f"  平均长度: {sum(chunk_sizes)/len(chunk_sizes):.0f}")
        print(f"  最短: {min(chunk_sizes)}, 最长: {max(chunk_sizes)}")
        print(f"  分块耗时: {chunk_time:.2f}s")

        # Step 2: 构建向量索引
        db_name = f"./eval_db_{chunker.name.lower()}"
        vectorstore = Chroma.from_documents(
            chunks, self.embeddings, persist_directory=db_name
        )

        # Step 3: 检索评估
        results = []
        for qa in eval_questions:
            query = qa["question"]
            expected = qa["expected_keywords"]

            # 检索
            docs = vectorstore.similarity_search(query, k=3)
            retrieved_text = " ".join([d.page_content for d in docs])

            # 检查关键词覆盖
            keywords_found = sum(
                1 for kw in expected if kw in retrieved_text
            )
            recall = keywords_found / len(expected)

            results.append({
                "question": query,
                "recall": recall,
                "keywords_found": keywords_found,
                "total_keywords": len(expected),
            })

            print(f"\n  Q: {query}")
            print(f"  关键词覆盖: {keywords_found}/{len(expected)} = {recall:.2f}")

        avg_recall = sum(r["recall"] for r in results) / len(results)
        print(f"\n  平均关键词召回率: {avg_recall:.2f}")

        return {
            "chunker": chunker.name,
            "num_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "chunk_time": chunk_time,
            "avg_recall": avg_recall,
            "detail_results": results,
        }


# =============================================================================
# 主程序
# =============================================================================

def main():
    # 准备测试文档（Markdown 格式的技术文档）
    sample_doc = Document(page_content="""
# 向量数据库选型指南

## 1. Milvus

### 1.1 架构特点
Milvus 采用存算分离的云原生架构。核心组件包括 Proxy、Coordinator、Worker 和 Storage。
支持水平扩展，适合大规模部署。最新版本引入了 GPU 加速的 CAGRA 索引。

### 1.2 性能数据
在 1 亿条 768 维向量上，HNSW 索引的 P99 延迟为 5ms，QPS 可达 10000。
IVF_PQ 索引内存占用仅为 HNSW 的 1/10，但 recall 下降约 5%。

## 2. Qdrant

### 2.1 架构特点
Qdrant 使用 Rust 实现，单机性能优异。支持丰富的 payload 过滤。
内置 scalar quantization 和 binary quantization 压缩。

### 2.2 适用场景
适合中等规模（千万级）、需要复杂过滤的场景。
部署简单，Docker 一键启动，运维成本低。

## 3. pgvector

### 3.1 架构特点
pgvector 是 PostgreSQL 的扩展，复用 PG 的事务、权限、备份等基础设施。
支持 HNSW 和 IVFFlat 两种索引。

### 3.2 适用场景
适合小规模（百万级）、已有 PG 基础设施的场景。
SQL 接口降低学习成本，支持向量与关系数据联合查询。

## 4. 选型建议
- 超大规模（亿级）：Milvus
- 中等规模 + 高性能过滤：Qdrant
- 小规模 + 已有 PG：pgvector
""", metadata={"source": "vector_db_guide.md"})

    # 评估问题集
    eval_questions = [
        {
            "question": "Milvus 的架构是什么样的？",
            "expected_keywords": ["存算分离", "云原生", "Proxy", "Coordinator"],
        },
        {
            "question": "哪个向量数据库适合小团队使用？",
            "expected_keywords": ["pgvector", "PostgreSQL", "小规模", "SQL"],
        },
        {
            "question": "Qdrant 用什么语言实现？有什么压缩方案？",
            "expected_keywords": ["Rust", "quantization", "压缩"],
        },
        {
            "question": "Milvus 的性能数据如何？",
            "expected_keywords": ["5ms", "10000", "1亿", "recall"],
        },
    ]

    # 分块策略
    chunkers = [
        FixedSizeChunker(chunk_size=300, overlap=0),
        RecursiveChunker(chunk_size=300, overlap=50),
        StructureChunker(),
    ]

    # 运行评估
    evaluator = ChunkingEvaluator()
    all_results = []

    for chunker in chunkers:
        result = evaluator.evaluate_chunker(chunker, [sample_doc], eval_questions)
        all_results.append(result)

    # 汇总对比
    print("\n" + "=" * 80)
    print("分块策略对比汇总")
    print("=" * 80)
    print(f"{'策略':<15} {'分块数':<8} {'平均长度':<10} {'耗时(s)':<10} {'平均召回':<10}")
    print("-" * 53)
    for r in all_results:
        print(f"{r['chunker']:<15} {r['num_chunks']:<8} {r['avg_chunk_size']:<10.0f} "
              f"{r['chunk_time']:<10.2f} {r['avg_recall']:<10.2f}")


if __name__ == "__main__":
    main()
