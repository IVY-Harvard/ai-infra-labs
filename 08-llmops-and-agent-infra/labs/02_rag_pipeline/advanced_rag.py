"""
Lab 02: Advanced RAG - HyDE + Reranker
HyDE: 用 LLM 生成假设性答案来改善检索
Reranker: 使用 Cross-Encoder 对检索结果精细排序
"""
import os
from typing import Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader


# =============================================================================
# 配置
# =============================================================================

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-72b-instruct")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:8001/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")


def get_llm(temperature: float = 0.3):
    return ChatOpenAI(
        base_url=LLM_BASE_URL, model=LLM_MODEL,
        api_key="not-needed", temperature=temperature,
    )


def get_embeddings():
    return OpenAIEmbeddings(
        base_url=EMBEDDING_BASE_URL, model=EMBEDDING_MODEL, api_key="not-needed",
    )


# =============================================================================
# HyDE (Hypothetical Document Embedding)
# =============================================================================

class HyDERetriever:
    """
    HyDE 检索器
    原理：先用 LLM 生成假设性答案，再用该答案的 embedding 检索
    效果：假设性答案与真实文档在 embedding 空间更近似
    """

    def __init__(self, vectorstore, llm, top_k: int = 5):
        self.vectorstore = vectorstore
        self.llm = llm
        self.top_k = top_k

        self.hyde_prompt = ChatPromptTemplate.from_template(
            """请针对以下问题，写一段可能的回答（即使你不确定也请尝试）。
这段回答将用于文档检索，所以请包含尽可能多的相关关键词和概念。

问题：{question}

假设性回答："""
        )

        self.hyde_chain = self.hyde_prompt | self.llm | StrOutputParser()

    def retrieve(self, question: str) -> list:
        """HyDE 检索：生成假设答案 → 用假设答案检索"""
        # Step 1: 生成假设性答案
        hypothetical_answer = self.hyde_chain.invoke({"question": question})
        print(f"  [HyDE] 假设性答案: {hypothetical_answer[:100]}...")

        # Step 2: 用假设答案检索（而非原始查询）
        docs = self.vectorstore.similarity_search(
            hypothetical_answer, k=self.top_k
        )

        return docs

    def retrieve_with_comparison(self, question: str) -> dict:
        """对比 HyDE vs 直接检索"""
        # 直接检索
        direct_docs = self.vectorstore.similarity_search(
            question, k=self.top_k
        )

        # HyDE 检索
        hyde_docs = self.retrieve(question)

        return {
            "direct": direct_docs,
            "hyde": hyde_docs,
        }


# =============================================================================
# Reranker
# =============================================================================

class CrossEncoderReranker:
    """
    Cross-Encoder Reranker
    对 (query, document) 对进行精细化打分
    比 Bi-Encoder (Embedding) 更精确但更慢
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        try:
            from FlagEmbedding import FlagReranker
            self.reranker = FlagReranker(model_name, use_fp16=True)
            self.available = True
        except ImportError:
            print("  [警告] FlagEmbedding 未安装，使用 LLM 模拟 Reranker")
            self.reranker = None
            self.available = False

    def rerank(self, query: str, documents: list,
               top_k: int = 3) -> list:
        """重排序文档"""
        if not documents:
            return []

        if self.available and self.reranker:
            return self._rerank_with_model(query, documents, top_k)
        else:
            return self._rerank_with_llm(query, documents, top_k)

    def _rerank_with_model(self, query, documents, top_k):
        """使用 Cross-Encoder 模型重排序"""
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.reranker.compute_score(pairs)

        if isinstance(scores, float):
            scores = [scores]

        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        print(f"  [Reranker] 重排序分数: {[f'{s:.3f}' for _, s in scored_docs[:top_k]]}")

        return [doc for doc, _ in scored_docs[:top_k]]

    def _rerank_with_llm(self, query, documents, top_k):
        """LLM 模拟重排序（当模型不可用时的 fallback）"""
        llm = get_llm(temperature=0)
        scored_docs = []

        for doc in documents:
            prompt = f"""请评估以下文档与查询的相关性，返回 0-10 的分数。

查询：{query}
文档：{doc.page_content[:300]}

只返回数字分数："""
            try:
                score_str = llm.predict(prompt).strip()
                score = float(score_str)
            except (ValueError, Exception):
                score = 5.0
            scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]


# =============================================================================
# Advanced RAG Pipeline
# =============================================================================

class AdvancedRAG:
    """
    完整的 Advanced RAG 流水线：
    Query → HyDE → 向量检索 → Reranker → LLM 生成
    """

    def __init__(self, vectorstore, use_hyde: bool = True,
                 use_reranker: bool = True, top_k_retrieval: int = 10,
                 top_k_rerank: int = 3):
        self.vectorstore = vectorstore
        self.llm = get_llm()
        self.use_hyde = use_hyde
        self.use_reranker = use_reranker
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank

        if use_hyde:
            self.hyde_retriever = HyDERetriever(
                vectorstore, self.llm, top_k=top_k_retrieval
            )

        if use_reranker:
            self.reranker = CrossEncoderReranker()

        self.qa_prompt = ChatPromptTemplate.from_template(
            """基于以下上下文回答问题。如果上下文不包含答案，请说明。

上下文：
{context}

问题：{question}

请提供准确、有据可查的回答："""
        )

    def query(self, question: str) -> dict:
        """执行 Advanced RAG 查询"""
        print(f"\n{'='*60}")
        print(f"查询: {question}")
        print(f"配置: HyDE={self.use_hyde}, Reranker={self.use_reranker}")
        print(f"{'='*60}")

        # Step 1: 检索
        if self.use_hyde:
            print("\n[Step 1] HyDE 检索...")
            docs = self.hyde_retriever.retrieve(question)
        else:
            print("\n[Step 1] 直接向量检索...")
            docs = self.vectorstore.similarity_search(
                question, k=self.top_k_retrieval
            )
        print(f"  检索到 {len(docs)} 个文档块")

        # Step 2: Reranker
        if self.use_reranker and len(docs) > self.top_k_rerank:
            print("\n[Step 2] Reranker 重排序...")
            docs = self.reranker.rerank(
                question, docs, top_k=self.top_k_rerank
            )
            print(f"  保留 Top-{self.top_k_rerank} 文档")
        else:
            docs = docs[:self.top_k_rerank]

        # Step 3: 生成回答
        print("\n[Step 3] LLM 生成回答...")
        context = "\n\n---\n\n".join(
            [doc.page_content for doc in docs]
        )
        chain = self.qa_prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        result = {
            "question": question,
            "answer": answer,
            "source_documents": docs,
            "config": {
                "hyde": self.use_hyde,
                "reranker": self.use_reranker,
            },
        }

        print(f"\n回答: {answer}")
        return result


# =============================================================================
# 对比实验
# =============================================================================

def run_comparison(vectorstore, questions: list[str]):
    """对比 Naive RAG vs Advanced RAG"""
    configs = [
        {"name": "Naive RAG", "hyde": False, "reranker": False},
        {"name": "HyDE only", "hyde": True, "reranker": False},
        {"name": "Reranker only", "hyde": False, "reranker": True},
        {"name": "HyDE + Reranker", "hyde": True, "reranker": True},
    ]

    results = {}
    for config in configs:
        rag = AdvancedRAG(
            vectorstore,
            use_hyde=config["hyde"],
            use_reranker=config["reranker"],
        )
        results[config["name"]] = []
        for q in questions:
            result = rag.query(q)
            results[config["name"]].append(result)

    # 打印对比表
    print("\n" + "=" * 80)
    print("对比结果摘要")
    print("=" * 80)
    for q_idx, q in enumerate(questions):
        print(f"\n问题 {q_idx+1}: {q}")
        for config_name, config_results in results.items():
            answer = config_results[q_idx]["answer"]
            print(f"  [{config_name}]: {answer[:100]}...")


# =============================================================================
# 主程序
# =============================================================================

def main():
    # 创建示例数据
    from pathlib import Path
    sample_dir = Path("./sample_docs")
    sample_dir.mkdir(exist_ok=True)

    sample_content = """
    # NVIDIA H20 GPU 技术规格

    ## 硬件参数
    H20 GPU 搭载 96GB HBM3 显存，带宽为 4TB/s。
    采用 Hopper 架构，支持 FP8 精度推理。
    TDP 功耗为 400W，支持 PCIe 5.0 和 NVLink 连接。

    ## 推理性能
    在 LLM 推理场景下，单张 H20 可运行 7B 参数模型，
    吞吐量约 2000 tokens/s。部署 70B 模型需要 4 张 H20（TP=4），
    吞吐量约 350-400 tokens/s。

    ## 与 A100 对比
    H20 的 HBM 带宽是 A100 的 2 倍，显存容量也更大（96GB vs 80GB）。
    但 H20 的计算单元数量（SM 数）少于 A100，
    因此训练性能不及 A100，但推理性能（尤其是大模型）更优。

    ## 部署建议
    8 张 H20 的集群可以同时部署：
    - 1 个 72B 模型实例（4 卡 TP）
    - 1 个 7B 模型实例（1 卡）
    - Embedding 服务（1 卡）
    - Reranker 服务（1 卡）
    - 预留 1 卡弹性扩展
    """

    (sample_dir / "h20_specs.txt").write_text(sample_content, encoding="utf-8")

    # 构建向量存储
    documents = TextLoader(str(sample_dir / "h20_specs.txt"), encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50
    ).split_documents(documents)

    vectorstore = Chroma.from_documents(
        chunks, get_embeddings(), persist_directory="./advanced_rag_db"
    )

    # 测试问题（包含模糊查询和精确查询）
    questions = [
        "H20 能跑多大的模型？",  # 模糊，HyDE 应该有帮助
        "H20 和 A100 的区别",    # 比较类问题
        "8 卡 H20 怎么分配资源？", # 具体规划问题
    ]

    # 运行对比
    run_comparison(vectorstore, questions)


if __name__ == "__main__":
    main()
