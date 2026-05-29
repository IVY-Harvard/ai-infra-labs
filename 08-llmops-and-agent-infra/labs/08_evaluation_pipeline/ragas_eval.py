"""
Lab 08: Ragas RAG 评估
使用 Ragas 框架对 RAG 系统进行多维度评估
"""
import os
from typing import Optional

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness, answer_relevancy,
        context_precision, context_recall,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("ragas 未安装: pip install ragas datasets")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-72b-instruct")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:8001/v1")


# =============================================================================
# 评估数据集
# =============================================================================

EVAL_DATASET = {
    "question": [
        "H20 GPU 的显存是多少？",
        "如何部署 72B 模型？",
        "RAG 系统推荐什么向量数据库？",
        "Reranker 的作用是什么？",
    ],
    "answer": [
        "H20 GPU 拥有 96GB HBM3 显存。",
        "72B 模型需要使用 4 张 H20 GPU 通过 Tensor Parallel (TP=4) 方式部署，推荐使用 vLLM 推理框架。",
        "对于大规模场景推荐 Milvus，中等规模推荐 Qdrant，小规模可使用 pgvector。",
        "Reranker 用于对检索结果进行精细化重排序，使用 Cross-Encoder 模型对 query-doc 对打分，提升检索精度。",
    ],
    "contexts": [
        ["H20 GPU 搭载 96GB HBM3 显存，带宽为 4TB/s。采用 Hopper 架构。"],
        ["部署 70B 模型需要 4 张 H20（TP=4），吞吐量约 350-400 tokens/s。推荐使用 vLLM 推理引擎。"],
        ["向量数据库选型：Milvus 适合大规模（亿级），Qdrant 适合中规模（千万级），pgvector 适合小规模（百万级）。"],
        ["Reranker 使用 Cross-Encoder 对检索到的文档进行精细打分。相比 Bi-Encoder，精度更高但速度更慢。常用于 top-50 召回后精排到 top-5。"],
    ],
    "ground_truth": [
        "96GB HBM3",
        "使用 4 张 H20 GPU，通过 Tensor Parallel 部署，推荐 vLLM",
        "大规模用 Milvus，中规模用 Qdrant，小规模用 pgvector",
        "Reranker 使用 Cross-Encoder 对检索结果重排序，提升精度",
    ],
}


# =============================================================================
# Ragas 评估
# =============================================================================

def run_ragas_evaluation():
    """运行 Ragas 评估"""
    if not RAGAS_AVAILABLE:
        print("Ragas 不可用，展示概念")
        _show_concept()
        return

    print("=" * 60)
    print("Ragas RAG 评估")
    print("=" * 60)

    # 准备数据集
    dataset = Dataset.from_dict(EVAL_DATASET)

    # 配置 LLM 和 Embeddings
    llm = ChatOpenAI(
        base_url=LLM_BASE_URL, model=LLM_MODEL,
        api_key="not-needed", temperature=0,
    )
    embeddings = OpenAIEmbeddings(
        base_url=EMBEDDING_BASE_URL, model="bge-m3", api_key="not-needed",
    )

    # 运行评估
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    print("\n评估中...")
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )

    # 打印结果
    print(f"\n{'='*60}")
    print("评估结果")
    print(f"{'='*60}")
    for metric_name, score in results.items():
        print(f"  {metric_name}: {score:.4f}")

    # 逐题分析
    print(f"\n{'='*60}")
    print("逐题分析")
    print(f"{'='*60}")
    df = results.to_pandas()
    for idx, row in df.iterrows():
        print(f"\n  Q{idx+1}: {EVAL_DATASET['question'][idx]}")
        print(f"  Faithfulness: {row.get('faithfulness', 'N/A'):.3f}")
        print(f"  Relevancy: {row.get('answer_relevancy', 'N/A'):.3f}")


def _show_concept():
    """概念展示"""
    print("=" * 60)
    print("Ragas 评估概念")
    print("=" * 60)
    print(f"""
Ragas 核心指标：

1. Faithfulness（忠实度）0-1
   回答中的每个陈述是否能从上下文推导出来
   检测方法：将回答分解为原子声明，逐一验证

2. Answer Relevancy（答案相关性）0-1
   回答是否与问题相关（不答非所问）
   检测方法：从回答反向生成问题，计算与原问题的相似度

3. Context Precision（上下文精度）0-1
   检索到的上下文中，有用信息排在前面的比例
   类似搜索引擎的 NDCG

4. Context Recall（上下文召回）0-1
   ground truth 中的信息是否都在检索到的上下文中出现
   检测真实所需信息的覆盖率

评估数据格式：
  question:     用户问题
  answer:       RAG 系统的实际回答
  contexts:     检索到的上下文文档列表
  ground_truth: 标准答案（用于 recall 计算）
""")

    # 模拟评估结果
    print("\n模拟评估结果：")
    mock_results = {
        "faithfulness": 0.92,
        "answer_relevancy": 0.88,
        "context_precision": 0.85,
        "context_recall": 0.90,
    }
    for metric, score in mock_results.items():
        status = "✓" if score >= 0.85 else "⚠"
        print(f"  {status} {metric}: {score:.2f}")


if __name__ == "__main__":
    run_ragas_evaluation()
