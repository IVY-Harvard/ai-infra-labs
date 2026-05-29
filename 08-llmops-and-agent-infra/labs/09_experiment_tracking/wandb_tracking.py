"""
Lab 09: W&B 实验追踪
使用 Weights & Biases 追踪 LLM 实验
"""
import os
import json
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb 未安装: pip install wandb")


def demo_wandb_tracking():
    """W&B 追踪演示"""
    if not WANDB_AVAILABLE:
        _show_concept()
        return

    print("=" * 60)
    print("Weights & Biases 实验追踪演示")
    print("=" * 60)

    # 初始化（离线模式，不需要账号）
    os.environ["WANDB_MODE"] = "offline"

    run = wandb.init(
        project="rag-optimization",
        name="hybrid_search_v2",
        config={
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "embedding_model": "bge-m3",
            "reranker": "bge-reranker-v2-m3",
            "top_k": 10,
            "hyde_enabled": True,
            "llm_model": "qwen2.5-72b",
            "temperature": 0.3,
        },
    )

    # 模拟逐步评估（类似训练过程中的 step logging）
    eval_questions = [
        "H20 GPU 的显存是多少？",
        "如何部署 72B 模型？",
        "推荐什么向量数据库？",
        "Reranker 的作用是什么？",
        "RAG 的评估指标有哪些？",
    ]

    for step, question in enumerate(eval_questions):
        # 模拟评估分数
        metrics = {
            "faithfulness": 0.85 + np.random.uniform(0, 0.15),
            "relevancy": 0.80 + np.random.uniform(0, 0.15),
            "latency_ms": 1000 + np.random.uniform(0, 500),
        }
        wandb.log(metrics, step=step)
        print(f"  Step {step}: faith={metrics['faithfulness']:.3f}, "
              f"rel={metrics['relevancy']:.3f}")

    # 记录评估表格
    eval_table = wandb.Table(
        columns=["question", "answer", "faithfulness", "relevancy"]
    )
    for q in eval_questions:
        eval_table.add_data(
            q, f"回答: {q}...",
            0.85 + np.random.uniform(0, 0.15),
            0.80 + np.random.uniform(0, 0.15),
        )
    wandb.log({"evaluation_results": eval_table})

    # 记录聚合指标
    wandb.summary["best_faithfulness"] = 0.95
    wandb.summary["best_relevancy"] = 0.92
    wandb.summary["avg_latency_ms"] = 1250

    wandb.finish()
    print("\n✓ W&B 追踪完成")
    print("  离线模式下数据保存在 ./wandb/ 目录")
    print("  使用 wandb sync 可上传到云端")


def _show_concept():
    """概念展示"""
    print("=" * 60)
    print("W&B 实验追踪概念")
    print("=" * 60)
    print("""
W&B vs MLflow 对比：

特性        | W&B               | MLflow
可视化      | 强大（交互式图表） | 基础
部署方式    | Cloud / 自托管     | 自托管
团队协作    | 原生支持           | 基础
LLM 追踪   | W&B Prompts        | mlflow.llm
报告        | W&B Reports        | 需自行构建

W&B 核心 API:
  wandb.init(project="...", config={...})   # 初始化
  wandb.log({"metric": value}, step=N)      # 逐步记录
  wandb.log({"table": wandb.Table(...)})    # 表格数据
  wandb.summary["key"] = value              # 汇总指标
  wandb.finish()                            # 结束
""")


if __name__ == "__main__":
    demo_wandb_tracking()
