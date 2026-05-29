"""
Lab 09: MLflow 实验追踪
使用 MLflow 追踪 RAG 实验的参数、指标和配置
"""
import os
import json
import time
from pathlib import Path

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("mlflow 未安装: pip install mlflow")


def demo_mlflow_tracking():
    """MLflow 实验追踪演示"""
    if not MLFLOW_AVAILABLE:
        _show_concept()
        return

    print("=" * 60)
    print("MLflow 实验追踪演示")
    print("=" * 60)

    # 设置追踪 URI（本地文件存储）
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("rag-optimization")

    # 模拟多组实验
    experiments = [
        {
            "name": "baseline_naive_rag",
            "params": {
                "chunk_size": 500, "chunk_overlap": 50,
                "embedding_model": "bge-m3", "top_k": 5,
                "reranker": "none", "hyde": False,
                "llm_model": "qwen2.5-72b", "temperature": 0.3,
            },
            "metrics": {
                "faithfulness": 0.78, "relevancy": 0.72,
                "correctness": 0.70, "latency_p50_ms": 800,
                "latency_p95_ms": 1500, "cost_per_query": 0.015,
            },
        },
        {
            "name": "advanced_hyde_reranker",
            "params": {
                "chunk_size": 1000, "chunk_overlap": 200,
                "embedding_model": "bge-m3", "top_k": 10,
                "reranker": "bge-reranker-v2-m3", "hyde": True,
                "llm_model": "qwen2.5-72b", "temperature": 0.3,
            },
            "metrics": {
                "faithfulness": 0.92, "relevancy": 0.88,
                "correctness": 0.85, "latency_p50_ms": 1200,
                "latency_p95_ms": 2500, "cost_per_query": 0.025,
            },
        },
        {
            "name": "hybrid_search_optimized",
            "params": {
                "chunk_size": 800, "chunk_overlap": 100,
                "embedding_model": "bge-m3", "top_k": 10,
                "reranker": "bge-reranker-v2-m3", "hyde": True,
                "search_type": "hybrid", "bm25_weight": 0.3,
                "llm_model": "qwen2.5-72b", "temperature": 0.2,
            },
            "metrics": {
                "faithfulness": 0.94, "relevancy": 0.91,
                "correctness": 0.88, "latency_p50_ms": 1300,
                "latency_p95_ms": 2800, "cost_per_query": 0.028,
            },
        },
    ]

    for exp in experiments:
        with mlflow.start_run(run_name=exp["name"]):
            # 记录参数
            mlflow.log_params(exp["params"])

            # 记录指标
            mlflow.log_metrics(exp["metrics"])

            # 记录配置文件
            config_path = Path(f"./configs/{exp['name']}.json")
            config_path.parent.mkdir(exist_ok=True)
            config_path.write_text(json.dumps(exp["params"], indent=2))
            mlflow.log_artifact(str(config_path))

            # 添加标签
            mlflow.set_tags({
                "team": "ml-platform",
                "stage": "experiment",
                "has_reranker": str(exp["params"].get("reranker", "none") != "none"),
            })

            print(f"\n✓ 记录实验: {exp['name']}")
            print(f"  Faithfulness: {exp['metrics']['faithfulness']:.2f}")
            print(f"  Relevancy: {exp['metrics']['relevancy']:.2f}")
            print(f"  Latency P95: {exp['metrics']['latency_p95_ms']}ms")

    # 查询实验结果
    print(f"\n{'='*60}")
    print("实验结果查询")
    print(f"{'='*60}")

    client = MlflowClient()
    experiment = client.get_experiment_by_name("rag-optimization")

    if experiment:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.faithfulness DESC"],
        )

        print(f"\n按 Faithfulness 排序:")
        for run in runs:
            name = run.info.run_name
            faith = run.data.metrics.get("faithfulness", 0)
            rel = run.data.metrics.get("relevancy", 0)
            lat = run.data.metrics.get("latency_p95_ms", 0)
            print(f"  {name}: faith={faith:.2f}, rel={rel:.2f}, lat={lat:.0f}ms")

    print(f"\n启动 MLflow UI: mlflow ui --port 5001")
    print(f"然后访问 http://localhost:5001")


def _show_concept():
    """概念展示"""
    print("=" * 60)
    print("MLflow 实验追踪概念")
    print("=" * 60)
    print("""
MLflow 核心概念：

1. Experiment（实验）
   一组相关的运行，如 "rag-optimization"

2. Run（运行）
   一次具体的实验执行
   包含：Parameters、Metrics、Artifacts、Tags

3. 核心 API：
   mlflow.log_params({"chunk_size": 500})   # 记录参数
   mlflow.log_metrics({"faith": 0.92})      # 记录指标
   mlflow.log_artifact("config.yaml")       # 记录文件
   mlflow.set_tags({"team": "ml"})          # 记录标签

4. MLflow UI
   可视化实验对比、参数搜索、指标趋势

5. Model Registry
   管理模型版本：Staging → Production → Archived
""")


if __name__ == "__main__":
    demo_mlflow_tracking()
