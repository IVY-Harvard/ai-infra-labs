"""
Lab 07: Prompt A/B 测试框架
"""
import os
import hashlib
import random
import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser


LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-72b-instruct")


@dataclass
class ABExperiment:
    name: str
    control_prompt: str
    treatment_prompt: str
    traffic_split: float = 0.5  # treatment 组的比例
    metrics: dict = field(default_factory=dict)
    control_results: list = field(default_factory=list)
    treatment_results: list = field(default_factory=list)
    status: str = "running"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class PromptABTest:
    """Prompt A/B 测试框架"""

    def __init__(self):
        self.experiments: dict[str, ABExperiment] = {}
        self.llm = ChatOpenAI(
            base_url=LLM_BASE_URL, model=LLM_MODEL,
            api_key="not-needed", temperature=0.3,
        )

    def create_experiment(self, name: str, control: str,
                          treatment: str, split: float = 0.5):
        """创建 A/B 实验"""
        self.experiments[name] = ABExperiment(
            name=name, control_prompt=control,
            treatment_prompt=treatment, traffic_split=split,
        )
        print(f"✓ 创建实验: {name} (split: {(1-split)*100:.0f}% / {split*100:.0f}%)")

    def route_request(self, experiment_name: str,
                      user_id: str) -> str:
        """确定性路由（同一用户始终看到同一版本）"""
        hash_val = hashlib.md5(
            f"{experiment_name}:{user_id}".encode()
        ).hexdigest()
        bucket = int(hash_val[:8], 16) / 0xFFFFFFFF

        exp = self.experiments[experiment_name]
        return "treatment" if bucket < exp.traffic_split else "control"

    def run_query(self, experiment_name: str, user_id: str,
                  question: str, context: str = "") -> dict:
        """运行查询并记录结果"""
        exp = self.experiments[experiment_name]
        group = self.route_request(experiment_name, user_id)

        # 选择 Prompt
        template_str = (
            exp.treatment_prompt if group == "treatment"
            else exp.control_prompt
        )
        prompt = ChatPromptTemplate.from_template(template_str)
        chain = prompt | self.llm | StrOutputParser()

        # 执行
        import time
        start = time.time()
        answer = chain.invoke({"question": question, "context": context})
        latency = (time.time() - start) * 1000

        result = {
            "user_id": user_id,
            "group": group,
            "question": question,
            "answer": answer,
            "latency_ms": latency,
            "timestamp": datetime.now().isoformat(),
        }

        if group == "treatment":
            exp.treatment_results.append(result)
        else:
            exp.control_results.append(result)

        return result

    def evaluate_experiment(self, experiment_name: str,
                            eval_fn=None) -> dict:
        """评估实验结果"""
        exp = self.experiments[experiment_name]

        if eval_fn is None:
            # 默认评估：使用 LLM 打分
            eval_fn = self._default_eval

        control_scores = [eval_fn(r) for r in exp.control_results]
        treatment_scores = [eval_fn(r) for r in exp.treatment_results]

        # 统计分析
        from scipy import stats
        if len(control_scores) > 1 and len(treatment_scores) > 1:
            t_stat, p_value = stats.ttest_ind(control_scores, treatment_scores)
        else:
            t_stat, p_value = 0, 1.0

        result = {
            "experiment": experiment_name,
            "control": {
                "n": len(control_scores),
                "mean_score": np.mean(control_scores) if control_scores else 0,
                "std_score": np.std(control_scores) if control_scores else 0,
                "mean_latency": np.mean([r["latency_ms"] for r in exp.control_results]) if exp.control_results else 0,
            },
            "treatment": {
                "n": len(treatment_scores),
                "mean_score": np.mean(treatment_scores) if treatment_scores else 0,
                "std_score": np.std(treatment_scores) if treatment_scores else 0,
                "mean_latency": np.mean([r["latency_ms"] for r in exp.treatment_results]) if exp.treatment_results else 0,
            },
            "statistical_test": {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
            },
            "recommendation": self._get_recommendation(
                control_scores, treatment_scores, p_value
            ),
        }
        return result

    def _default_eval(self, result: dict) -> float:
        """默认评估（基于回答长度和关键词 — 简化版）"""
        answer = result["answer"]
        score = min(len(answer) / 500, 1.0)  # 长度分
        if "根据" in answer or "基于" in answer:
            score += 0.1  # 有引用加分
        if "不确定" in answer or "无法" in answer:
            score -= 0.1  # 不确定减分
        return max(0, min(1, score))

    def _get_recommendation(self, control, treatment, p_value):
        if not control or not treatment:
            return "样本不足"
        if p_value >= 0.05:
            return "差异不显著，建议继续收集数据"
        if np.mean(treatment) > np.mean(control):
            return "Treatment 显著优于 Control，建议采纳"
        return "Control 更优，拒绝 Treatment"


def main():
    print("=" * 60)
    print("Prompt A/B 测试演示")
    print("=" * 60)

    ab = PromptABTest()

    # 创建实验
    ab.create_experiment(
        name="qa_prompt_v2",
        control="请回答以下问题：{question}",
        treatment="""你是一个专业的技术顾问。
请基于以下上下文回答问题。如果不确定，请说明。

上下文：{context}
问题：{question}

回答：""",
        split=0.5,
    )

    # 模拟请求
    test_data = [
        {"question": "H20 GPU 的显存是多少？", "context": "H20 GPU 有 96GB HBM3 显存"},
        {"question": "推荐什么向量数据库？", "context": "Milvus 适合大规模，Qdrant 适合中规模"},
        {"question": "RAG 的核心指标有哪些？", "context": "Faithfulness、Relevancy、Correctness"},
    ]

    for i in range(20):
        user_id = f"user_{i}"
        data = test_data[i % len(test_data)]
        result = ab.run_query("qa_prompt_v2", user_id, **data)
        group = result["group"]
        print(f"  user_{i} → {group}: {result['answer'][:50]}... ({result['latency_ms']:.0f}ms)")

    # 评估
    print(f"\n{'='*60}")
    print("实验结果")
    print(f"{'='*60}")
    eval_result = ab.evaluate_experiment("qa_prompt_v2")

    print(f"\nControl: n={eval_result['control']['n']}, "
          f"score={eval_result['control']['mean_score']:.3f}")
    print(f"Treatment: n={eval_result['treatment']['n']}, "
          f"score={eval_result['treatment']['mean_score']:.3f}")
    print(f"P-value: {eval_result['statistical_test']['p_value']:.4f}")
    print(f"建议: {eval_result['recommendation']}")


if __name__ == "__main__":
    main()
