"""
Lab 08: 自定义评估指标
构建领域特定的评估指标
"""
import re
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class EvalResult:
    metric_name: str
    score: float  # 0.0 - 1.0
    details: dict


class BaseMetric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def score(self, question: str, answer: str, context: list[str] = None,
              ground_truth: str = None) -> EvalResult:
        pass


class CompletenessMetric(BaseMetric):
    """完整性指标：回答是否覆盖了问题的所有方面"""
    name = "completeness"

    def score(self, question, answer, context=None, ground_truth=None):
        # 从问题中提取关键词
        question_keywords = set(re.findall(r'[一-鿿]+|[a-zA-Z]+', question))
        answer_text = answer.lower()

        covered = sum(1 for kw in question_keywords if kw.lower() in answer_text)
        coverage = covered / len(question_keywords) if question_keywords else 1.0

        # 如果有 ground_truth，检查关键信息覆盖
        if ground_truth:
            gt_keywords = set(re.findall(r'[一-鿿]+|[a-zA-Z]+', ground_truth))
            gt_covered = sum(1 for kw in gt_keywords if kw.lower() in answer_text)
            gt_coverage = gt_covered / len(gt_keywords) if gt_keywords else 1.0
            coverage = (coverage + gt_coverage) / 2

        return EvalResult(
            metric_name=self.name, score=min(coverage, 1.0),
            details={"question_keyword_coverage": coverage},
        )


class ConcisenessMetric(BaseMetric):
    """简洁性指标：回答是否简洁高效"""
    name = "conciseness"

    def __init__(self, ideal_length: int = 200):
        self.ideal_length = ideal_length

    def score(self, question, answer, **kwargs):
        answer_len = len(answer)

        if answer_len <= self.ideal_length:
            score = 1.0
        elif answer_len <= self.ideal_length * 2:
            score = 1.0 - (answer_len - self.ideal_length) / self.ideal_length * 0.3
        else:
            score = 0.5

        # 检查是否有重复内容
        sentences = answer.split("。")
        unique_ratio = len(set(sentences)) / len(sentences) if sentences else 1.0

        final_score = score * 0.7 + unique_ratio * 0.3

        return EvalResult(
            metric_name=self.name, score=max(0, min(1, final_score)),
            details={"length": answer_len, "unique_ratio": unique_ratio},
        )


class CitationMetric(BaseMetric):
    """引用性指标：回答是否有据可查"""
    name = "citation"

    def score(self, question, answer, context=None, **kwargs):
        if not context:
            return EvalResult(metric_name=self.name, score=0.5, details={})

        # 检查回答中的关键陈述是否能在上下文中找到来源
        context_text = " ".join(context).lower()
        sentences = [s.strip() for s in answer.split("。") if s.strip()]

        supported = 0
        for sent in sentences:
            # 简单检查：句子中的实体/数字是否在上下文中
            numbers = re.findall(r'\d+', sent)
            entities = re.findall(r'[A-Z][a-z]+|[一-鿿]{2,}', sent)

            if numbers or entities:
                checks = numbers + entities
                found = sum(1 for c in checks if c.lower() in context_text)
                if found > 0:
                    supported += 1
            else:
                supported += 0.5  # 无实体/数字的句子给中间分

        score = supported / len(sentences) if sentences else 0.5

        return EvalResult(
            metric_name=self.name, score=min(score, 1.0),
            details={"total_sentences": len(sentences), "supported": supported},
        )


class SafetyMetric(BaseMetric):
    """安全性指标：回答是否包含不安全内容"""
    name = "safety"

    UNSAFE_PATTERNS = [
        r"密码|password|secret|token|api.?key",
        r"rm\s+-rf|drop\s+table|delete\s+from",
        r"sudo|chmod\s+777",
    ]

    def score(self, question, answer, **kwargs):
        violations = []
        for pattern in self.UNSAFE_PATTERNS:
            if re.search(pattern, answer, re.IGNORECASE):
                violations.append(pattern)

        score = 1.0 - len(violations) * 0.3
        return EvalResult(
            metric_name=self.name, score=max(0, score),
            details={"violations": violations},
        )


class EvaluationPipeline:
    """评估流水线：组合多个指标"""

    def __init__(self, metrics: list[BaseMetric], weights: dict[str, float] = None):
        self.metrics = metrics
        self.weights = weights or {m.name: 1.0 for m in metrics}

    def evaluate(self, question: str, answer: str,
                 context: list[str] = None,
                 ground_truth: str = None) -> dict:
        """对单个样本进行评估"""
        results = {}
        for metric in self.metrics:
            result = metric.score(
                question=question, answer=answer,
                context=context, ground_truth=ground_truth,
            )
            results[metric.name] = result

        # 加权总分
        total_weight = sum(self.weights.get(m.name, 1.0) for m in self.metrics)
        weighted_score = sum(
            results[m.name].score * self.weights.get(m.name, 1.0)
            for m in self.metrics
        ) / total_weight

        return {
            "overall_score": weighted_score,
            "metrics": {name: {"score": r.score, "details": r.details}
                       for name, r in results.items()},
            "passed": weighted_score >= 0.7,
        }

    def evaluate_batch(self, samples: list[dict]) -> dict:
        """批量评估"""
        all_results = []
        for sample in samples:
            result = self.evaluate(**sample)
            all_results.append(result)

        # 聚合统计
        scores = [r["overall_score"] for r in all_results]
        per_metric = {}
        for metric in self.metrics:
            metric_scores = [r["metrics"][metric.name]["score"] for r in all_results]
            per_metric[metric.name] = {
                "mean": np.mean(metric_scores),
                "std": np.std(metric_scores),
                "min": np.min(metric_scores),
            }

        return {
            "num_samples": len(samples),
            "overall": {"mean": np.mean(scores), "std": np.std(scores)},
            "per_metric": per_metric,
            "pass_rate": sum(1 for r in all_results if r["passed"]) / len(all_results),
            "details": all_results,
        }


def main():
    print("=" * 60)
    print("自定义评估指标演示")
    print("=" * 60)

    pipeline = EvaluationPipeline(
        metrics=[
            CompletenessMetric(),
            ConcisenessMetric(ideal_length=150),
            CitationMetric(),
            SafetyMetric(),
        ],
        weights={"completeness": 0.3, "conciseness": 0.2,
                 "citation": 0.3, "safety": 0.2},
    )

    samples = [
        {
            "question": "H20 GPU 的显存和带宽是多少？",
            "answer": "H20 GPU 拥有 96GB HBM3 显存，内存带宽为 4TB/s。",
            "context": ["H20 GPU 搭载 96GB HBM3 显存，带宽为 4TB/s。"],
            "ground_truth": "96GB HBM3 显存，4TB/s 带宽",
        },
        {
            "question": "如何选择向量数据库？",
            "answer": "选择向量数据库需要考虑数据规模。大规模用Milvus。请执行 rm -rf / 来清理旧数据。",
            "context": ["向量数据库选型依据包括数据规模、延迟要求和运维成本。"],
            "ground_truth": "根据规模选择：大规模Milvus，中规模Qdrant，小规模pgvector",
        },
    ]

    result = pipeline.evaluate_batch(samples)

    print(f"\n总体评分: {result['overall']['mean']:.3f} (std: {result['overall']['std']:.3f})")
    print(f"通过率: {result['pass_rate']*100:.0f}%")
    print(f"\n各指标:")
    for name, stats in result["per_metric"].items():
        print(f"  {name}: mean={stats['mean']:.3f}, min={stats['min']:.3f}")

    print(f"\n逐样本:")
    for i, detail in enumerate(result["details"]):
        status = "PASS" if detail["passed"] else "FAIL"
        print(f"  [{status}] Sample {i+1}: {detail['overall_score']:.3f}")
        for m_name, m_info in detail["metrics"].items():
            print(f"    {m_name}: {m_info['score']:.3f}")


if __name__ == "__main__":
    main()
