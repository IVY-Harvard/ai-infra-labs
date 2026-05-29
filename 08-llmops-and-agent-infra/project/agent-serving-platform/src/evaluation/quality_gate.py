"""质量门禁 - 版本发布前的自动化质量检查"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GateResult:
    passed: bool
    scores: dict
    failed_checks: list[str] = field(default_factory=list)
    message: str = ""


class QualityGate:
    """
    质量门禁
    在 Agent/Prompt/RAG 配置变更发布前，自动检查质量
    """

    def __init__(self):
        self.thresholds = {
            "faithfulness": 0.85,
            "relevancy": 0.80,
            "correctness": 0.75,
            "safety": 0.95,
            "latency_p95_ms": 3000,
            "error_rate": 0.05,
        }

    def check(self, eval_results: dict) -> GateResult:
        """执行质量门禁检查"""
        failed = []

        for metric, threshold in self.thresholds.items():
            actual = eval_results.get(metric)
            if actual is None:
                continue

            if metric in ("latency_p95_ms", "error_rate"):
                # 越低越好
                if actual > threshold:
                    failed.append(
                        f"{metric}: {actual:.3f} > {threshold:.3f}"
                    )
            else:
                # 越高越好
                if actual < threshold:
                    failed.append(
                        f"{metric}: {actual:.3f} < {threshold:.3f}"
                    )

        passed = len(failed) == 0

        return GateResult(
            passed=passed,
            scores=eval_results,
            failed_checks=failed,
            message="质量检查通过" if passed else f"质量检查失败: {'; '.join(failed)}",
        )

    def update_threshold(self, metric: str, value: float):
        """更新阈值"""
        if metric in self.thresholds:
            self.thresholds[metric] = value
