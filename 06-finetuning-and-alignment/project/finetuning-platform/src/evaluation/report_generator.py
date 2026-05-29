"""
评估报告生成器
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional
from dataclasses import asdict

from .evaluator import EvalResult


class ReportGenerator:
    """评估报告生成器"""

    def generate(self, eval_result: EvalResult, baseline: Optional[Dict] = None,
                output_dir: str = "./reports") -> str:
        """生成评估报告"""
        os.makedirs(output_dir, exist_ok=True)

        report = {
            "metadata": {
                "model_path": eval_result.model_path,
                "eval_time": datetime.now().isoformat(),
                "eval_duration_sec": eval_result.eval_time_sec,
                "passed": eval_result.passed,
            },
            "benchmarks": eval_result.benchmarks,
            "custom_metrics": eval_result.custom_metrics,
            "safety": eval_result.safety_scores,
        }

        # 与 baseline 对比
        if baseline:
            report["comparison"] = self._compare(eval_result, baseline)

        # 保存 JSON
        report_path = os.path.join(output_dir, "eval_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 生成文本摘要
        summary = self._format_summary(report)
        summary_path = os.path.join(output_dir, "eval_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)

        print(summary)
        return report_path

    def _compare(self, current: EvalResult, baseline: Dict) -> Dict:
        """与 baseline 对比"""
        comparison = {}
        for bench, score in current.benchmarks.items():
            base_score = baseline.get("benchmarks", {}).get(bench, 0)
            comparison[bench] = {
                "current": score,
                "baseline": base_score,
                "delta": score - base_score,
                "improved": score > base_score,
            }
        return comparison

    def _format_summary(self, report: Dict) -> str:
        """格式化文本摘要"""
        lines = []
        lines.append("=" * 60)
        lines.append("模型评估报告")
        lines.append("=" * 60)
        lines.append(f"模型: {report['metadata']['model_path']}")
        lines.append(f"时间: {report['metadata']['eval_time']}")
        lines.append(f"状态: {'通过' if report['metadata']['passed'] else '未通过'}")
        lines.append("")

        if report["benchmarks"]:
            lines.append("基准测试:")
            for bench, score in report["benchmarks"].items():
                lines.append(f"  {bench}: {score:.2%}")

        if report["custom_metrics"]:
            lines.append("\n自定义指标:")
            for metric, score in report["custom_metrics"].items():
                lines.append(f"  {metric}: {score:.2%}")

        if report["safety"]:
            lines.append("\n安全性:")
            for metric, score in report["safety"].items():
                lines.append(f"  {metric}: {score:.2%}")

        if "comparison" in report:
            lines.append("\n与 Baseline 对比:")
            for bench, comp in report["comparison"].items():
                delta = comp["delta"]
                symbol = "+" if delta > 0 else ""
                lines.append(f"  {bench}: {comp['current']:.2%} "
                           f"({symbol}{delta:.2%})")

        lines.append("")
        return "\n".join(lines)
