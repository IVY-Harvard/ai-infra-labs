"""
内置基准测试
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class BenchmarkTask:
    """基准测试任务"""
    name: str
    description: str
    task_type: str  # multichoice, generation, code
    lm_eval_task: str  # lm-eval-harness 中的任务名


# 预定义基准测试
BENCHMARKS = {
    # 通用能力
    "mmlu": BenchmarkTask("MMLU", "通用知识 57 学科", "multichoice", "mmlu"),
    "arc": BenchmarkTask("ARC-Challenge", "科学推理", "multichoice", "arc_challenge"),
    "hellaswag": BenchmarkTask("HellaSwag", "常识推理", "multichoice", "hellaswag"),

    # 数学推理
    "gsm8k": BenchmarkTask("GSM8K", "数学应用题", "generation", "gsm8k"),

    # 代码
    "humaneval": BenchmarkTask("HumanEval", "代码生成", "code", "humaneval"),
    "mbpp": BenchmarkTask("MBPP", "Python 编程", "code", "mbpp"),

    # 中文
    "ceval": BenchmarkTask("C-Eval", "中文知识", "multichoice", "ceval-valid"),
    "cmmlu": BenchmarkTask("CMMLU", "中文理解", "multichoice", "cmmlu"),
}


class BenchmarkRunner:
    """基准测试运行器"""

    def __init__(self, model_path: str):
        self.model_path = model_path

    def run(self, benchmark_names: List[str], batch_size: int = 8) -> Dict[str, float]:
        """运行多个基准测试"""
        results = {}
        for name in benchmark_names:
            if name not in BENCHMARKS:
                print(f"  跳过未知基准: {name}")
                continue

            bench = BENCHMARKS[name]
            print(f"  运行 {bench.name}...")

            score = self._run_single(bench, batch_size)
            results[name] = score

        return results

    def _run_single(self, bench: BenchmarkTask, batch_size: int) -> float:
        """运行单个基准测试"""
        import subprocess

        cmd = [
            "lm_eval", "--model", "hf",
            "--model_args", f"pretrained={self.model_path},trust_remote_code=True,dtype=bfloat16",
            "--tasks", bench.lm_eval_task,
            "--batch_size", str(batch_size),
            "--output_path", f"./eval_results/{bench.name}",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                return self._parse_score(result.stdout)
            return 0.0
        except Exception as e:
            print(f"    评测失败: {e}")
            return 0.0

    def _parse_score(self, output: str) -> float:
        """从输出中解析分数"""
        import re
        # 尝试匹配 acc 或 acc_norm
        match = re.search(r'acc[_norm]*\s*\|?\s*([\d.]+)', output)
        if match:
            return float(match.group(1))
        return 0.0

    def get_available_benchmarks(self) -> Dict[str, str]:
        """列出可用的基准测试"""
        return {name: bench.description for name, bench in BENCHMARKS.items()}
