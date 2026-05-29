"""
多基准测试套件：自动运行多个评测并汇总结果
集成 lm-eval-harness 和自定义评测

用法:
    python benchmark_suite.py --model ./my_model --tasks mmlu gsm8k humaneval
    python benchmark_suite.py --model ./my_model --full  # 运行所有测试
"""

import argparse
import json
import os
import subprocess
import time
from typing import Dict, List


class BenchmarkSuite:
    """多基准测试套件"""

    AVAILABLE_BENCHMARKS = {
        # 通用能力
        "mmlu": {"type": "lm_eval", "task": "mmlu", "desc": "通用知识(57学科)"},
        "arc_challenge": {"type": "lm_eval", "task": "arc_challenge", "desc": "科学推理"},
        "hellaswag": {"type": "lm_eval", "task": "hellaswag", "desc": "常识推理"},
        "winogrande": {"type": "lm_eval", "task": "winogrande", "desc": "指代消解"},

        # 数学/推理
        "gsm8k": {"type": "lm_eval", "task": "gsm8k", "desc": "数学应用题"},
        "math": {"type": "lm_eval", "task": "minerva_math", "desc": "数学竞赛"},

        # 代码
        "humaneval": {"type": "lm_eval", "task": "humaneval", "desc": "代码生成"},
        "mbpp": {"type": "lm_eval", "task": "mbpp", "desc": "Python编程"},

        # 中文
        "ceval": {"type": "lm_eval", "task": "ceval-valid", "desc": "中文知识"},
        "cmmlu": {"type": "lm_eval", "task": "cmmlu", "desc": "中文理解"},
    }

    PRESET_SUITES = {
        "quick": ["mmlu", "gsm8k", "humaneval"],
        "chinese": ["ceval", "cmmlu", "mmlu"],
        "reasoning": ["gsm8k", "math", "arc_challenge"],
        "full": list(AVAILABLE_BENCHMARKS.keys()),
    }

    def __init__(self, model_path: str, output_dir: str = "./benchmark_results"):
        self.model_path = model_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_lm_eval(self, task: str, batch_size: int = 8,
                    num_fewshot: int = None) -> Dict:
        """使用 lm-eval-harness 运行评测"""
        output_path = os.path.join(self.output_dir, f"{task}")

        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={self.model_path},trust_remote_code=True,dtype=bfloat16",
            "--tasks", task,
            "--batch_size", str(batch_size),
            "--output_path", output_path,
        ]

        if num_fewshot is not None:
            cmd.extend(["--num_fewshot", str(num_fewshot)])

        print(f"  运行命令: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600
            )
            if result.returncode == 0:
                # 读取结果
                results_file = os.path.join(output_path, "results.json")
                if os.path.exists(results_file):
                    with open(results_file) as f:
                        return json.load(f)
                return {"status": "completed", "output": result.stdout[-500:]}
            else:
                return {"status": "failed", "error": result.stderr[-500:]}
        except subprocess.TimeoutExpired:
            return {"status": "timeout"}
        except FileNotFoundError:
            return {"status": "lm_eval_not_found",
                    "note": "请安装: pip install lm-eval"}

    def run_suite(self, tasks: List[str], batch_size: int = 8) -> Dict:
        """运行一组基准测试"""
        results = {}

        for task_name in tasks:
            if task_name not in self.AVAILABLE_BENCHMARKS:
                print(f"  跳过未知任务: {task_name}")
                continue

            bench_info = self.AVAILABLE_BENCHMARKS[task_name]
            print(f"\n[{task_name}] {bench_info['desc']}")

            start_time = time.time()

            if bench_info["type"] == "lm_eval":
                result = self.run_lm_eval(bench_info["task"], batch_size)
            else:
                result = {"status": "unsupported_type"}

            elapsed = time.time() - start_time
            result["time_sec"] = elapsed
            results[task_name] = result

            # 提取分数
            score = self._extract_score(result, task_name)
            if score is not None:
                print(f"  分数: {score:.2%} (用时 {elapsed:.0f}s)")
            else:
                print(f"  状态: {result.get('status', 'unknown')} (用时 {elapsed:.0f}s)")

        return results

    def _extract_score(self, result: Dict, task_name: str) -> float:
        """从结果中提取分数"""
        if "results" in result:
            for task, metrics in result["results"].items():
                if "acc" in metrics:
                    return metrics["acc"]
                if "acc_norm" in metrics:
                    return metrics["acc_norm"]
                if "pass@1" in metrics:
                    return metrics["pass@1"]
        return None

    def generate_report(self, results: Dict) -> str:
        """生成评估报告"""
        report = []
        report.append("=" * 60)
        report.append("基准测试报告")
        report.append(f"模型: {self.model_path}")
        report.append("=" * 60)
        report.append("")
        report.append(f"{'任务':<15} {'描述':<20} {'分数':<10} {'用时':<8}")
        report.append("-" * 60)

        for task_name, result in results.items():
            desc = self.AVAILABLE_BENCHMARKS.get(task_name, {}).get("desc", "")
            score = self._extract_score(result, task_name)
            time_sec = result.get("time_sec", 0)

            score_str = f"{score:.2%}" if score is not None else result.get("status", "N/A")
            report.append(f"{task_name:<15} {desc:<20} {score_str:<10} {time_sec:<8.0f}s")

        report.append("")
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="多基准测试套件")
    parser.add_argument("--model", required=True, help="模型路径")
    parser.add_argument("--tasks", nargs="+", default=None, help="指定任务")
    parser.add_argument("--suite", choices=["quick", "chinese", "reasoning", "full"],
                       default=None, help="使用预设套件")
    parser.add_argument("--full", action="store_true", help="运行所有测试")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", default="./benchmark_results")
    args = parser.parse_args()

    # 确定要运行的任务
    if args.full:
        tasks = BenchmarkSuite.PRESET_SUITES["full"]
    elif args.suite:
        tasks = BenchmarkSuite.PRESET_SUITES[args.suite]
    elif args.tasks:
        tasks = args.tasks
    else:
        tasks = BenchmarkSuite.PRESET_SUITES["quick"]

    print("=" * 60)
    print("基准测试套件")
    print(f"模型: {args.model}")
    print(f"任务: {tasks}")
    print("=" * 60)

    suite = BenchmarkSuite(args.model, args.output_dir)
    results = suite.run_suite(tasks, args.batch_size)

    # 生成报告
    report = suite.generate_report(results)
    print("\n" + report)

    # 保存
    report_file = os.path.join(args.output_dir, "benchmark_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"结果已保存到: {report_file}")


if __name__ == "__main__":
    main()
