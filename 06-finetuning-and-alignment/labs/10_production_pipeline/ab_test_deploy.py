"""
A/B 测试部署工具：对比两个模型版本的线上效果

用法:
    python ab_test_deploy.py --model_a ./model_v1 --model_b ./model_v2 --split 0.5
    python ab_test_deploy.py --analyze --results ab_results.jsonl
"""

import argparse
import json
import hashlib
import time
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class ABTestConfig:
    """A/B 测试配置"""
    experiment_name: str = "ab_test"
    model_a_path: str = ""
    model_b_path: str = ""
    traffic_split: float = 0.5  # A 的流量比例
    duration_hours: float = 24
    min_samples: int = 100


class ABTestManager:
    """A/B 测试管理器"""

    def __init__(self, config: ABTestConfig):
        self.config = config
        self.results_file = f"ab_results_{config.experiment_name}.jsonl"
        self.results = []

    def route_request(self, user_id: str) -> str:
        """确定性路由：同一用户始终路由到同一模型"""
        hash_input = f"{self.config.experiment_name}:{user_id}"
        hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        ratio = (hash_val % 10000) / 10000.0

        if ratio < self.config.traffic_split:
            return "model_a"
        else:
            return "model_b"

    def record_result(self, user_id: str, group: str, prompt: str,
                     response: str, score: Optional[float] = None,
                     latency_ms: Optional[float] = None):
        """记录一次请求结果"""
        entry = {
            "user_id": user_id,
            "group": group,
            "prompt": prompt[:200],
            "response": response[:500],
            "score": score,
            "latency_ms": latency_ms,
            "timestamp": datetime.now().isoformat(),
        }
        self.results.append(entry)

        # 追加写入文件
        with open(self.results_file, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def analyze(self) -> Dict:
        """分析 A/B 测试结果"""
        if not self.results:
            # 从文件加载
            if os.path.exists(self.results_file):
                with open(self.results_file, "r") as f:
                    self.results = [json.loads(line) for line in f if line.strip()]

        a_results = [r for r in self.results if r["group"] == "model_a"]
        b_results = [r for r in self.results if r["group"] == "model_b"]

        analysis = {
            "experiment": self.config.experiment_name,
            "total_requests": len(self.results),
            "model_a": self._compute_metrics(a_results),
            "model_b": self._compute_metrics(b_results),
        }

        # 统计检验
        if len(a_results) >= 30 and len(b_results) >= 30:
            analysis["significance"] = self._significance_test(a_results, b_results)

        return analysis

    def _compute_metrics(self, results: List[Dict]) -> Dict:
        """计算组的指标"""
        if not results:
            return {"count": 0}

        scores = [r["score"] for r in results if r["score"] is not None]
        latencies = [r["latency_ms"] for r in results if r["latency_ms"] is not None]

        metrics = {"count": len(results)}

        if scores:
            metrics["avg_score"] = sum(scores) / len(scores)
            metrics["score_std"] = (sum((s - metrics["avg_score"])**2 for s in scores) / len(scores)) ** 0.5

        if latencies:
            metrics["avg_latency_ms"] = sum(latencies) / len(latencies)
            sorted_lat = sorted(latencies)
            metrics["p50_latency_ms"] = sorted_lat[len(sorted_lat) // 2]
            metrics["p99_latency_ms"] = sorted_lat[int(len(sorted_lat) * 0.99)]

        # 回答长度
        lengths = [len(r["response"]) for r in results]
        metrics["avg_response_length"] = sum(lengths) / len(lengths)

        return metrics

    def _significance_test(self, a_results: List, b_results: List) -> Dict:
        """统计显著性检验"""
        a_scores = [r["score"] for r in a_results if r["score"] is not None]
        b_scores = [r["score"] for r in b_results if r["score"] is not None]

        if not a_scores or not b_scores:
            return {"test": "insufficient_data"}

        # 简单 t-test
        import math
        n_a, n_b = len(a_scores), len(b_scores)
        mean_a = sum(a_scores) / n_a
        mean_b = sum(b_scores) / n_b
        var_a = sum((x - mean_a)**2 for x in a_scores) / (n_a - 1) if n_a > 1 else 0
        var_b = sum((x - mean_b)**2 for x in b_scores) / (n_b - 1) if n_b > 1 else 0

        se = math.sqrt(var_a/n_a + var_b/n_b) if (var_a/n_a + var_b/n_b) > 0 else 1
        t_stat = (mean_a - mean_b) / se

        # 粗略判断显著性
        significant = abs(t_stat) > 1.96  # ~95% confidence

        return {
            "t_statistic": t_stat,
            "significant_95": significant,
            "mean_diff": mean_a - mean_b,
            "winner": "model_a" if mean_a > mean_b else "model_b" if mean_b > mean_a else "tie",
        }


def simulate_ab_test(config: ABTestConfig, num_requests: int = 200):
    """模拟 A/B 测试（Demo）"""
    import random
    random.seed(42)

    manager = ABTestManager(config)

    print(f"模拟 {num_requests} 次请求...")
    prompts = [
        "什么是机器学习？",
        "如何学好编程？",
        "解释量子计算",
        "推荐几本好书",
    ]

    for i in range(num_requests):
        user_id = f"user_{random.randint(1, 50)}"
        prompt = random.choice(prompts)
        group = manager.route_request(user_id)

        # 模拟不同模型的表现
        if group == "model_a":
            score = random.gauss(3.5, 0.8)  # A 平均 3.5
            latency = random.gauss(200, 50)
        else:
            score = random.gauss(4.0, 0.7)  # B 平均 4.0 (更好)
            latency = random.gauss(220, 60)

        score = max(1, min(5, score))
        latency = max(50, latency)

        manager.record_result(
            user_id=user_id,
            group=group,
            prompt=prompt,
            response=f"[{group}] 模拟回答...",
            score=score,
            latency_ms=latency,
        )

    # 分析结果
    analysis = manager.analyze()
    return analysis


def print_analysis(analysis: Dict):
    """打印分析结果"""
    print("\n" + "=" * 60)
    print(f"A/B 测试分析: {analysis['experiment']}")
    print("=" * 60)
    print(f"总请求数: {analysis['total_requests']}")

    for group in ["model_a", "model_b"]:
        metrics = analysis[group]
        print(f"\n{group}:")
        print(f"  请求数: {metrics['count']}")
        if "avg_score" in metrics:
            print(f"  平均评分: {metrics['avg_score']:.3f} (std: {metrics.get('score_std', 0):.3f})")
        if "avg_latency_ms" in metrics:
            print(f"  平均延迟: {metrics['avg_latency_ms']:.0f}ms "
                  f"(P50: {metrics.get('p50_latency_ms', 0):.0f}ms, "
                  f"P99: {metrics.get('p99_latency_ms', 0):.0f}ms)")
        if "avg_response_length" in metrics:
            print(f"  平均回答长度: {metrics['avg_response_length']:.0f} 字符")

    if "significance" in analysis:
        sig = analysis["significance"]
        print(f"\n统计检验:")
        print(f"  均值差异: {sig['mean_diff']:.3f}")
        print(f"  t统计量: {sig['t_statistic']:.3f}")
        print(f"  95%显著: {'是' if sig['significant_95'] else '否'}")
        print(f"  优胜: {sig['winner']}")


def main():
    parser = argparse.ArgumentParser(description="A/B 测试部署")
    parser.add_argument("--model_a", default="./model_v1")
    parser.add_argument("--model_b", default="./model_v2")
    parser.add_argument("--split", type=float, default=0.5)
    parser.add_argument("--name", default="ab_test_v1_vs_v2")
    parser.add_argument("--simulate", action="store_true", default=True)
    parser.add_argument("--num_requests", type=int, default=200)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--results", default=None)
    args = parser.parse_args()

    config = ABTestConfig(
        experiment_name=args.name,
        model_a_path=args.model_a,
        model_b_path=args.model_b,
        traffic_split=args.split,
    )

    if args.analyze and args.results:
        manager = ABTestManager(config)
        manager.results_file = args.results
        analysis = manager.analyze()
        print_analysis(analysis)
    elif args.simulate:
        analysis = simulate_ab_test(config, args.num_requests)
        print_analysis(analysis)

        # 保存分析
        with open(f"ab_analysis_{args.name}.json", "w") as f:
            json.dump(analysis, f, indent=2)


if __name__ == "__main__":
    main()
