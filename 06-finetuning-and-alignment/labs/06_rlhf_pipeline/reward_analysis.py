"""
Reward Model 分析工具
分析 RM 的打分分布、准确率、校准情况

用法:
    python reward_analysis.py --model ./output/reward_model --data test_data.jsonl
"""

import argparse
import json
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict


class RewardAnalyzer:
    """Reward Model 分析器"""

    def __init__(self, model_path: str):
        print(f"加载 RM: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def score(self, text: str) -> float:
        """对文本打分"""
        inputs = self.tokenizer(
            text, return_tensors="pt",
            max_length=1024, truncation=True,
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits[0, 0].item()

    def score_pair(self, prompt: str, chosen: str, rejected: str) -> Dict:
        """对偏好对打分"""
        chosen_score = self.score(f"{prompt}\n{chosen}")
        rejected_score = self.score(f"{prompt}\n{rejected}")
        return {
            "chosen_score": chosen_score,
            "rejected_score": rejected_score,
            "margin": chosen_score - rejected_score,
            "correct": chosen_score > rejected_score,
        }

    def analyze_dataset(self, data: List[Dict]) -> Dict:
        """分析整个数据集"""
        results = []
        chosen_scores = []
        rejected_scores = []

        for i, item in enumerate(data):
            result = self.score_pair(
                item["prompt"], item["chosen"], item["rejected"]
            )
            results.append(result)
            chosen_scores.append(result["chosen_score"])
            rejected_scores.append(result["rejected_score"])

            if (i + 1) % 20 == 0:
                print(f"  已分析 {i+1}/{len(data)} 对")

        # 统计
        accuracy = sum(1 for r in results if r["correct"]) / len(results)
        margins = [r["margin"] for r in results]

        report = {
            "accuracy": accuracy,
            "total_pairs": len(results),
            "correct_pairs": sum(1 for r in results if r["correct"]),

            "chosen_scores": {
                "mean": np.mean(chosen_scores),
                "std": np.std(chosen_scores),
                "min": np.min(chosen_scores),
                "max": np.max(chosen_scores),
            },
            "rejected_scores": {
                "mean": np.mean(rejected_scores),
                "std": np.std(rejected_scores),
                "min": np.min(rejected_scores),
                "max": np.max(rejected_scores),
            },
            "margin_stats": {
                "mean": np.mean(margins),
                "std": np.std(margins),
                "min": np.min(margins),
                "max": np.max(margins),
                "median": np.median(margins),
            },

            # 按 margin 分桶的准确率
            "accuracy_by_margin": self._accuracy_by_margin(results),
        }

        return report

    def _accuracy_by_margin(self, results: List[Dict]) -> Dict:
        """按 margin 大小分桶统计准确率"""
        # 按 |margin| 分组
        bins = {"margin<0.5": [], "0.5-1.0": [], "1.0-2.0": [], "margin>2.0": []}
        for r in results:
            m = abs(r["margin"])
            if m < 0.5:
                bins["margin<0.5"].append(r["correct"])
            elif m < 1.0:
                bins["0.5-1.0"].append(r["correct"])
            elif m < 2.0:
                bins["1.0-2.0"].append(r["correct"])
            else:
                bins["margin>2.0"].append(r["correct"])

        return {k: sum(v)/max(len(v),1) for k, v in bins.items()}

    def score_responses(self, prompt: str, responses: List[str]) -> List[Dict]:
        """对同一 prompt 的多个回答打分并排序"""
        scored = []
        for resp in responses:
            score = self.score(f"{prompt}\n{resp}")
            scored.append({"response": resp[:100], "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored


def create_demo_test_data():
    """创建测试数据"""
    return [
        {
            "prompt": "什么是Python？",
            "chosen": "Python是一种高级编程语言，由Guido van Rossum于1991年创建。它以简洁的语法和强大的生态系统著称，广泛应用于Web开发、数据分析、AI等领域。",
            "rejected": "Python是一种语言。"
        },
        {
            "prompt": "如何学习编程？",
            "chosen": "学习编程建议：1)选择一门语言入门（推荐Python）2)学习基础语法 3)做项目实践 4)阅读优秀代码 5)持续学习新技术",
            "rejected": "去网上搜一下吧。"
        },
    ] * 20  # 扩展为 40 对


def main():
    parser = argparse.ArgumentParser(description="Reward Model 分析")
    parser.add_argument("--model", default="./output/reward_model", help="RM 路径")
    parser.add_argument("--data", default=None, help="测试数据文件")
    parser.add_argument("--output", default="reward_analysis_report.json")
    args = parser.parse_args()

    # 加载 RM
    analyzer = RewardAnalyzer(args.model)

    # 加载测试数据
    if args.data:
        data = []
        with open(args.data, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        print("使用内置测试数据...")
        data = create_demo_test_data()

    # 分析
    print(f"\n分析 {len(data)} 对偏好数据...")
    report = analyzer.analyze_dataset(data)

    # 打印报告
    print("\n" + "=" * 60)
    print("Reward Model 分析报告")
    print("=" * 60)
    print(f"\n准确率: {report['accuracy']:.2%}")
    print(f"  正确: {report['correct_pairs']}/{report['total_pairs']}")

    print(f"\nChosen 分数分布:")
    cs = report["chosen_scores"]
    print(f"  均值: {cs['mean']:.3f}, 标准差: {cs['std']:.3f}")
    print(f"  范围: [{cs['min']:.3f}, {cs['max']:.3f}]")

    print(f"\nRejected 分数分布:")
    rs = report["rejected_scores"]
    print(f"  均值: {rs['mean']:.3f}, 标准差: {rs['std']:.3f}")
    print(f"  范围: [{rs['min']:.3f}, {rs['max']:.3f}]")

    print(f"\nMargin 统计:")
    ms = report["margin_stats"]
    print(f"  均值: {ms['mean']:.3f}, 中位数: {ms['median']:.3f}")
    print(f"  标准差: {ms['std']:.3f}")

    print(f"\n按 Margin 分桶准确率:")
    for bucket, acc in report["accuracy_by_margin"].items():
        print(f"  {bucket}: {acc:.2%}")

    # 保存报告
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\n报告已保存到: {args.output}")


if __name__ == "__main__":
    main()
