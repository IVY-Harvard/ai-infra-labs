"""
分层标注流水线

实现三层标注：
- Tier 1 (自动): 规则 + 模型自动标注
- Tier 2 (众包): 中等难度任务分发给众包标注
- Tier 3 (专家): 高难度任务由专家处理

用法：
    python annotation_pipeline.py --input feedback.jsonl --output annotated.jsonl
"""

import os
import json
import time
import argparse
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class AnnotationTier(str, Enum):
    AUTO = "auto"
    CROWD = "crowd"
    EXPERT = "expert"


class AnnotationLabel(str, Enum):
    GOOD = "good"
    BAD = "bad"
    NEEDS_IMPROVEMENT = "needs_improvement"
    HARMFUL = "harmful"
    UNSURE = "unsure"


@dataclass
class AnnotationResult:
    """标注结果"""
    sample_id: str
    tier: str
    label: str
    confidence: float
    feedback: str = ""
    annotator: str = "auto"
    duration_s: float = 0


class AnnotationPipeline:
    """分层标注流水线"""

    def __init__(self, auto_confidence_threshold: float = 0.8):
        self.auto_threshold = auto_confidence_threshold
        self.stats = {
            "total": 0,
            "auto_passed": 0,
            "escalated_crowd": 0,
            "escalated_expert": 0,
        }

    def auto_annotate(self, instruction: str,
                      response: str) -> Optional[AnnotationResult]:
        """Tier 1: 自动标注"""
        confidence = 0.0
        label = AnnotationLabel.UNSURE

        # 规则 1: 回复过短 → 低质量
        if len(response.split()) < 10:
            return AnnotationResult(
                sample_id="", tier=AnnotationTier.AUTO.value,
                label=AnnotationLabel.BAD.value,
                confidence=0.9, feedback="回复过短",
            )

        # 规则 2: 拒绝回答 → 标记为需要改进
        refusal_markers = ["i cannot", "i'm sorry", "as an ai",
                          "我无法", "抱歉", "作为ai"]
        if any(m in response.lower() for m in refusal_markers):
            return AnnotationResult(
                sample_id="", tier=AnnotationTier.AUTO.value,
                label=AnnotationLabel.NEEDS_IMPROVEMENT.value,
                confidence=0.85, feedback="包含拒绝标记",
            )

        # 规则 3: 长度和内容合理 → 可能是好的
        words = response.split()
        unique_ratio = len(set(words)) / max(len(words), 1)

        if len(words) >= 20 and unique_ratio > 0.4:
            confidence = 0.7 + min(unique_ratio * 0.3, 0.25)
            label = AnnotationLabel.GOOD
        else:
            confidence = 0.5
            label = AnnotationLabel.NEEDS_IMPROVEMENT

        if confidence >= self.auto_threshold:
            return AnnotationResult(
                sample_id="", tier=AnnotationTier.AUTO.value,
                label=label.value, confidence=confidence,
                feedback="自动规则判定",
            )

        return None  # 信心不够，需要升级

    def crowd_annotate(self, instruction: str,
                       response: str) -> AnnotationResult:
        """Tier 2: 模拟众包标注"""
        # 实际中这里对接众包平台 API
        # 此处模拟
        import random
        time.sleep(0.01)  # 模拟延迟

        # 基于简单启发式模拟标注结果
        words = response.split()
        if len(words) > 30 and len(set(words)) / len(words) > 0.5:
            label = AnnotationLabel.GOOD
            confidence = 0.85
        else:
            label = AnnotationLabel.NEEDS_IMPROVEMENT
            confidence = 0.75

        return AnnotationResult(
            sample_id="", tier=AnnotationTier.CROWD.value,
            label=label.value, confidence=confidence,
            feedback="众包标注",
            annotator=f"crowd_{random.randint(1,100):03d}",
        )

    def process_batch(self, samples: List[Dict]) -> List[Dict]:
        """处理一批样本"""
        results = []

        for sample in samples:
            self.stats["total"] += 1
            instruction = sample.get("instruction", "")
            response = sample.get("chosen", sample.get("model_output", ""))
            sample_id = sample.get("feedback_id",
                                   str(self.stats["total"]))

            # Tier 1: 自动标注
            auto_result = self.auto_annotate(instruction, response)

            if auto_result and auto_result.confidence >= self.auto_threshold:
                auto_result.sample_id = sample_id
                self.stats["auto_passed"] += 1
                annotated = {**sample, "annotation": asdict(auto_result)}
            else:
                # Tier 2: 升级到众包
                crowd_result = self.crowd_annotate(instruction, response)
                crowd_result.sample_id = sample_id
                self.stats["escalated_crowd"] += 1
                annotated = {**sample, "annotation": asdict(crowd_result)}

            results.append(annotated)

        return results

    def print_stats(self):
        """打印统计"""
        total = self.stats["total"]
        if total == 0:
            return

        print(f"\n{'='*50}")
        print(f"标注流水线统计:")
        print(f"  总样本: {total}")
        print(f"  自动标注: {self.stats['auto_passed']} "
              f"({self.stats['auto_passed']/total*100:.1f}%)")
        print(f"  众包标注: {self.stats['escalated_crowd']} "
              f"({self.stats['escalated_crowd']/total*100:.1f}%)")
        print(f"  专家标注: {self.stats['escalated_expert']} "
              f"({self.stats['escalated_expert']/total*100:.1f}%)")

        auto_cost = 0
        crowd_cost = self.stats["escalated_crowd"] * 0.5
        expert_cost = self.stats["escalated_expert"] * 10
        total_cost = auto_cost + crowd_cost + expert_cost
        avg_cost = total_cost / total if total > 0 else 0

        print(f"\n  成本估算:")
        print(f"    自动: $0")
        print(f"    众包: ${crowd_cost:.2f} ($0.5/样本)")
        print(f"    专家: ${expert_cost:.2f} ($10/样本)")
        print(f"    总计: ${total_cost:.2f} (平均 ${avg_cost:.2f}/样本)")


def main():
    parser = argparse.ArgumentParser(description="标注流水线")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--auto-threshold", type=float, default=0.8)
    args = parser.parse_args()

    # 加载数据
    samples = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    print(f"加载 {len(samples)} 条待标注数据")

    # 标注
    pipeline = AnnotationPipeline(
        auto_confidence_threshold=args.auto_threshold
    )
    annotated = pipeline.process_batch(samples)
    pipeline.print_stats()

    # 保存
    output_path = args.output or args.input.replace(".jsonl", "_annotated.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in annotated:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n标注结果保存到: {output_path}")


if __name__ == "__main__":
    main()
