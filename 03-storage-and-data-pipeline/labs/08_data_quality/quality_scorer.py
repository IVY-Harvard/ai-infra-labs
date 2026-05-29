"""
多维度数据质量打分器

评估维度：
1. 语言质量（流畅度、语法）
2. 信息密度（唯一词比例、内容丰富度）
3. 安全性（有害内容检测）
4. 格式质量（结构化程度）

用法：
    python quality_scorer.py --input-file data.jsonl --output scored_data.jsonl
    python quality_scorer.py --generate-demo --input-file /tmp/quality_demo.jsonl
"""

import os
import json
import math
import argparse
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from collections import Counter


@dataclass
class QualityScore:
    """质量评分"""
    language_quality: float    # [0,1] 语言质量
    info_density: float        # [0,1] 信息密度
    safety: float              # [0,1] 安全性
    format_quality: float      # [0,1] 格式质量
    overall: float             # [0,1] 综合分
    details: Dict = None


class QualityScorer:
    """多维度质量打分器"""

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "language": 0.35,
            "info_density": 0.25,
            "safety": 0.20,
            "format": 0.20,
        }

        # 有害内容关键词（简化版，实际用分类器）
        self.harmful_patterns = [
            r'\b(hack|exploit|attack)\b.*\b(tutorial|guide|how to)\b',
        ]

    def score(self, text: str) -> QualityScore:
        """对文本进行综合打分"""
        lang = self._score_language(text)
        info = self._score_info_density(text)
        safe = self._score_safety(text)
        fmt = self._score_format(text)

        overall = (
            self.weights["language"] * lang +
            self.weights["info_density"] * info +
            self.weights["safety"] * safe +
            self.weights["format"] * fmt
        )

        return QualityScore(
            language_quality=lang,
            info_density=info,
            safety=safe,
            format_quality=fmt,
            overall=overall,
        )

    def _score_language(self, text: str) -> float:
        """语言质量评分"""
        score = 1.0

        # 检查 1: 长度
        words = text.split()
        word_count = len(words)
        if word_count < 20:
            score *= 0.3
        elif word_count < 50:
            score *= 0.7

        # 检查 2: 特殊字符比例
        if text:
            special = sum(1 for c in text
                         if not c.isalnum() and not c.isspace()
                         and c not in ".,!?;:'\"()-")
            special_ratio = special / len(text)
            if special_ratio > 0.2:
                score *= 0.4
            elif special_ratio > 0.1:
                score *= 0.7

        # 检查 3: 全大写比例
        if text:
            upper_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if upper_ratio > 0.5:
                score *= 0.5

        # 检查 4: 重复行
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if lines:
            unique_lines = len(set(lines))
            line_unique_ratio = unique_lines / len(lines)
            score *= max(line_unique_ratio, 0.3)

        # 检查 5: 句子结构
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [s.strip() for s in sentences
                          if len(s.strip().split()) >= 3]
        if sentences:
            sentence_ratio = len(valid_sentences) / max(len(sentences), 1)
            score *= max(sentence_ratio, 0.5)

        return min(max(score, 0.0), 1.0)

    def _score_info_density(self, text: str) -> float:
        """信息密度评分"""
        words = text.lower().split()
        if not words:
            return 0.0

        # Type-Token Ratio（唯一词比例）
        ttr = len(set(words)) / len(words)

        # 平均词长
        avg_len = sum(len(w) for w in words) / len(words)
        len_score = min(avg_len / 5.5, 1.0)

        # 停用词比例（高停用词比例=低信息密度）
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be",
                     "been", "being", "have", "has", "had", "do", "does",
                     "did", "will", "would", "could", "should", "may",
                     "might", "shall", "can", "need", "dare", "to", "of",
                     "in", "for", "on", "with", "at", "by", "from", "it",
                     "this", "that", "and", "or", "but", "if", "not"}
        stopword_ratio = sum(1 for w in words if w in stopwords) / len(words)
        content_ratio = 1 - stopword_ratio

        # 综合
        score = ttr * 0.4 + len_score * 0.3 + content_ratio * 0.3
        return min(max(score, 0.0), 1.0)

    def _score_safety(self, text: str) -> float:
        """安全性评分（简化版）"""
        score = 1.0
        text_lower = text.lower()

        # 检测有害模式
        for pattern in self.harmful_patterns:
            if re.search(pattern, text_lower):
                score *= 0.3

        # 检测个人信息模式
        # 邮箱
        if re.search(r'\b[\w.-]+@[\w.-]+\.\w+\b', text):
            score *= 0.8
        # 电话号码
        if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text):
            score *= 0.8

        return score

    def _score_format(self, text: str) -> float:
        """格式质量评分"""
        score = 1.0

        # 是否有基本结构（段落）
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) == 0:
            score *= 0.5
        elif len(paragraphs) == 1 and len(text) > 1000:
            score *= 0.7  # 长文本没有段落分隔

        # 是否有乱码
        if text:
            printable_ratio = sum(
                1 for c in text if c.isprintable() or c in "\n\t"
            ) / len(text)
            score *= printable_ratio

        return min(max(score, 0.0), 1.0)

    def score_batch(self, texts: List[Dict]) -> List[Dict]:
        """批量打分"""
        results = []
        for item in texts:
            text = item.get("text", "")
            score = self.score(text)
            result = {**item, "quality_score": asdict(score)}
            results.append(result)
        return results


def generate_demo_data(output_path: str, num_samples: int = 100):
    """生成质量参差不齐的演示数据"""
    import numpy as np

    samples = []

    # 高质量样本
    for i in range(num_samples // 3):
        text = (
            f"Machine learning has transformed modern computing. "
            f"Deep neural networks can now process complex patterns "
            f"in data with remarkable accuracy. The key insight is that "
            f"hierarchical representations learned through backpropagation "
            f"capture increasingly abstract features at each layer. "
            f"This enables applications from computer vision to natural "
            f"language understanding. Research continues to push boundaries "
            f"in areas like few-shot learning and model efficiency."
        )
        samples.append({"id": f"high_{i}", "text": text, "expected": "high"})

    # 中等质量
    for i in range(num_samples // 3):
        text = f"this is sample {i} with some content " * 10
        samples.append({"id": f"mid_{i}", "text": text, "expected": "medium"})

    # 低质量
    for i in range(num_samples // 3):
        text = "a " * 50 + "!!!" * 20
        samples.append({"id": f"low_{i}", "text": text, "expected": "low"})

    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"生成 {len(samples)} 个演示样本到 {output_path}")


def main():
    parser = argparse.ArgumentParser(description="数据质量打分")
    parser.add_argument("--input-file", type=str,
                       default="/tmp/quality_demo.jsonl")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--generate-demo", action="store_true")
    args = parser.parse_args()

    if args.generate_demo:
        generate_demo_data(args.input_file)

    # 加载数据
    texts = []
    with open(args.input_file, "r") as f:
        for line in f:
            texts.append(json.loads(line.strip()))

    # 打分
    scorer = QualityScorer()
    results = scorer.score_batch(texts)

    # 统计
    scores = [r["quality_score"]["overall"] for r in results]
    print(f"\n{'='*50}")
    print(f"质量打分结果 ({len(results)} 篇文档)")
    print(f"{'='*50}")
    print(f"  平均分: {sum(scores)/len(scores):.3f}")
    print(f"  最高分: {max(scores):.3f}")
    print(f"  最低分: {min(scores):.3f}")

    # 分档统计
    bins = [(0.8, 1.0, "优秀"), (0.6, 0.8, "良好"),
            (0.4, 0.6, "一般"), (0.0, 0.4, "较差")]
    for low, high, label in bins:
        count = sum(1 for s in scores if low <= s < high)
        print(f"  {label} [{low:.1f}, {high:.1f}): {count} "
              f"({count/len(scores)*100:.1f}%)")

    # 保存结果
    output_path = args.output or args.input_file.replace(".jsonl", "_scored.jsonl")
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n结果保存到: {output_path}")


if __name__ == "__main__":
    main()
