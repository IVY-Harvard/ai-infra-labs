"""
数据集画像工具

统计数据集的分布特征：
- 文本长度分布
- 语言分布
- 话题分布
- 质量分布
- 时间分布

用法：
    python data_profiler.py --input-dir /path/to/data --output profile_report.json
"""

import os
import json
import argparse
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List
from dataclasses import dataclass, asdict


@dataclass
class DataProfile:
    """数据集画像"""
    total_documents: int
    total_characters: int
    total_words: int
    avg_doc_length_chars: float
    avg_doc_length_words: float
    length_distribution: Dict[str, int]
    language_distribution: Dict[str, int]
    format_distribution: Dict[str, int]
    quality_distribution: Dict[str, int]


class DataProfiler:
    """数据集画像工具"""

    def __init__(self):
        self.doc_lengths: List[int] = []
        self.word_counts: List[int] = []
        self.languages: Counter = Counter()
        self.formats: Counter = Counter()
        self.total_chars = 0
        self.total_words = 0
        self.total_docs = 0

    def profile_text(self, text: str, metadata: Dict = None):
        """分析单篇文档"""
        self.total_docs += 1

        # 长度统计
        char_len = len(text)
        word_len = len(text.split())
        self.doc_lengths.append(char_len)
        self.word_counts.append(word_len)
        self.total_chars += char_len
        self.total_words += word_len

        # 语言检测（简化版）
        lang = self._detect_language(text)
        self.languages[lang] += 1

        # 格式检测
        fmt = self._detect_format(text)
        self.formats[fmt] += 1

    def _detect_language(self, text: str) -> str:
        """简易语言检测"""
        # 中文字符比例
        chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
        if chinese_chars / max(len(text), 1) > 0.3:
            return "zh"

        # 日文
        japanese_chars = sum(1 for c in text
                           if '぀' <= c <= 'ゟ' or
                           '゠' <= c <= 'ヿ')
        if japanese_chars / max(len(text), 1) > 0.1:
            return "ja"

        return "en"

    def _detect_format(self, text: str) -> str:
        """检测文本格式"""
        if text.strip().startswith("{") or text.strip().startswith("["):
            return "json"
        if re.search(r'^#+\s', text, re.MULTILINE):
            return "markdown"
        if re.search(r'<html|<div|<p>', text, re.IGNORECASE):
            return "html"
        if re.search(r'^(def |class |import |from )', text, re.MULTILINE):
            return "code"
        return "plain_text"

    def profile_directory(self, input_dir: str) -> DataProfile:
        """分析整个目录"""
        input_path = Path(input_dir)

        for filepath in sorted(input_path.rglob("*")):
            if filepath.suffix == ".jsonl":
                with open(filepath, "r", encoding="utf-8",
                         errors="ignore") as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            text = data.get("text", "")
                            if text:
                                self.profile_text(text, data)
                        except json.JSONDecodeError:
                            continue

            elif filepath.suffix == ".txt":
                with open(filepath, "r", encoding="utf-8",
                         errors="ignore") as f:
                    text = f.read()
                    if text:
                        self.profile_text(text)

        return self._build_profile()

    def _build_profile(self) -> DataProfile:
        """构建画像结果"""
        # 长度分布分档
        length_bins = {
            "<100": 0, "100-500": 0, "500-1K": 0,
            "1K-5K": 0, "5K-10K": 0, ">10K": 0,
        }
        for length in self.doc_lengths:
            if length < 100:
                length_bins["<100"] += 1
            elif length < 500:
                length_bins["100-500"] += 1
            elif length < 1000:
                length_bins["500-1K"] += 1
            elif length < 5000:
                length_bins["1K-5K"] += 1
            elif length < 10000:
                length_bins["5K-10K"] += 1
            else:
                length_bins[">10K"] += 1

        return DataProfile(
            total_documents=self.total_docs,
            total_characters=self.total_chars,
            total_words=self.total_words,
            avg_doc_length_chars=(self.total_chars / self.total_docs
                                  if self.total_docs > 0 else 0),
            avg_doc_length_words=(self.total_words / self.total_docs
                                  if self.total_docs > 0 else 0),
            length_distribution=length_bins,
            language_distribution=dict(self.languages.most_common()),
            format_distribution=dict(self.formats.most_common()),
            quality_distribution={},
        )

    def print_report(self, profile: DataProfile):
        """打印画像报告"""
        print(f"\n{'='*60}")
        print(f"数据集画像报告")
        print(f"{'='*60}")

        print(f"\n--- 基本统计 ---")
        print(f"  总文档数: {profile.total_documents:,}")
        print(f"  总字符数: {profile.total_characters:,}")
        print(f"  总词数: {profile.total_words:,}")
        print(f"  平均文档长度: {profile.avg_doc_length_chars:.0f} 字符, "
              f"{profile.avg_doc_length_words:.0f} 词")

        print(f"\n--- 长度分布 ---")
        for bucket, count in profile.length_distribution.items():
            pct = count / max(profile.total_documents, 1) * 100
            bar = "█" * int(pct / 2)
            print(f"  {bucket:>10}: {count:>6} ({pct:>5.1f}%) {bar}")

        print(f"\n--- 语言分布 ---")
        for lang, count in profile.language_distribution.items():
            pct = count / max(profile.total_documents, 1) * 100
            print(f"  {lang}: {count:>6} ({pct:.1f}%)")

        print(f"\n--- 格式分布 ---")
        for fmt, count in profile.format_distribution.items():
            pct = count / max(profile.total_documents, 1) * 100
            print(f"  {fmt}: {count:>6} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="数据集画像")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="profile_report.json")
    args = parser.parse_args()

    profiler = DataProfiler()
    profile = profiler.profile_directory(args.input_dir)
    profiler.print_report(profile)

    # 保存
    with open(args.output, "w") as f:
        json.dump(asdict(profile), f, indent=2, ensure_ascii=False)
    print(f"\n报告保存到: {args.output}")


if __name__ == "__main__":
    main()
