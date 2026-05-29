"""
合成数据质量过滤器

对 Self-Instruct / Evol-Instruct 生成的数据进行多维度过滤：
1. 长度过滤
2. 重复检测
3. 格式检查
4. 内容质量评估
5. 多样性采样

用法：
    python quality_filter.py --input generated.jsonl --output filtered.jsonl --min-score 0.7
"""

import os
import json
import argparse
import hashlib
from typing import List, Dict, Set
from collections import Counter


class SyntheticDataFilter:
    """合成数据质量过滤器"""

    def __init__(self, min_instruction_len: int = 10,
                 max_instruction_len: int = 500,
                 min_response_len: int = 20,
                 max_response_len: int = 5000,
                 dedup_threshold: float = 0.8):
        self.min_instruction_len = min_instruction_len
        self.max_instruction_len = max_instruction_len
        self.min_response_len = min_response_len
        self.max_response_len = max_response_len
        self.dedup_threshold = dedup_threshold

        self.seen_instructions: Set[str] = set()
        self.stats = {
            "total": 0,
            "passed": 0,
            "filtered_length": 0,
            "filtered_format": 0,
            "filtered_quality": 0,
            "filtered_duplicate": 0,
        }

    def filter_length(self, item: Dict) -> bool:
        """长度过滤"""
        instruction = item.get("instruction", "")
        response = item.get("response", "")

        if len(instruction) < self.min_instruction_len:
            return False
        if len(instruction) > self.max_instruction_len:
            return False
        if len(response) < self.min_response_len:
            return False
        if len(response) > self.max_response_len:
            return False

        return True

    def filter_format(self, item: Dict) -> bool:
        """格式过滤"""
        instruction = item.get("instruction", "")
        response = item.get("response", "")

        # 检查指令是否包含不当标记
        bad_markers = [
            "as an ai", "i cannot", "i'm sorry",
            "作为一个 ai", "我无法", "抱歉",
            "#given prompt#", "#rewritten prompt#",
        ]
        combined = (instruction + response).lower()
        for marker in bad_markers:
            if marker in combined:
                return False

        # 检查是否只是重复原始 prompt 模板
        if "your objective is to" in combined:
            return False

        return True

    def filter_quality(self, item: Dict) -> float:
        """质量评分（返回 0-1 分数）"""
        instruction = item.get("instruction", "")
        response = item.get("response", "")

        score = 1.0

        # 指令质量
        # 是否是完整句子
        if not any(instruction.endswith(c) for c in ".?!。？！"):
            score *= 0.9

        # 回复质量
        words = response.split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                score *= 0.5

        # 回复是否有实质内容
        if len(response.split()) < 10:
            score *= 0.6

        # 指令和回复是否相关（简单检查：共享关键词）
        inst_words = set(instruction.lower().split())
        resp_words = set(response.lower().split())
        overlap = inst_words & resp_words
        if len(overlap) == 0 and len(inst_words) > 3:
            score *= 0.7

        return score

    def filter_duplicate(self, item: Dict) -> bool:
        """去重"""
        instruction = item.get("instruction", "").lower().strip()

        # 精确去重
        inst_hash = hashlib.md5(instruction.encode()).hexdigest()
        if inst_hash in self.seen_instructions:
            return False
        self.seen_instructions.add(inst_hash)

        return True

    def filter_batch(self, items: List[Dict],
                     min_quality_score: float = 0.7) -> List[Dict]:
        """批量过滤"""
        passed = []

        for item in items:
            self.stats["total"] += 1

            # 长度检查
            if not self.filter_length(item):
                self.stats["filtered_length"] += 1
                continue

            # 格式检查
            if not self.filter_format(item):
                self.stats["filtered_format"] += 1
                continue

            # 去重
            if not self.filter_duplicate(item):
                self.stats["filtered_duplicate"] += 1
                continue

            # 质量评分
            quality = self.filter_quality(item)
            if quality < min_quality_score:
                self.stats["filtered_quality"] += 1
                continue

            item["quality_score"] = quality
            passed.append(item)
            self.stats["passed"] += 1

        return passed

    def print_stats(self):
        """打印过滤统计"""
        total = self.stats["total"]
        if total == 0:
            print("未处理任何数据")
            return

        print(f"\n{'='*50}")
        print(f"过滤统计:")
        print(f"  总输入: {total}")
        print(f"  通过: {self.stats['passed']} "
              f"({self.stats['passed']/total*100:.1f}%)")
        print(f"  --- 过滤原因 ---")
        print(f"  长度不合格: {self.stats['filtered_length']}")
        print(f"  格式不合格: {self.stats['filtered_format']}")
        print(f"  重复: {self.stats['filtered_duplicate']}")
        print(f"  质量不足: {self.stats['filtered_quality']}")


def main():
    parser = argparse.ArgumentParser(description="合成数据质量过滤")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--min-score", type=float, default=0.7)
    args = parser.parse_args()

    # 加载数据
    items = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line.strip()))

    print(f"加载 {len(items)} 条数据")

    # 过滤
    filter_obj = SyntheticDataFilter()
    filtered = filter_obj.filter_batch(items, min_quality_score=args.min_score)
    filter_obj.print_stats()

    # 保存
    output_path = args.output or args.input.replace(".jsonl", "_filtered.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in filtered:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n过滤后数据保存到: {output_path} ({len(filtered)} 条)")


if __name__ == "__main__":
    main()
