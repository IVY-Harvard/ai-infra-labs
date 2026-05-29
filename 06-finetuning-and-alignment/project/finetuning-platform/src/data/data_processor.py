"""
数据预处理模块
"""

import re
from typing import List, Dict, Optional
from datasets import Dataset


class DataProcessor:
    """数据预处理器"""

    def __init__(self, tokenizer, max_seq_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def process(self, dataset: Dataset, remove_long: bool = True) -> Dataset:
        """完整的预处理流水线"""
        # 1. 文本规范化
        dataset = dataset.map(self._normalize, desc="Normalizing")

        # 2. 长度过滤
        if remove_long:
            dataset = dataset.filter(
                lambda x: len(self.tokenizer.encode(x["text"])) <= self.max_seq_length,
                desc="Filtering long sequences"
            )

        # 3. 去重
        dataset = self._deduplicate(dataset)

        return dataset

    def _normalize(self, example: Dict) -> Dict:
        """文本规范化"""
        text = example.get("text", "")
        # 统一空白符
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        # 去除多余空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 统一引号
        text = text.replace('“', '"').replace('”', '"')
        text = text.replace('‘', "'").replace('’', "'")
        return {"text": text}

    def _deduplicate(self, dataset: Dataset) -> Dataset:
        """基于内容去重"""
        seen = set()
        unique_indices = []

        for i, item in enumerate(dataset):
            text_hash = hash(item["text"][:500])  # 用前 500 字符做 hash
            if text_hash not in seen:
                seen.add(text_hash)
                unique_indices.append(i)

        removed = len(dataset) - len(unique_indices)
        if removed > 0:
            print(f"  去重: 移除 {removed} 条重复数据")

        return dataset.select(unique_indices)

    def compute_stats(self, dataset: Dataset) -> Dict:
        """计算数据集统计信息"""
        lengths = []
        for item in dataset:
            tokens = self.tokenizer.encode(item["text"])
            lengths.append(len(tokens))

        lengths.sort()
        return {
            "count": len(lengths),
            "avg_tokens": sum(lengths) / max(len(lengths), 1),
            "min_tokens": lengths[0] if lengths else 0,
            "max_tokens": lengths[-1] if lengths else 0,
            "median_tokens": lengths[len(lengths)//2] if lengths else 0,
            "p90_tokens": lengths[int(len(lengths)*0.9)] if lengths else 0,
            "total_tokens": sum(lengths),
        }
