"""
数据质量过滤器
"""

import re
from typing import List, Dict, Callable
from datasets import Dataset


class QualityFilter:
    """数据质量过滤器"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.min_length = self.config.get("min_length", 10)
        self.max_length = self.config.get("max_length", 10000)
        self.filters: List[Callable] = []
        self._setup_default_filters()

    def _setup_default_filters(self):
        """设置默认过滤规则"""
        self.filters = [
            self._filter_empty,
            self._filter_length,
            self._filter_special_chars,
            self._filter_repetition,
        ]

    def filter(self, dataset: Dataset) -> Dataset:
        """应用所有过滤器"""
        original_size = len(dataset)

        for filter_fn in self.filters:
            dataset = dataset.filter(filter_fn)

        removed = original_size - len(dataset)
        print(f"  质量过滤: {original_size} → {len(dataset)} (移除 {removed})")
        return dataset

    def _filter_empty(self, example: Dict) -> bool:
        """过滤空内容"""
        text = example.get("text", "")
        return len(text.strip()) > 0

    def _filter_length(self, example: Dict) -> bool:
        """长度过滤"""
        text = example.get("text", "")
        length = len(text)
        return self.min_length <= length <= self.max_length

    def _filter_special_chars(self, example: Dict) -> bool:
        """过滤含有过多特殊字符的样本"""
        text = example.get("text", "")
        # 控制字符
        if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', text):
            return False
        # 过多连续特殊符号
        if re.search(r'[^\w\s]{20,}', text):
            return False
        return True

    def _filter_repetition(self, example: Dict) -> bool:
        """过滤高重复率文本"""
        text = example.get("text", "")
        if len(text) < 100:
            return True

        # 检查连续重复 n-gram
        words = text.split()
        if len(words) < 20:
            return True

        # 5-gram 重复率
        ngrams = [tuple(words[i:i+5]) for i in range(len(words)-4)]
        unique_ratio = len(set(ngrams)) / max(len(ngrams), 1)

        return unique_ratio > 0.3  # 重复率 < 70%

    def add_filter(self, filter_fn: Callable):
        """添加自定义过滤器"""
        self.filters.append(filter_fn)
