"""
数据加载器：支持多种数据格式
"""

import json
import os
from typing import List, Dict, Optional
from datasets import Dataset, load_dataset


class DataLoader:
    """统一数据加载器"""

    SUPPORTED_FORMATS = ["messages", "sharegpt", "alpaca", "preference"]

    def __init__(self, tokenizer, max_seq_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def load(self, data_path: str, data_format: str = "auto",
             split: Optional[str] = None) -> Dataset:
        """加载数据"""
        if data_format == "auto":
            data_format = self._detect_format(data_path)

        # 加载原始数据
        raw_data = self._load_file(data_path)

        # 转换为统一格式
        if data_format == "messages":
            processed = self._process_messages(raw_data)
        elif data_format == "sharegpt":
            processed = self._process_sharegpt(raw_data)
        elif data_format == "alpaca":
            processed = self._process_alpaca(raw_data)
        elif data_format == "preference":
            return Dataset.from_list(raw_data)
        else:
            raise ValueError(f"不支持的格式: {data_format}")

        dataset = Dataset.from_list(processed)

        if split:
            splits = dataset.train_test_split(test_size=float(split), seed=42)
            return splits

        return dataset

    def _load_file(self, path: str) -> List[Dict]:
        """加载文件"""
        data = []
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                data = loaded if isinstance(loaded, list) else [loaded]
        else:
            raise ValueError(f"不支持的文件格式: {path}")
        return data

    def _detect_format(self, path: str) -> str:
        """自动检测数据格式"""
        sample = self._load_file(path)[:5]
        if not sample:
            return "messages"

        first = sample[0]
        if "messages" in first:
            return "messages"
        elif "conversations" in first:
            return "sharegpt"
        elif "instruction" in first:
            return "alpaca"
        elif "chosen" in first and "rejected" in first:
            return "preference"
        return "messages"

    def _process_messages(self, data: List[Dict]) -> List[Dict]:
        """处理 messages 格式"""
        processed = []
        for item in data:
            text = self.tokenizer.apply_chat_template(
                item["messages"], tokenize=False, add_generation_prompt=False
            )
            processed.append({"text": text})
        return processed

    def _process_sharegpt(self, data: List[Dict]) -> List[Dict]:
        """处理 ShareGPT 格式"""
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        processed = []

        for item in data:
            messages = []
            for conv in item.get("conversations", []):
                role = role_map.get(conv.get("from", ""), conv.get("from", ""))
                messages.append({"role": role, "content": conv.get("value", "")})

            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            processed.append({"text": text})

        return processed

    def _process_alpaca(self, data: List[Dict]) -> List[Dict]:
        """处理 Alpaca 格式"""
        processed = []
        for item in data:
            messages = []
            if item.get("system"):
                messages.append({"role": "system", "content": item["system"]})

            user_content = item.get("instruction", "")
            if item.get("input"):
                user_content += f"\n{item['input']}"
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": item.get("output", "")})

            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            processed.append({"text": text})

        return processed
