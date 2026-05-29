"""
Token 统计分析工具

分析 tokenized 数据的分布特征：
- 序列长度分布
- Token 频率分布
- 词汇覆盖率
- 数据集大小估算

用法：
    python token_statistics.py --tokenized-dir /tmp/tok_demo/tokenized
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict


class TokenStatistics:
    """Token 统计分析"""

    def __init__(self, tokenizer_name: str = "gpt2"):
        self.tokenizer_name = tokenizer_name
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.vocab_size = self.tokenizer.vocab_size
        except ImportError:
            self.tokenizer = None
            self.vocab_size = 50000

    def analyze_binary(self, bin_path: str, max_length: int = 2048) -> Dict:
        """分析二进制 token 文件"""
        data = np.fromfile(bin_path, dtype=np.int32)
        total_tokens = len(data)

        # 按固定长度切分为序列
        num_sequences = total_tokens // max_length
        sequences = data[:num_sequences * max_length].reshape(-1, max_length)

        print(f"文件: {bin_path}")
        print(f"总 token: {total_tokens:,}")
        print(f"序列数: {num_sequences}")
        print(f"序列长度: {max_length}")
        print()

        # Token 频率分析
        print("--- Token 频率分析 ---")
        token_counts = Counter(data.tolist())
        unique_tokens = len(token_counts)
        coverage = unique_tokens / self.vocab_size * 100

        print(f"  唯一 token 数: {unique_tokens:,}")
        print(f"  词汇表覆盖: {coverage:.1f}% "
              f"({unique_tokens}/{self.vocab_size})")

        # Top tokens
        top_20 = token_counts.most_common(20)
        print(f"\n  Top 20 频率最高的 token:")
        for token_id, count in top_20:
            pct = count / total_tokens * 100
            token_str = ""
            if self.tokenizer:
                try:
                    token_str = repr(self.tokenizer.decode([token_id]))
                except Exception:
                    token_str = f"id={token_id}"
            print(f"    {token_id:>6}: {count:>10,} ({pct:>5.2f}%) {token_str}")

        # 序列级统计
        print(f"\n--- 序列级统计 ---")

        # 每个序列的唯一 token 比例
        unique_ratios = []
        for seq in sequences[:1000]:  # 采样前 1000 个序列
            unique_ratio = len(set(seq)) / len(seq)
            unique_ratios.append(unique_ratio)

        unique_ratios = np.array(unique_ratios)
        print(f"  唯一 token 比例:")
        print(f"    均值: {unique_ratios.mean():.3f}")
        print(f"    中位数: {np.median(unique_ratios):.3f}")
        print(f"    标准差: {unique_ratios.std():.3f}")

        # 数据集大小估算
        print(f"\n--- 数据集大小估算 ---")
        file_size = os.path.getsize(bin_path)
        print(f"  文件大小: {file_size/1024/1024:.1f}MB")
        print(f"  Token/MB: {total_tokens / (file_size/1024/1024):.0f}")

        # 估算训练 epoch 数
        typical_training_tokens = {
            "7B": 2e12,
            "13B": 2e12,
            "70B": 2e12,
        }
        print(f"\n  训练所需估算:")
        for model, needed in typical_training_tokens.items():
            epochs = needed / total_tokens
            print(f"    {model} 模型: 需要 {epochs:.0f} epochs "
                  f"(或 {needed/1e9:.0f}B tokens)")

        return {
            "total_tokens": total_tokens,
            "unique_tokens": unique_tokens,
            "vocab_coverage": coverage,
            "num_sequences": num_sequences,
            "unique_ratio_mean": float(unique_ratios.mean()),
        }

    def analyze_directory(self, tokenized_dir: str) -> Dict:
        """分析整个 tokenized 目录"""
        dir_path = Path(tokenized_dir)

        # 查找 metadata
        meta_path = dir_path / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"元数据:")
            for k, v in meta.items():
                print(f"  {k}: {v}")
            print()
            max_length = meta.get("max_length", 2048)
        else:
            max_length = 2048

        # 分析 bin 文件
        bin_files = list(dir_path.glob("*.bin"))
        if not bin_files:
            print("未找到 .bin 文件")
            return {}

        results = {}
        for bin_file in bin_files:
            print(f"\n{'='*50}")
            result = self.analyze_binary(str(bin_file), max_length)
            results[str(bin_file)] = result

        return results


def main():
    parser = argparse.ArgumentParser(description="Token 统计分析")
    parser.add_argument("--tokenized-dir", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    args = parser.parse_args()

    stats = TokenStatistics(tokenizer_name=args.tokenizer)
    stats.analyze_directory(args.tokenized_dir)


if __name__ == "__main__":
    main()
