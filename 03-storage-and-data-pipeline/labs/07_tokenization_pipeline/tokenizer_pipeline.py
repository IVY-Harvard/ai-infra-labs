"""
Tokenization 流水线

完整流程：原始文本 → 清洗 → Tokenization → 分组打包 → 输出

用法：
    python tokenizer_pipeline.py --input-dir /path/to/texts --output-dir /tmp/tokenized
    python tokenizer_pipeline.py --generate-demo  # 生成演示数据
"""

import os
import time
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Iterator, Dict
from dataclasses import dataclass


@dataclass
class PipelineStats:
    """流水线统计"""
    total_files: int = 0
    total_docs: int = 0
    total_tokens: int = 0
    filtered_docs: int = 0
    processing_time_s: float = 0
    tokens_per_second: float = 0


class TokenizerPipeline:
    """Tokenization 流水线

    处理步骤：
    1. 读取原始文本文件（txt/jsonl）
    2. 文本清洗（去除控制字符、过短过长过滤）
    3. Tokenization（使用 HuggingFace tokenizers）
    4. 分组打包（将短文本拼接成固定长度序列）
    5. 输出为训练格式（numpy memmap / jsonl）
    """

    def __init__(self, tokenizer_name: str = "gpt2",
                 max_length: int = 2048, min_text_length: int = 50):
        self.max_length = max_length
        self.min_text_length = min_text_length
        self.stats = PipelineStats()

        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except ImportError:
            print("警告: transformers 未安装，使用简易 tokenizer")
            self.tokenizer = None

    def read_texts(self, input_dir: str) -> Iterator[str]:
        """读取原始文本"""
        input_path = Path(input_dir)

        for filepath in sorted(input_path.rglob("*")):
            if filepath.suffix == ".txt":
                with open(filepath, "r", encoding="utf-8",
                         errors="ignore") as f:
                    yield f.read()
                self.stats.total_files += 1

            elif filepath.suffix == ".jsonl":
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            text = data.get("text", "")
                            if text:
                                yield text
                        except json.JSONDecodeError:
                            continue
                self.stats.total_files += 1

    def clean_text(self, text: str) -> str:
        """文本清洗"""
        # 去除控制字符（保留换行和空格）
        text = "".join(
            c for c in text
            if c.isprintable() or c in ("\n", "\t", " ")
        )

        # 合并多余空行
        lines = text.split("\n")
        cleaned_lines = []
        prev_empty = False
        for line in lines:
            is_empty = not line.strip()
            if is_empty and prev_empty:
                continue
            cleaned_lines.append(line)
            prev_empty = is_empty

        return "\n".join(cleaned_lines).strip()

    def filter_text(self, text: str) -> bool:
        """文本质量过滤，返回 True 表示保留"""
        if len(text) < self.min_text_length:
            return False

        # 过滤高重复率内容
        words = text.split()
        if len(words) < 10:
            return False

        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.2:
            return False

        return True

    def tokenize(self, text: str) -> List[int]:
        """Tokenization"""
        if self.tokenizer:
            encoded = self.tokenizer(
                text,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )
            return encoded["input_ids"]
        else:
            # 简易 tokenizer（空格分词 + 编码）
            words = text.split()
            return [hash(w) % 50000 for w in words]

    def pack_sequences(self, token_streams: Iterator[List[int]],
                       eos_token_id: int = None) -> Iterator[np.ndarray]:
        """将多个短序列拼接成固定长度的训练序列

        策略：贪心拼接，用 EOS 分隔不同文档
        """
        if eos_token_id is None:
            eos_token_id = (self.tokenizer.eos_token_id
                           if self.tokenizer else 0)

        buffer = []

        for tokens in token_streams:
            buffer.extend(tokens)
            buffer.append(eos_token_id)

            while len(buffer) >= self.max_length:
                sequence = buffer[:self.max_length]
                buffer = buffer[self.max_length:]
                yield np.array(sequence, dtype=np.int32)
                self.stats.total_tokens += self.max_length

    def run(self, input_dir: str, output_dir: str):
        """运行完整流水线"""
        os.makedirs(output_dir, exist_ok=True)

        print(f"Tokenization 流水线启动")
        print(f"  输入: {input_dir}")
        print(f"  输出: {output_dir}")
        print(f"  序列长度: {self.max_length}")
        print()

        t_start = time.perf_counter()

        # 读取 → 清洗 → 过滤 → Tokenize
        def token_stream():
            for text in self.read_texts(input_dir):
                self.stats.total_docs += 1

                text = self.clean_text(text)
                if not self.filter_text(text):
                    self.stats.filtered_docs += 1
                    continue

                tokens = self.tokenize(text)
                yield tokens

        # 打包并保存
        output_file = os.path.join(output_dir, "train_tokens.bin")
        sequences_written = 0

        with open(output_file, "wb") as f:
            for seq in self.pack_sequences(token_stream()):
                f.write(seq.tobytes())
                sequences_written += 1

                if sequences_written % 1000 == 0:
                    elapsed = time.perf_counter() - t_start
                    tps = self.stats.total_tokens / elapsed
                    print(f"  进度: {sequences_written} sequences, "
                          f"{self.stats.total_tokens/1e6:.1f}M tokens, "
                          f"{tps:.0f} tokens/s")

        # 保存元数据
        self.stats.processing_time_s = time.perf_counter() - t_start
        self.stats.tokens_per_second = (
            self.stats.total_tokens / self.stats.processing_time_s
            if self.stats.processing_time_s > 0 else 0
        )

        meta = {
            "max_length": self.max_length,
            "num_sequences": sequences_written,
            "total_tokens": self.stats.total_tokens,
            "dtype": "int32",
            "vocab_size": (self.tokenizer.vocab_size
                          if self.tokenizer else 50000),
        }
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        self._print_stats()

    def _print_stats(self):
        """打印统计信息"""
        s = self.stats
        print(f"\n{'='*50}")
        print(f"流水线统计:")
        print(f"  文件数: {s.total_files}")
        print(f"  文档数: {s.total_docs}")
        print(f"  过滤文档: {s.filtered_docs} "
              f"({s.filtered_docs/max(s.total_docs,1)*100:.1f}%)")
        print(f"  总 token: {s.total_tokens:,}")
        print(f"  处理时间: {s.processing_time_s:.2f}s")
        print(f"  吞吐量: {s.tokens_per_second:.0f} tokens/s")


def generate_demo_data(output_dir: str, num_files: int = 100):
    """生成演示用的文本数据"""
    os.makedirs(output_dir, exist_ok=True)

    topics = [
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks consist of layers of interconnected nodes.",
        "Deep learning has revolutionized computer vision and NLP.",
        "Transformers use self-attention mechanisms for sequence modeling.",
        "Large language models are trained on vast amounts of text data.",
    ]

    for i in range(num_files):
        filepath = os.path.join(output_dir, f"doc_{i:04d}.txt")
        lines = []
        for _ in range(50):
            line = topics[np.random.randint(0, len(topics))]
            lines.append(line + " " + " ".join(
                [f"word{np.random.randint(0,1000)}" for _ in range(20)]))
        with open(filepath, "w") as f:
            f.write("\n".join(lines))

    print(f"生成 {num_files} 个演示文本文件到 {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Tokenization 流水线")
    parser.add_argument("--input-dir", type=str, default="/tmp/tok_demo/raw")
    parser.add_argument("--output-dir", type=str, default="/tmp/tok_demo/tokenized")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--generate-demo", action="store_true")
    args = parser.parse_args()

    if args.generate_demo:
        generate_demo_data(args.input_dir)

    pipeline = TokenizerPipeline(
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
    )
    pipeline.run(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
