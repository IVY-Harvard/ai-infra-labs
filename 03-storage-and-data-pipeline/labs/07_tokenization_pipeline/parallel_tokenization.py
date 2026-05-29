"""
多进程并行 Tokenization

利用全部 CPU 核心并行处理，大幅提升 Tokenization 吞吐量。

用法：
    python parallel_tokenization.py --input-dir /path/to/texts --output-dir /tmp/parallel_tok
    python parallel_tokenization.py --input-dir /tmp/tok_demo/raw --num-workers 8
"""

import os
import time
import json
import argparse
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import List, Tuple


def tokenize_file(filepath: str, tokenizer_name: str,
                  max_length: int, min_text_length: int) -> dict:
    """处理单个文件（在子进程中执行）

    注意：每个子进程需要独立加载 tokenizer（不能跨进程共享）
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except ImportError:
        tokenizer = None

    results = []
    total_tokens = 0
    filtered = 0

    try:
        if filepath.endswith(".jsonl"):
            with open(filepath, "r", encoding="utf-8") as f:
                texts = []
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        text = data.get("text", "")
                        if text:
                            texts.append(text)
                    except json.JSONDecodeError:
                        continue
        else:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                texts = [f.read()]

        for text in texts:
            # 过滤
            if len(text) < min_text_length:
                filtered += 1
                continue

            # Tokenize
            if tokenizer:
                encoded = tokenizer(
                    text,
                    max_length=max_length,
                    truncation=True,
                    padding=False,
                    return_attention_mask=False,
                )
                token_ids = encoded["input_ids"]
            else:
                token_ids = [hash(w) % 50000 for w in text.split()][:max_length]

            results.append(token_ids)
            total_tokens += len(token_ids)

    except Exception as e:
        return {"error": str(e), "filepath": filepath}

    return {
        "filepath": filepath,
        "num_sequences": len(results),
        "total_tokens": total_tokens,
        "filtered": filtered,
        "token_ids": results,
    }


def parallel_tokenize(input_dir: str, output_dir: str,
                      tokenizer_name: str = "gpt2",
                      max_length: int = 2048,
                      min_text_length: int = 50,
                      num_workers: int = None):
    """多进程并行 Tokenization"""
    if num_workers is None:
        num_workers = cpu_count()

    os.makedirs(output_dir, exist_ok=True)

    # 收集所有文件
    input_path = Path(input_dir)
    files = sorted(
        [str(f) for f in input_path.rglob("*")
         if f.suffix in (".txt", ".jsonl")]
    )

    if not files:
        print(f"在 {input_dir} 中未找到文本文件")
        return

    print(f"并行 Tokenization")
    print(f"  输入目录: {input_dir}")
    print(f"  文件数量: {len(files)}")
    print(f"  Worker 数: {num_workers}")
    print(f"  Tokenizer: {tokenizer_name}")
    print()

    # 创建 worker 函数
    worker_fn = partial(
        tokenize_file,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        min_text_length=min_text_length,
    )

    # 并行处理
    total_tokens = 0
    total_sequences = 0
    total_filtered = 0
    errors = 0

    t_start = time.perf_counter()

    with Pool(num_workers) as pool:
        all_token_ids = []

        for i, result in enumerate(pool.imap_unordered(worker_fn, files)):
            if "error" in result:
                errors += 1
                continue

            total_tokens += result["total_tokens"]
            total_sequences += result["num_sequences"]
            total_filtered += result["filtered"]

            # 收集 token ids
            all_token_ids.extend(result["token_ids"])

            if (i + 1) % 100 == 0 or i + 1 == len(files):
                elapsed = time.perf_counter() - t_start
                tps = total_tokens / elapsed if elapsed > 0 else 0
                print(f"  进度: {i+1}/{len(files)} files, "
                      f"{total_tokens/1e6:.1f}M tokens, "
                      f"{tps:.0f} tokens/s")

    processing_time = time.perf_counter() - t_start

    # 保存结果
    print(f"\n保存结果...")
    output_file = os.path.join(output_dir, "tokenized.bin")
    with open(output_file, "wb") as f:
        for token_ids in all_token_ids:
            arr = np.array(token_ids, dtype=np.int32)
            f.write(arr.tobytes())

    # 保存元数据
    meta = {
        "total_files": len(files),
        "total_sequences": total_sequences,
        "total_tokens": total_tokens,
        "filtered_docs": total_filtered,
        "errors": errors,
        "processing_time_s": processing_time,
        "tokens_per_second": total_tokens / processing_time,
        "num_workers": num_workers,
        "tokenizer": tokenizer_name,
        "max_length": max_length,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # 打印汇总
    print(f"\n{'='*50}")
    print(f"并行 Tokenization 完成:")
    print(f"  总文件: {len(files)}")
    print(f"  总 token: {total_tokens:,}")
    print(f"  过滤文档: {total_filtered}")
    print(f"  错误: {errors}")
    print(f"  处理时间: {processing_time:.2f}s")
    print(f"  吞吐量: {total_tokens/processing_time:.0f} tokens/s")
    print(f"  vs 单线程估计: ~{total_tokens/processing_time*num_workers/total_tokens:.0f}x 加速")
    print(f"  输出: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="并行 Tokenization")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="/tmp/parallel_tok")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    parallel_tokenize(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
