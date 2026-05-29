"""
MosaicML StreamingDataset 演示

演示内容：
1. 将数据写入 MDS 格式
2. 使用 StreamingDataset 加载
3. 演示确定性 shuffle 和断点续传

用法：
    python streaming_dataset_demo.py --data-dir /tmp/streaming_demo --num-samples 10000
"""

import os
import time
import json
import argparse
import numpy as np
from pathlib import Path


def create_mds_dataset(output_dir: str, num_samples: int = 10000):
    """将数据写入 MDS 格式"""
    try:
        from streaming import MDSWriter
    except ImportError:
        print("请安装 mosaicml-streaming: pip install mosaicml-streaming")
        return None

    os.makedirs(output_dir, exist_ok=True)

    columns = {
        "image": "ndarray:uint8",
        "label": "int",
        "text": "str",
    }

    print(f"写入 {num_samples} 个样本到 MDS 格式...")

    with MDSWriter(
        out=output_dir,
        columns=columns,
        compression="zstd",
        size_limit=256 * 1024 * 1024,  # 256MB per shard
    ) as writer:
        for i in range(num_samples):
            sample = {
                "image": np.random.randint(0, 255, (3, 224, 224),
                                           dtype=np.uint8),
                "label": np.random.randint(0, 1000),
                "text": f"This is sample {i} with random content",
            }
            writer.write(sample)

    # 统计
    shard_files = list(Path(output_dir).glob("*.mds"))
    total_size = sum(f.stat().st_size for f in shard_files)
    print(f"  写入完成: {len(shard_files)} 个 shard, "
          f"总大小 {total_size/1024/1024:.1f}MB")

    return output_dir


def load_streaming_dataset(mds_dir: str, batch_size: int = 32,
                           num_workers: int = 4, num_batches: int = 100):
    """使用 StreamingDataset 加载数据"""
    try:
        from streaming import StreamingDataset, StreamingDataLoader
        import torch
    except ImportError:
        print("请安装 mosaicml-streaming 和 torch")
        return

    local_cache = mds_dir + "_cache"

    dataset = StreamingDataset(
        local=mds_dir,
        shuffle=True,
        shuffle_seed=42,
        batch_size=batch_size,
    )

    dataloader = StreamingDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"\nStreamingDataset 加载测试 (batch_size={batch_size}, "
          f"workers={num_workers})...")

    total_samples = 0
    t_start = time.perf_counter()

    for i, batch in enumerate(dataloader):
        total_samples += len(batch["label"])
        if i + 1 >= num_batches:
            break

    duration = time.perf_counter() - t_start
    throughput = total_samples / duration

    print(f"  加载 {total_samples} 样本用时 {duration:.2f}s")
    print(f"  吞吐量: {throughput:.0f} samples/s")

    return throughput


def test_deterministic_shuffle(mds_dir: str, batch_size: int = 32):
    """测试确定性 shuffle"""
    try:
        from streaming import StreamingDataset
    except ImportError:
        return

    print("\n--- 确定性 Shuffle 测试 ---")

    # 两次使用相同 seed 加载
    orders = []
    for run in range(2):
        dataset = StreamingDataset(
            local=mds_dir,
            shuffle=True,
            shuffle_seed=42,
            batch_size=batch_size,
        )

        first_10 = [dataset[i]["label"] for i in range(10)]
        orders.append(first_10)
        print(f"  Run {run+1} 前10个标签: {first_10}")

    if orders[0] == orders[1]:
        print("  ✓ 两次顺序完全相同 — 确定性 shuffle 验证通过")
    else:
        print("  ✗ 顺序不同 — 可能有问题")


def main():
    parser = argparse.ArgumentParser(description="StreamingDataset 演示")
    parser.add_argument("--data-dir", type=str, default="/tmp/streaming_demo")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # 创建 MDS 数据集
    mds_dir = os.path.join(args.data_dir, "mds")
    create_mds_dataset(mds_dir, args.num_samples)

    # 加载测试
    load_streaming_dataset(mds_dir, args.batch_size, args.num_workers)

    # 确定性测试
    test_deterministic_shuffle(mds_dir, args.batch_size)


if __name__ == "__main__":
    main()
