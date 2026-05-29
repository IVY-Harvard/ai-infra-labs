"""
数据加载吞吐量基准测试

对比三种数据加载方案：
1. PyTorch 原生 DataLoader + 小文件
2. WebDataset + tar shard
3. StreamingDataset + MDS

用法：
    python throughput_benchmark.py --num-samples 50000 --batch-size 64
"""

import os
import time
import argparse
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class ThroughputResult:
    """吞吐量测试结果"""
    method: str
    samples_per_second: float
    mb_per_second: float
    batch_size: int
    num_workers: int
    total_samples: int
    duration_s: float


def benchmark_pytorch_native(data_dir: str, num_samples: int,
                              batch_size: int, num_workers: int,
                              num_batches: int = 200) -> ThroughputResult:
    """测试 PyTorch 原生 DataLoader"""
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
    except ImportError:
        print("PyTorch 未安装")
        return None

    # 创建小文件
    raw_dir = os.path.join(data_dir, "pytorch_raw")
    os.makedirs(raw_dir, exist_ok=True)

    sample_data = np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8)
    for i in range(min(num_samples, 10000)):
        np.save(os.path.join(raw_dir, f"{i:06d}.npy"), sample_data)

    class NumpyFileDataset(Dataset):
        def __init__(self, directory):
            self.files = sorted(Path(directory).glob("*.npy"))

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            data = np.load(self.files[idx])
            return torch.from_numpy(data).float(), 0

    dataset = NumpyFileDataset(raw_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )

    print(f"  PyTorch DataLoader: {len(dataset)} samples...")

    total_samples = 0
    t_start = time.perf_counter()

    for i, (images, labels) in enumerate(dataloader):
        total_samples += images.shape[0]
        if i + 1 >= num_batches:
            break

    duration = time.perf_counter() - t_start
    sample_size_kb = sample_data.nbytes / 1024

    return ThroughputResult(
        method="PyTorch DataLoader",
        samples_per_second=total_samples / duration,
        mb_per_second=total_samples * sample_size_kb / 1024 / duration,
        batch_size=batch_size,
        num_workers=num_workers,
        total_samples=total_samples,
        duration_s=duration,
    )


def benchmark_webdataset(data_dir: str, num_samples: int,
                          batch_size: int, num_workers: int,
                          num_batches: int = 200) -> ThroughputResult:
    """测试 WebDataset"""
    try:
        import webdataset as wds
        import torch
        from torch.utils.data import DataLoader
        import io
    except ImportError:
        print("  webdataset 或 torch 未安装")
        return None

    shard_dir = os.path.join(data_dir, "wds_shards")

    # 创建 shard（如果不存在）
    if not list(Path(shard_dir).glob("*.tar")):
        os.makedirs(shard_dir, exist_ok=True)
        pattern = os.path.join(shard_dir, "shard-%06d.tar")
        with wds.ShardWriter(pattern, maxcount=500) as sink:
            for i in range(min(num_samples, 10000)):
                arr = np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8)
                sink.write({
                    "__key__": f"{i:06d}",
                    "npy": arr.tobytes(),
                    "cls": json.dumps({"label": 0}).encode(),
                })

    urls = sorted(str(f) for f in Path(shard_dir).glob("shard-*.tar"))

    def decode_sample(sample):
        npy = np.frombuffer(sample["npy"], dtype=np.uint8).reshape(3, 224, 224)
        return npy.astype(np.float32), 0

    dataset = (
        wds.WebDataset(urls, shardshuffle=True)
        .shuffle(1000)
        .map(decode_sample)
        .batched(batch_size, partial=False)
    )

    dataloader = DataLoader(dataset, batch_size=None,
                           num_workers=num_workers, pin_memory=True)

    print(f"  WebDataset: {len(urls)} shards...")

    total_samples = 0
    t_start = time.perf_counter()

    for i, (images, labels) in enumerate(dataloader):
        total_samples += len(images)
        if i + 1 >= num_batches:
            break

    duration = time.perf_counter() - t_start
    sample_size_kb = 3 * 224 * 224 / 1024

    return ThroughputResult(
        method="WebDataset",
        samples_per_second=total_samples / duration,
        mb_per_second=total_samples * sample_size_kb / 1024 / duration,
        batch_size=batch_size,
        num_workers=num_workers,
        total_samples=total_samples,
        duration_s=duration,
    )


def main():
    parser = argparse.ArgumentParser(description="数据加载吞吐量测试")
    parser.add_argument("--data-dir", type=str, default="/tmp/throughput_bench")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    results = []

    print(f"\n{'#'*50}")
    print(f"# 数据加载吞吐量基准测试")
    print(f"# batch_size={args.batch_size}, workers={args.num_workers}")
    print(f"{'#'*50}")

    # 测试 1: PyTorch 原生
    print("\n--- PyTorch 原生 DataLoader ---")
    r = benchmark_pytorch_native(args.data_dir, args.num_samples,
                                  args.batch_size, args.num_workers)
    if r:
        results.append(r)
        print(f"  {r.samples_per_second:.0f} samples/s, "
              f"{r.mb_per_second:.1f} MB/s")

    # 测试 2: WebDataset
    print("\n--- WebDataset ---")
    r = benchmark_webdataset(args.data_dir, args.num_samples,
                              args.batch_size, args.num_workers)
    if r:
        results.append(r)
        print(f"  {r.samples_per_second:.0f} samples/s, "
              f"{r.mb_per_second:.1f} MB/s")

    # 汇总
    if results:
        print(f"\n{'='*60}")
        print("吞吐量对比")
        print(f"{'='*60}")
        print(f"{'方法':<25} {'samples/s':<12} {'MB/s':<10} {'时间(s)':<8}")
        print("-" * 60)

        for r in results:
            print(f"{r.method:<25} {r.samples_per_second:<12.0f} "
                  f"{r.mb_per_second:<10.1f} {r.duration_s:<8.2f}")


if __name__ == "__main__":
    main()
