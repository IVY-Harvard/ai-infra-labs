"""
WebDataset 流式数据加载演示

演示内容：
1. 将散落的小文件打成 WebDataset tar shard
2. 流式读取 + shuffle + transform + batch
3. 测量数据加载吞吐量

用法：
    python webdataset_demo.py --data-dir /tmp/wds_demo --num-samples 10000
"""

import os
import io
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Iterator


def create_sample_data(output_dir: str, num_samples: int = 10000,
                       image_size: tuple = (3, 224, 224)):
    """创建模拟的训练数据"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"创建 {num_samples} 个样本...")

    for i in range(num_samples):
        sample_dir = os.path.join(output_dir, "raw")
        os.makedirs(sample_dir, exist_ok=True)

        # 模拟图片数据（随机 numpy 数组）
        img = np.random.randint(0, 255, image_size, dtype=np.uint8)
        np.save(os.path.join(sample_dir, f"{i:06d}.npy"), img)

        # 标签
        label = {"class": np.random.randint(0, 1000), "text": f"sample_{i}"}
        with open(os.path.join(sample_dir, f"{i:06d}.json"), "w") as f:
            json.dump(label, f)

    print(f"  创建完成: {num_samples} 个样本")
    return os.path.join(output_dir, "raw")


def create_webdataset_shards(raw_dir: str, output_dir: str,
                              samples_per_shard: int = 500):
    """将散落文件打成 WebDataset tar shard"""
    try:
        import webdataset as wds
    except ImportError:
        print("请安装 webdataset: pip install webdataset")
        return

    os.makedirs(output_dir, exist_ok=True)
    pattern = os.path.join(output_dir, "shard-%06d.tar")

    npy_files = sorted(Path(raw_dir).glob("*.npy"))
    print(f"打包 {len(npy_files)} 个样本为 WebDataset shard...")

    with wds.ShardWriter(pattern, maxcount=samples_per_shard) as sink:
        for npy_path in npy_files:
            key = npy_path.stem
            json_path = npy_path.with_suffix(".json")

            with open(npy_path, "rb") as f:
                npy_data = f.read()

            json_data = b"{}"
            if json_path.exists():
                with open(json_path, "rb") as f:
                    json_data = f.read()

            sample = {
                "__key__": key,
                "npy": npy_data,
                "json": json_data,
            }
            sink.write(sample)

    shard_files = list(Path(output_dir).glob("shard-*.tar"))
    print(f"  创建 {len(shard_files)} 个 shard")
    return output_dir


def load_webdataset(shard_dir: str, batch_size: int = 32,
                     num_workers: int = 4, num_batches: int = 100):
    """使用 WebDataset 流式加载数据"""
    try:
        import webdataset as wds
        from torch.utils.data import DataLoader
    except ImportError:
        print("请安装 webdataset 和 torch")
        return

    shard_pattern = os.path.join(shard_dir, "shard-{000000..000099}.tar")

    # 检查实际有多少 shard
    shard_files = sorted(Path(shard_dir).glob("shard-*.tar"))
    if not shard_files:
        print("未找到 shard 文件")
        return

    urls = [str(f) for f in shard_files]

    def process_sample(sample):
        npy_data = np.load(io.BytesIO(sample["npy"]))
        label = json.loads(sample["json"])
        return npy_data, label.get("class", 0)

    dataset = (
        wds.WebDataset(urls, shardshuffle=True)
        .shuffle(1000)
        .map(process_sample)
        .batched(batch_size, partial=False)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 测量吞吐量
    print(f"\nWebDataset 加载测试 (batch_size={batch_size}, "
          f"workers={num_workers})...")

    total_samples = 0
    t_start = time.perf_counter()

    for i, (images, labels) in enumerate(dataloader):
        total_samples += len(images)
        if i + 1 >= num_batches:
            break

    duration = time.perf_counter() - t_start
    throughput = total_samples / duration

    print(f"  加载 {total_samples} 样本用时 {duration:.2f}s")
    print(f"  吞吐量: {throughput:.0f} samples/s")

    return throughput


def main():
    parser = argparse.ArgumentParser(description="WebDataset 流式加载演示")
    parser.add_argument("--data-dir", type=str, default="/tmp/wds_demo")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # 创建样本数据
    raw_dir = create_sample_data(args.data_dir, args.num_samples)

    # 打成 shard
    shard_dir = os.path.join(args.data_dir, "shards")
    create_webdataset_shards(raw_dir, shard_dir)

    # 加载测试
    load_webdataset(shard_dir, args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()
