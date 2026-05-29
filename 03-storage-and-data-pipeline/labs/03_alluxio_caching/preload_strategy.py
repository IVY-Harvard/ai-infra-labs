"""
Alluxio 预加载策略测试

对比三种预加载方案：
1. 无预加载（冷启动）
2. 全量预加载
3. 分批渐进式预加载

用法：
    python preload_strategy.py --fuse-mount /mnt/alluxio --data-path /training-data/
"""

import os
import time
import json
import argparse
import subprocess
import threading
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class PreloadResult:
    """预加载策略测试结果"""
    strategy: str
    preload_time_s: float
    first_batch_latency_s: float   # 第一个 batch 的延迟
    avg_batch_latency_s: float     # 平均 batch 延迟
    total_throughput_mbps: float
    total_time_s: float


class PreloadStrategyBenchmark:
    """预加载策略对比"""

    def __init__(self, alluxio_home: str = None,
                 fuse_mount: str = "/mnt/alluxio"):
        self.alluxio_home = alluxio_home or os.environ.get(
            "ALLUXIO_HOME", "/opt/alluxio")
        self.fuse_mount = fuse_mount
        self.alluxio_bin = os.path.join(self.alluxio_home, "bin", "alluxio")
        self.results: List[PreloadResult] = []

    def _prepare_shard_files(self, data_dir: str,
                             num_shards: int = 32,
                             shard_size_mb: int = 64):
        """创建模拟的训练数据 shard 文件"""
        full_path = os.path.join(self.fuse_mount, data_dir.lstrip("/"))
        os.makedirs(full_path, exist_ok=True)

        existing = len([f for f in os.listdir(full_path) if f.endswith(".bin")])
        if existing >= num_shards:
            return

        print(f"创建 {num_shards} 个 shard 文件（每个 {shard_size_mb}MB）...")
        chunk = os.urandom(min(shard_size_mb * 1024 * 1024,
                              64 * 1024 * 1024))

        for i in range(num_shards):
            filepath = os.path.join(full_path, f"shard-{i:06d}.bin")
            remaining = shard_size_mb * 1024 * 1024
            with open(filepath, "wb") as f:
                while remaining > 0:
                    write_size = min(len(chunk), remaining)
                    f.write(chunk[:write_size])
                    remaining -= write_size

    def _free_all_cache(self, data_dir: str):
        """释放目录缓存"""
        subprocess.run(
            [self.alluxio_bin, "fs", "free", data_dir],
            capture_output=True,
        )
        time.sleep(2)

    def _read_shard(self, filepath: str) -> float:
        """读取单个 shard，返回耗时(秒)"""
        t0 = time.perf_counter()
        with open(filepath, "rb") as f:
            while f.read(4 * 1024 * 1024):
                pass
        return time.perf_counter() - t0

    def _simulate_training_epoch(self, data_dir: str) -> List[float]:
        """模拟训练一个 epoch，返回每个 batch 的延迟"""
        full_path = os.path.join(self.fuse_mount, data_dir.lstrip("/"))
        shards = sorted(Path(full_path).glob("shard-*.bin"))
        batch_latencies = []

        for shard in shards:
            latency = self._read_shard(str(shard))
            batch_latencies.append(latency)

        return batch_latencies

    def test_no_preload(self, data_dir: str):
        """策略 1：无预加载（冷启动）"""
        print(f"\n{'='*50}")
        print("策略: 无预加载（冷启动）")
        print(f"{'='*50}")

        self._free_all_cache(data_dir)

        t_start = time.perf_counter()
        latencies = self._simulate_training_epoch(data_dir)
        total_time = time.perf_counter() - t_start

        full_path = os.path.join(self.fuse_mount, data_dir.lstrip("/"))
        total_bytes = sum(
            os.path.getsize(os.path.join(full_path, f))
            for f in os.listdir(full_path) if f.endswith(".bin")
        )

        result = PreloadResult(
            strategy="no_preload",
            preload_time_s=0,
            first_batch_latency_s=latencies[0] if latencies else 0,
            avg_batch_latency_s=sum(latencies) / len(latencies) if latencies else 0,
            total_throughput_mbps=(total_bytes / 1024 / 1024) / total_time,
            total_time_s=total_time,
        )

        self.results.append(result)
        print(f"  首批延迟: {result.first_batch_latency_s:.2f}s")
        print(f"  平均延迟: {result.avg_batch_latency_s:.2f}s")
        print(f"  总吞吐: {result.total_throughput_mbps:.1f} MB/s")

    def test_full_preload(self, data_dir: str):
        """策略 2：全量预加载"""
        print(f"\n{'='*50}")
        print("策略: 全量预加载")
        print(f"{'='*50}")

        self._free_all_cache(data_dir)

        # 全量预加载
        print("  执行全量预加载...")
        t_preload = time.perf_counter()
        subprocess.run(
            [self.alluxio_bin, "fs", "load", data_dir, "--local"],
            capture_output=True,
        )
        preload_time = time.perf_counter() - t_preload
        print(f"  预加载耗时: {preload_time:.2f}s")

        # 训练
        t_start = time.perf_counter()
        latencies = self._simulate_training_epoch(data_dir)
        total_time = time.perf_counter() - t_start

        full_path = os.path.join(self.fuse_mount, data_dir.lstrip("/"))
        total_bytes = sum(
            os.path.getsize(os.path.join(full_path, f))
            for f in os.listdir(full_path) if f.endswith(".bin")
        )

        result = PreloadResult(
            strategy="full_preload",
            preload_time_s=preload_time,
            first_batch_latency_s=latencies[0] if latencies else 0,
            avg_batch_latency_s=sum(latencies) / len(latencies) if latencies else 0,
            total_throughput_mbps=(total_bytes / 1024 / 1024) / total_time,
            total_time_s=total_time + preload_time,
        )

        self.results.append(result)
        print(f"  首批延迟: {result.first_batch_latency_s:.2f}s")
        print(f"  平均延迟: {result.avg_batch_latency_s:.2f}s")
        print(f"  总吞吐: {result.total_throughput_mbps:.1f} MB/s")

    def test_progressive_preload(self, data_dir: str, lookahead: int = 4):
        """策略 3：渐进式预加载（边训练边预取下一批）"""
        print(f"\n{'='*50}")
        print(f"策略: 渐进式预加载（lookahead={lookahead}）")
        print(f"{'='*50}")

        self._free_all_cache(data_dir)

        full_path = os.path.join(self.fuse_mount, data_dir.lstrip("/"))
        shards = sorted(Path(full_path).glob("shard-*.bin"))

        latencies = []
        preload_time = 0

        def preload_shards(paths):
            """后台预加载"""
            for p in paths:
                alluxio_path = str(p).replace(self.fuse_mount, "")
                subprocess.run(
                    [self.alluxio_bin, "fs", "load", alluxio_path, "--local"],
                    capture_output=True,
                )

        t_start = time.perf_counter()

        # 预加载前 lookahead 个
        initial_shards = shards[:lookahead]
        t_pre = time.perf_counter()
        preload_shards(initial_shards)
        preload_time = time.perf_counter() - t_pre

        for i, shard in enumerate(shards):
            # 后台预加载未来的 shard
            if i + lookahead < len(shards):
                future_shards = shards[i + lookahead:i + lookahead + 1]
                t = threading.Thread(target=preload_shards,
                                    args=(future_shards,))
                t.start()

            latency = self._read_shard(str(shard))
            latencies.append(latency)

        total_time = time.perf_counter() - t_start

        total_bytes = sum(os.path.getsize(str(s)) for s in shards)

        result = PreloadResult(
            strategy=f"progressive_preload(lookahead={lookahead})",
            preload_time_s=preload_time,
            first_batch_latency_s=latencies[0] if latencies else 0,
            avg_batch_latency_s=sum(latencies) / len(latencies) if latencies else 0,
            total_throughput_mbps=(total_bytes / 1024 / 1024) / total_time,
            total_time_s=total_time,
        )

        self.results.append(result)
        print(f"  首批延迟: {result.first_batch_latency_s:.2f}s")
        print(f"  平均延迟: {result.avg_batch_latency_s:.2f}s")
        print(f"  总吞吐: {result.total_throughput_mbps:.1f} MB/s")

    def run_all(self, data_dir: str):
        """运行全部策略"""
        self._prepare_shard_files(data_dir)
        self.test_no_preload(data_dir)
        self.test_full_preload(data_dir)
        self.test_progressive_preload(data_dir)
        self.print_summary()

    def print_summary(self):
        """打印结果摘要"""
        print(f"\n{'='*70}")
        print("预加载策略对比结果")
        print(f"{'='*70}")
        print(f"{'策略':<40} {'首批(s)':<8} {'均延(s)':<8} "
              f"{'吞吐(MB/s)':<12} {'总时(s)':<8}")
        print("-" * 70)

        for r in self.results:
            print(f"{r.strategy:<40} "
                  f"{r.first_batch_latency_s:<8.2f} "
                  f"{r.avg_batch_latency_s:<8.2f} "
                  f"{r.total_throughput_mbps:<12.1f} "
                  f"{r.total_time_s:<8.1f}")


def main():
    parser = argparse.ArgumentParser(description="Alluxio 预加载策略测试")
    parser.add_argument("--alluxio-home", type=str, default=None)
    parser.add_argument("--fuse-mount", type=str, default="/mnt/alluxio")
    parser.add_argument("--data-path", type=str,
                       default="/test/preload-bench/")
    args = parser.parse_args()

    bench = PreloadStrategyBenchmark(
        alluxio_home=args.alluxio_home,
        fuse_mount=args.fuse_mount,
    )
    bench.run_all(args.data_path)


if __name__ == "__main__":
    main()
