"""
存储基准测试工具

测试不同存储方案在 AI 典型工作负载下的性能：
1. 顺序读（模型加载场景）
2. 随机读（数据加载场景）
3. 突发写（Checkpoint 场景）
4. 并发读（多 Worker 场景）

用法：
    python storage_benchmark.py --all --path /path/to/test/dir
    python storage_benchmark.py --test sequential_read --path /mnt/nfs/test
"""

import os
import sys
import time
import json
import argparse
import tempfile
import threading
import statistics
from pathlib import Path
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional


@dataclass
class BenchmarkResult:
    """单次测试结果"""
    test_name: str
    storage_path: str
    throughput_mbps: float       # MB/s
    iops: Optional[float]       # ops/s
    latency_avg_ms: float       # 平均延迟 ms
    latency_p99_ms: float       # P99 延迟 ms
    total_data_gb: float        # 总数据量 GB
    duration_seconds: float     # 总耗时
    num_workers: int = 1
    file_size_mb: float = 0
    notes: str = ""


class StorageBenchmark:
    """存储基准测试"""

    def __init__(self, test_dir: str, file_size_mb: int = 1024):
        """
        Args:
            test_dir: 测试目录路径
            file_size_mb: 测试文件大小（MB），默认 1GB
        """
        self.test_dir = Path(test_dir)
        self.file_size_mb = file_size_mb
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []

    def _create_test_file(self, filename: str, size_mb: int) -> str:
        """创建指定大小的测试文件"""
        filepath = self.test_dir / filename
        chunk_size = 64 * 1024 * 1024  # 64MB chunks
        remaining = size_mb * 1024 * 1024

        data = os.urandom(min(chunk_size, remaining))

        with open(filepath, "wb") as f:
            while remaining > 0:
                write_size = min(chunk_size, remaining)
                if write_size < len(data):
                    f.write(data[:write_size])
                else:
                    f.write(data)
                remaining -= write_size

        return str(filepath)

    def _create_small_files(self, num_files: int,
                            size_kb: int = 256) -> List[str]:
        """创建多个小文件"""
        small_dir = self.test_dir / "small_files"
        small_dir.mkdir(exist_ok=True)

        data = os.urandom(size_kb * 1024)
        files = []

        for i in range(num_files):
            filepath = small_dir / f"file_{i:06d}.bin"
            with open(filepath, "wb") as f:
                f.write(data)
            files.append(str(filepath))

        return files

    def test_sequential_read(self, file_size_mb: int = None) -> BenchmarkResult:
        """测试 1：顺序读性能（模拟模型加载）"""
        size = file_size_mb or self.file_size_mb
        print(f"\n{'='*60}")
        print(f"测试：顺序读（文件大小 {size}MB）")
        print(f"存储路径：{self.test_dir}")
        print(f"{'='*60}")

        # 准备测试文件
        print("准备测试文件...")
        filepath = self._create_test_file("seq_read_test.bin", size)

        # 清除 page cache（需要 root）
        self._drop_caches()

        # 执行顺序读
        read_size = 4 * 1024 * 1024  # 4MB 块读取
        latencies = []
        total_bytes = 0

        t_start = time.perf_counter()
        with open(filepath, "rb") as f:
            while True:
                t0 = time.perf_counter()
                data = f.read(read_size)
                if not data:
                    break
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)
                total_bytes += len(data)
        t_end = time.perf_counter()

        duration = t_end - t_start
        throughput = (total_bytes / 1024 / 1024) / duration

        result = BenchmarkResult(
            test_name="sequential_read",
            storage_path=str(self.test_dir),
            throughput_mbps=throughput,
            iops=len(latencies) / duration,
            latency_avg_ms=statistics.mean(latencies),
            latency_p99_ms=sorted(latencies)[int(len(latencies) * 0.99)],
            total_data_gb=total_bytes / 1024**3,
            duration_seconds=duration,
            file_size_mb=size,
        )

        print(f"结果：{throughput:.1f} MB/s, "
              f"总时间 {duration:.2f}s, "
              f"平均延迟 {result.latency_avg_ms:.2f}ms")

        self.results.append(result)
        os.remove(filepath)
        return result

    def test_random_read(self, num_files: int = 1000,
                         file_size_kb: int = 256) -> BenchmarkResult:
        """测试 2：随机读 IOPS（模拟数据加载）"""
        print(f"\n{'='*60}")
        print(f"测试：随机读（{num_files} 个 {file_size_kb}KB 文件）")
        print(f"存储路径：{self.test_dir}")
        print(f"{'='*60}")

        # 创建小文件
        print("准备测试文件...")
        files = self._create_small_files(num_files, file_size_kb)

        self._drop_caches()

        # 随机读取
        import random
        random.shuffle(files)

        latencies = []
        total_bytes = 0

        t_start = time.perf_counter()
        for filepath in files:
            t0 = time.perf_counter()
            with open(filepath, "rb") as f:
                data = f.read()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)
            total_bytes += len(data)
        t_end = time.perf_counter()

        duration = t_end - t_start
        iops = num_files / duration
        throughput = (total_bytes / 1024 / 1024) / duration

        result = BenchmarkResult(
            test_name="random_read",
            storage_path=str(self.test_dir),
            throughput_mbps=throughput,
            iops=iops,
            latency_avg_ms=statistics.mean(latencies),
            latency_p99_ms=sorted(latencies)[int(len(latencies) * 0.99)],
            total_data_gb=total_bytes / 1024**3,
            duration_seconds=duration,
            file_size_mb=file_size_kb / 1024,
            notes=f"{num_files} files",
        )

        print(f"结果：{iops:.0f} IOPS, "
              f"{throughput:.1f} MB/s, "
              f"平均延迟 {result.latency_avg_ms:.2f}ms")

        self.results.append(result)

        # 清理
        for f in files:
            os.remove(f)

        return result

    def test_burst_write(self, file_size_mb: int = None) -> BenchmarkResult:
        """测试 3：突发写（模拟 Checkpoint）"""
        size = file_size_mb or self.file_size_mb
        print(f"\n{'='*60}")
        print(f"测试：突发写（{size}MB 数据）")
        print(f"存储路径：{self.test_dir}")
        print(f"{'='*60}")

        filepath = self.test_dir / "burst_write_test.bin"
        chunk_size = 64 * 1024 * 1024  # 64MB
        data = os.urandom(chunk_size)
        remaining = size * 1024 * 1024

        latencies = []
        total_bytes = 0

        t_start = time.perf_counter()
        with open(filepath, "wb") as f:
            while remaining > 0:
                write_size = min(chunk_size, remaining)
                t0 = time.perf_counter()
                f.write(data[:write_size])
                f.flush()
                os.fsync(f.fileno())
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)
                total_bytes += write_size
                remaining -= write_size
        t_end = time.perf_counter()

        duration = t_end - t_start
        throughput = (total_bytes / 1024 / 1024) / duration

        result = BenchmarkResult(
            test_name="burst_write",
            storage_path=str(self.test_dir),
            throughput_mbps=throughput,
            iops=len(latencies) / duration,
            latency_avg_ms=statistics.mean(latencies),
            latency_p99_ms=sorted(latencies)[int(len(latencies) * 0.99)],
            total_data_gb=total_bytes / 1024**3,
            duration_seconds=duration,
            file_size_mb=size,
        )

        print(f"结果：{throughput:.1f} MB/s, "
              f"总时间 {duration:.2f}s, "
              f"平均延迟 {result.latency_avg_ms:.2f}ms")

        self.results.append(result)
        os.remove(filepath)
        return result

    def test_concurrent_read(self, num_workers: int = 8,
                             file_size_mb: int = 512) -> BenchmarkResult:
        """测试 4：并发读（模拟多 GPU DataLoader）"""
        print(f"\n{'='*60}")
        print(f"测试：{num_workers} 并发读（每个读 {file_size_mb}MB）")
        print(f"存储路径：{self.test_dir}")
        print(f"{'='*60}")

        # 创建多个测试文件
        print("准备测试文件...")
        filepaths = []
        for i in range(num_workers):
            fp = self._create_test_file(f"concurrent_{i}.bin", file_size_mb)
            filepaths.append(fp)

        self._drop_caches()

        # 并发读取
        results_per_worker = [None] * num_workers

        def worker_read(args):
            worker_id, filepath = args
            t0 = time.perf_counter()
            total = 0
            with open(filepath, "rb") as f:
                while True:
                    data = f.read(4 * 1024 * 1024)
                    if not data:
                        break
                    total += len(data)
            t1 = time.perf_counter()
            return total, t1 - t0

        t_start = time.perf_counter()
        with Pool(num_workers) as pool:
            worker_results = pool.map(
                worker_read,
                [(i, fp) for i, fp in enumerate(filepaths)]
            )
        t_end = time.perf_counter()

        duration = t_end - t_start
        total_bytes = sum(r[0] for r in worker_results)
        throughput = (total_bytes / 1024 / 1024) / duration
        per_worker_throughput = throughput / num_workers

        result = BenchmarkResult(
            test_name="concurrent_read",
            storage_path=str(self.test_dir),
            throughput_mbps=throughput,
            iops=None,
            latency_avg_ms=statistics.mean(r[1] * 1000 for r in worker_results),
            latency_p99_ms=max(r[1] * 1000 for r in worker_results),
            total_data_gb=total_bytes / 1024**3,
            duration_seconds=duration,
            num_workers=num_workers,
            file_size_mb=file_size_mb,
            notes=f"Per-worker: {per_worker_throughput:.1f} MB/s",
        )

        print(f"结果：聚合 {throughput:.1f} MB/s, "
              f"每 Worker {per_worker_throughput:.1f} MB/s, "
              f"总时间 {duration:.2f}s")

        self.results.append(result)

        for fp in filepaths:
            os.remove(fp)

        return result

    def _drop_caches(self):
        """尝试清除 page cache"""
        try:
            os.system("sync")
            with open("/proc/sys/vm/drop_caches", "w") as f:
                f.write("3")
        except (PermissionError, FileNotFoundError):
            print("  (无法清除 page cache，结果可能偏高)")

    def run_all(self) -> List[BenchmarkResult]:
        """运行全部测试"""
        print(f"\n{'#'*60}")
        print(f"# 存储基准测试")
        print(f"# 测试路径: {self.test_dir}")
        print(f"# 文件大小: {self.file_size_mb} MB")
        print(f"{'#'*60}")

        self.test_sequential_read()
        self.test_random_read()
        self.test_burst_write()
        self.test_concurrent_read()

        return self.results

    def save_results(self, output_path: str):
        """保存结果为 JSON"""
        data = {
            "test_dir": str(self.test_dir),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [asdict(r) for r in self.results],
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n结果已保存到: {output_path}")

    def print_summary(self):
        """打印汇总表格"""
        print(f"\n{'='*60}")
        print("汇总结果")
        print(f"{'='*60}")
        print(f"{'测试':<20} {'吞吐(MB/s)':<12} {'IOPS':<10} "
              f"{'延迟(ms)':<10} {'时间(s)':<8}")
        print("-" * 60)
        for r in self.results:
            iops_str = f"{r.iops:.0f}" if r.iops else "N/A"
            print(f"{r.test_name:<20} {r.throughput_mbps:<12.1f} "
                  f"{iops_str:<10} {r.latency_avg_ms:<10.2f} "
                  f"{r.duration_seconds:<8.2f}")


def compare_storage_paths(paths: List[str], file_size_mb: int = 1024):
    """对比多个存储路径的性能"""
    all_results = {}

    for path in paths:
        print(f"\n{'#'*60}")
        print(f"# 测试存储路径: {path}")
        print(f"{'#'*60}")

        bench = StorageBenchmark(path, file_size_mb)
        bench.run_all()
        bench.print_summary()
        all_results[path] = bench.results

    # 对比结果
    print(f"\n\n{'#'*60}")
    print("# 对比总结")
    print(f"{'#'*60}")

    test_names = ["sequential_read", "random_read", "burst_write",
                  "concurrent_read"]

    for test_name in test_names:
        print(f"\n--- {test_name} ---")
        for path, results in all_results.items():
            for r in results:
                if r.test_name == test_name:
                    print(f"  {path}: {r.throughput_mbps:.1f} MB/s")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="存储基准测试工具")
    parser.add_argument("--path", type=str, required=True,
                       help="测试目录路径")
    parser.add_argument("--test", type=str, default="all",
                       choices=["all", "sequential_read", "random_read",
                               "burst_write", "concurrent_read"],
                       help="测试项目")
    parser.add_argument("--file-size", type=int, default=1024,
                       help="测试文件大小(MB)")
    parser.add_argument("--compare", nargs="+", type=str,
                       help="对比多个存储路径")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                       help="结果输出路径")

    args = parser.parse_args()

    if args.compare:
        compare_storage_paths(args.compare, args.file_size)
    else:
        bench = StorageBenchmark(args.path, args.file_size)

        if args.test == "all":
            bench.run_all()
        elif args.test == "sequential_read":
            bench.test_sequential_read()
        elif args.test == "random_read":
            bench.test_random_read()
        elif args.test == "burst_write":
            bench.test_burst_write()
        elif args.test == "concurrent_read":
            bench.test_concurrent_read()

        bench.print_summary()
        bench.save_results(args.output)


if __name__ == "__main__":
    main()
