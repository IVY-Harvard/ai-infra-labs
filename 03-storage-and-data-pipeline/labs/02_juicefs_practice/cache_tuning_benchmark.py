"""
JuiceFS 缓存参数调优基准测试

测试不同缓存配置对 AI 工作负载的影响：
- cache-size: 缓存大小
- prefetch: 预读取块数
- buffer-size: 写缓冲区大小

用法：
    python cache_tuning_benchmark.py --jfs-mount /mnt/jfs --cache-dir /nvme/jfs-cache
"""

import os
import sys
import time
import json
import subprocess
import argparse
import tempfile
import statistics
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple


@dataclass
class CacheBenchResult:
    """缓存测试结果"""
    config_name: str
    cache_size_gb: int
    prefetch: int
    buffer_size_mb: int
    read_throughput_mbps: float
    write_throughput_mbps: float
    cache_hit_rate: float
    warmup_time_s: float
    model_load_time_s: float


class JuiceFSCacheBenchmark:
    """JuiceFS 缓存调优测试"""

    def __init__(self, jfs_mount: str, cache_dir: str,
                 meta_url: str = "redis://localhost:6379/1"):
        self.jfs_mount = jfs_mount
        self.cache_dir = cache_dir
        self.meta_url = meta_url
        self.results: List[CacheBenchResult] = []

    def _remount_jfs(self, cache_size_gb: int, prefetch: int,
                     buffer_size_mb: int) -> bool:
        """使用新参数重新挂载 JuiceFS"""
        print(f"  重新挂载: cache={cache_size_gb}GB, "
              f"prefetch={prefetch}, buffer={buffer_size_mb}MB")

        # 卸载
        subprocess.run(["juicefs", "umount", self.jfs_mount],
                      capture_output=True)
        time.sleep(2)

        # 清除缓存
        cache_path = Path(self.cache_dir)
        if cache_path.exists():
            subprocess.run(["rm", "-rf", str(cache_path / "*")],
                          capture_output=True, shell=True)

        # 重新挂载
        cmd = [
            "juicefs", "mount",
            self.meta_url,
            self.jfs_mount,
            "--cache-dir", self.cache_dir,
            "--cache-size", str(cache_size_gb * 1000),  # MB
            "--prefetch", str(prefetch),
            "--buffer-size", str(buffer_size_mb),
            "--max-uploads", "30",
            "-d",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  挂载失败: {result.stderr}")
            return False

        time.sleep(3)  # 等待挂载就绪
        return True

    def _prepare_test_data(self, size_mb: int = 2048) -> str:
        """准备测试数据"""
        test_file = os.path.join(self.jfs_mount, "bench_test_data.bin")
        if os.path.exists(test_file):
            current_size = os.path.getsize(test_file) / 1024 / 1024
            if abs(current_size - size_mb) < 10:
                return test_file

        print(f"  创建 {size_mb}MB 测试文件...")
        chunk = os.urandom(64 * 1024 * 1024)
        remaining = size_mb * 1024 * 1024

        with open(test_file, "wb") as f:
            while remaining > 0:
                write_size = min(len(chunk), remaining)
                f.write(chunk[:write_size])
                remaining -= write_size

        return test_file

    def _measure_read_throughput(self, filepath: str,
                                 clear_cache: bool = False) -> float:
        """测量读吞吐"""
        if clear_cache:
            # 清除 JuiceFS 缓存和 page cache
            subprocess.run(["juicefs", "warmup", "--evict", filepath],
                          capture_output=True)
            try:
                with open("/proc/sys/vm/drop_caches", "w") as f:
                    f.write("3")
            except (PermissionError, FileNotFoundError):
                pass

        total_bytes = 0
        t_start = time.perf_counter()

        with open(filepath, "rb") as f:
            while True:
                data = f.read(4 * 1024 * 1024)  # 4MB chunks
                if not data:
                    break
                total_bytes += len(data)

        t_end = time.perf_counter()
        duration = t_end - t_start
        return (total_bytes / 1024 / 1024) / duration

    def _measure_write_throughput(self, size_mb: int = 1024) -> float:
        """测量写吞吐"""
        filepath = os.path.join(self.jfs_mount, "bench_write_test.bin")
        chunk = os.urandom(64 * 1024 * 1024)
        remaining = size_mb * 1024 * 1024
        total_bytes = 0

        t_start = time.perf_counter()
        with open(filepath, "wb") as f:
            while remaining > 0:
                write_size = min(len(chunk), remaining)
                f.write(chunk[:write_size])
                total_bytes += write_size
                remaining -= write_size
            f.flush()
            os.fsync(f.fileno())
        t_end = time.perf_counter()

        duration = t_end - t_start
        os.remove(filepath)
        return (total_bytes / 1024 / 1024) / duration

    def _measure_warmup_time(self, filepath: str) -> float:
        """测量缓存预热时间"""
        t_start = time.perf_counter()
        result = subprocess.run(
            ["juicefs", "warmup", "-p", "8", filepath],
            capture_output=True, text=True,
        )
        t_end = time.perf_counter()
        return t_end - t_start

    def _get_cache_hit_rate(self) -> float:
        """获取缓存命中率"""
        result = subprocess.run(
            ["juicefs", "stats", self.jfs_mount, "--interval", "0"],
            capture_output=True, text=True,
        )
        # 解析 stats 输出
        for line in result.stdout.split("\n"):
            if "hitrate" in line.lower() or "hit" in line.lower():
                try:
                    # 尝试提取百分比
                    parts = line.split()
                    for part in parts:
                        if "%" in part:
                            return float(part.replace("%", "")) / 100
                except (ValueError, IndexError):
                    pass
        return 0.0

    def run_config_test(self, cache_size_gb: int, prefetch: int,
                        buffer_size_mb: int) -> CacheBenchResult:
        """测试特定配置"""
        config_name = (f"cache{cache_size_gb}G_pf{prefetch}_"
                      f"buf{buffer_size_mb}M")
        print(f"\n{'='*50}")
        print(f"测试配置: {config_name}")
        print(f"{'='*50}")

        # 重新挂载
        if not self._remount_jfs(cache_size_gb, prefetch, buffer_size_mb):
            print("  跳过（挂载失败）")
            return None

        # 准备数据
        test_file = self._prepare_test_data(2048)

        # 测试 1: 预热时间
        print("  测试预热时间...")
        warmup_time = self._measure_warmup_time(test_file)
        print(f"  预热时间: {warmup_time:.2f}s")

        # 测试 2: 缓存命中时的读吞吐
        print("  测试缓存命中读吞吐...")
        read_throughput = self._measure_read_throughput(test_file,
                                                        clear_cache=False)
        print(f"  读吞吐: {read_throughput:.1f} MB/s")

        # 测试 3: 写吞吐
        print("  测试写吞吐...")
        write_throughput = self._measure_write_throughput(1024)
        print(f"  写吞吐: {write_throughput:.1f} MB/s")

        # 测试 4: 模型加载时间模拟（2GB 文件顺序读）
        print("  测试模型加载时间...")
        model_load_time = os.path.getsize(test_file) / 1024 / 1024 / read_throughput
        print(f"  模型加载时间(2GB): {model_load_time:.2f}s")

        # 缓存命中率
        cache_hit_rate = self._get_cache_hit_rate()

        result = CacheBenchResult(
            config_name=config_name,
            cache_size_gb=cache_size_gb,
            prefetch=prefetch,
            buffer_size_mb=buffer_size_mb,
            read_throughput_mbps=read_throughput,
            write_throughput_mbps=write_throughput,
            cache_hit_rate=cache_hit_rate,
            warmup_time_s=warmup_time,
            model_load_time_s=model_load_time,
        )

        self.results.append(result)
        return result

    def run_all_configs(self):
        """测试所有配置组合"""
        configs = [
            # (cache_size_gb, prefetch, buffer_size_mb)
            (100, 1, 1024),    # 基线：小缓存
            (300, 1, 1024),    # 中缓存
            (500, 1, 1024),    # 大缓存
            (300, 3, 1024),    # 中缓存+预读取
            (300, 5, 1024),    # 中缓存+更多预读取
            (300, 3, 2048),    # 中缓存+大写缓冲
            (300, 3, 4096),    # 中缓存+超大写缓冲
            (500, 3, 4096),    # 最优配置
        ]

        for cache_gb, prefetch, buffer_mb in configs:
            self.run_config_test(cache_gb, prefetch, buffer_mb)

    def print_summary(self):
        """打印结果摘要"""
        print(f"\n{'='*80}")
        print("缓存调优结果汇总")
        print(f"{'='*80}")
        print(f"{'配置':<30} {'读(MB/s)':<10} {'写(MB/s)':<10} "
              f"{'预热(s)':<8} {'加载(s)':<8}")
        print("-" * 80)

        for r in self.results:
            print(f"{r.config_name:<30} {r.read_throughput_mbps:<10.1f} "
                  f"{r.write_throughput_mbps:<10.1f} "
                  f"{r.warmup_time_s:<8.1f} {r.model_load_time_s:<8.2f}")

        # 推荐配置
        if self.results:
            best_read = max(self.results, key=lambda x: x.read_throughput_mbps)
            best_write = max(self.results, key=lambda x: x.write_throughput_mbps)

            print(f"\n推荐配置：")
            print(f"  最佳读性能: {best_read.config_name} "
                  f"({best_read.read_throughput_mbps:.1f} MB/s)")
            print(f"  最佳写性能: {best_write.config_name} "
                  f"({best_write.write_throughput_mbps:.1f} MB/s)")

    def save_results(self, output_path: str):
        """保存结果"""
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "jfs_mount": self.jfs_mount,
            "results": [asdict(r) for r in self.results],
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="JuiceFS 缓存调优测试")
    parser.add_argument("--jfs-mount", type=str, default="/mnt/jfs",
                       help="JuiceFS 挂载点")
    parser.add_argument("--cache-dir", type=str, default="/nvme/jfs-cache",
                       help="缓存目录")
    parser.add_argument("--meta-url", type=str,
                       default="redis://localhost:6379/1",
                       help="元数据引擎 URL")
    parser.add_argument("--output", type=str,
                       default="cache_tuning_results.json",
                       help="结果输出路径")
    args = parser.parse_args()

    bench = JuiceFSCacheBenchmark(
        jfs_mount=args.jfs_mount,
        cache_dir=args.cache_dir,
        meta_url=args.meta_url,
    )

    bench.run_all_configs()
    bench.print_summary()
    bench.save_results(args.output)


if __name__ == "__main__":
    main()
