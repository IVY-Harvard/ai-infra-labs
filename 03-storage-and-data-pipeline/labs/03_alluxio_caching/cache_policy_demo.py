"""
Alluxio 缓存策略对比测试

对比 CACHE / CACHE_PROMOTE / NO_CACHE 三种读策略
在 AI 训练数据加载场景下的性能差异。

用法：
    python cache_policy_demo.py --alluxio-host localhost --alluxio-port 19998
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class PolicyTestResult:
    """策略测试结果"""
    policy: str
    first_read_mbps: float      # 首次读（缓存未命中）
    second_read_mbps: float     # 二次读（缓存命中）
    speedup: float              # 加速比
    cache_usage_gb: float
    description: str


class AlluxioCachePolicyDemo:
    """Alluxio 缓存策略对比"""

    def __init__(self, alluxio_home: str = None,
                 fuse_mount: str = "/mnt/alluxio"):
        self.alluxio_home = alluxio_home or os.environ.get(
            "ALLUXIO_HOME", "/opt/alluxio")
        self.fuse_mount = fuse_mount
        self.alluxio_bin = os.path.join(self.alluxio_home, "bin", "alluxio")
        self.results: List[PolicyTestResult] = []

    def _alluxio_cmd(self, *args) -> subprocess.CompletedProcess:
        """执行 Alluxio CLI 命令"""
        cmd = [self.alluxio_bin, "fs"] + list(args)
        return subprocess.run(cmd, capture_output=True, text=True)

    def _create_test_data(self, path: str, size_mb: int = 512):
        """在 Alluxio 中创建测试数据"""
        local_path = os.path.join(self.fuse_mount, path.lstrip("/"))
        if os.path.exists(local_path):
            return

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        data = os.urandom(64 * 1024 * 1024)
        remaining = size_mb * 1024 * 1024

        with open(local_path, "wb") as f:
            while remaining > 0:
                write_size = min(len(data), remaining)
                f.write(data[:write_size])
                remaining -= write_size

    def _free_cache(self, path: str):
        """释放指定路径的缓存"""
        self._alluxio_cmd("free", path)
        time.sleep(2)

    def _read_throughput(self, path: str) -> float:
        """测量读吞吐"""
        local_path = os.path.join(self.fuse_mount, path.lstrip("/"))
        total_bytes = 0

        t_start = time.perf_counter()
        with open(local_path, "rb") as f:
            while True:
                data = f.read(4 * 1024 * 1024)
                if not data:
                    break
                total_bytes += len(data)
        t_end = time.perf_counter()

        duration = t_end - t_start
        if duration == 0:
            return 0
        return (total_bytes / 1024 / 1024) / duration

    def _get_cache_usage(self) -> float:
        """获取缓存使用量(GB)"""
        result = subprocess.run(
            [self.alluxio_bin, "fsadmin", "report", "capacity"],
            capture_output=True, text=True,
        )
        for line in result.stdout.split("\n"):
            if "Used" in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "Used":
                            size_str = parts[i + 1]
                            if "GB" in size_str:
                                return float(size_str.replace("GB", ""))
                            elif "MB" in size_str:
                                return float(size_str.replace("MB", "")) / 1024
                except (ValueError, IndexError):
                    pass
        return 0.0

    def test_cache_policy(self, test_path: str = "/test/cache_test.bin",
                          size_mb: int = 512):
        """测试 CACHE 策略"""
        print(f"\n{'='*50}")
        print("策略: CACHE（默认 — 读时缓存）")
        print(f"{'='*50}")

        self._create_test_data(test_path, size_mb)
        self._free_cache(test_path)

        # 首次读（缓存未命中 → 从后端存储读取并缓存）
        print("  首次读（缓存未命中）...")
        first_read = self._read_throughput(test_path)
        print(f"  首次读吞吐: {first_read:.1f} MB/s")

        # 二次读（缓存命中 → 从本地缓存读取）
        print("  二次读（缓存命中）...")
        second_read = self._read_throughput(test_path)
        print(f"  二次读吞吐: {second_read:.1f} MB/s")

        speedup = second_read / first_read if first_read > 0 else 0

        self.results.append(PolicyTestResult(
            policy="CACHE",
            first_read_mbps=first_read,
            second_read_mbps=second_read,
            speedup=speedup,
            cache_usage_gb=self._get_cache_usage(),
            description="读时自动缓存，适合大多数训练场景",
        ))

    def test_no_cache_policy(self, test_path: str = "/test/cache_test.bin"):
        """测试 NO_CACHE 策略"""
        print(f"\n{'='*50}")
        print("策略: NO_CACHE（不缓存，每次从后端读）")
        print(f"{'='*50}")

        self._free_cache(test_path)

        # 两次读都应该是从后端存储
        print("  首次读...")
        first_read = self._read_throughput(test_path)
        print(f"  首次读吞吐: {first_read:.1f} MB/s")

        print("  二次读...")
        second_read = self._read_throughput(test_path)
        print(f"  二次读吞吐: {second_read:.1f} MB/s")

        self.results.append(PolicyTestResult(
            policy="NO_CACHE",
            first_read_mbps=first_read,
            second_read_mbps=second_read,
            speedup=second_read / first_read if first_read > 0 else 0,
            cache_usage_gb=self._get_cache_usage(),
            description="不缓存，适合一次性扫描的冷数据",
        ))

    def test_preload_then_read(self, test_path: str = "/test/cache_test.bin"):
        """测试预加载后读取"""
        print(f"\n{'='*50}")
        print("策略: 预加载 + CACHE（训练前预热）")
        print(f"{'='*50}")

        self._free_cache(test_path)

        # 预加载到缓存
        print("  预加载到缓存...")
        t0 = time.perf_counter()
        self._alluxio_cmd("load", test_path, "--local")
        preload_time = time.perf_counter() - t0
        print(f"  预加载时间: {preload_time:.2f}s")

        # 读取（应该全部命中缓存）
        print("  缓存读取...")
        read_mbps = self._read_throughput(test_path)
        print(f"  读吞吐: {read_mbps:.1f} MB/s")

        self.results.append(PolicyTestResult(
            policy="PRELOAD + CACHE",
            first_read_mbps=read_mbps,
            second_read_mbps=read_mbps,
            speedup=1.0,
            cache_usage_gb=self._get_cache_usage(),
            description="训练前主动预热，首次读就命中缓存",
        ))

    def run_all_tests(self):
        """运行全部策略测试"""
        print(f"\n{'#'*50}")
        print("# Alluxio 缓存策略对比测试")
        print(f"{'#'*50}")

        self.test_cache_policy()
        self.test_no_cache_policy()
        self.test_preload_then_read()
        self.print_summary()

    def print_summary(self):
        """打印结果摘要"""
        print(f"\n{'='*70}")
        print("策略对比结果")
        print(f"{'='*70}")
        print(f"{'策略':<25} {'首读(MB/s)':<12} {'二读(MB/s)':<12} "
              f"{'加速比':<8} {'说明'}")
        print("-" * 70)

        for r in self.results:
            print(f"{r.policy:<25} {r.first_read_mbps:<12.1f} "
                  f"{r.second_read_mbps:<12.1f} {r.speedup:<8.1f}x "
                  f"{r.description}")


def main():
    parser = argparse.ArgumentParser(description="Alluxio 缓存策略测试")
    parser.add_argument("--alluxio-home", type=str, default=None,
                       help="Alluxio 安装目录")
    parser.add_argument("--fuse-mount", type=str, default="/mnt/alluxio",
                       help="Alluxio FUSE 挂载点")
    args = parser.parse_args()

    demo = AlluxioCachePolicyDemo(
        alluxio_home=args.alluxio_home,
        fuse_mount=args.fuse_mount,
    )
    demo.run_all_tests()


if __name__ == "__main__":
    main()
