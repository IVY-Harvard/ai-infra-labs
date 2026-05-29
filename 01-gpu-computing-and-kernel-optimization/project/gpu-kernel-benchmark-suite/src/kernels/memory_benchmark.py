"""
GPU 显存带宽基准测试

测试 HBM 的实际读写带宽。
使用大数组 copy 操作（memory-bound）来逼近硬件带宽上限。
"""

import torch
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MemoryResult:
    """显存带宽测试结果"""
    test_name: str
    data_size_mb: float
    time_ms: float
    bandwidth_gb_s: float
    peak_bandwidth_gb_s: float
    utilization: float


class MemoryBenchmark:
    """GPU 显存带宽基准测试"""

    DEFAULT_PEAK_BW = 4000.0  # H20: 4 TB/s = 4000 GB/s

    def __init__(self, device_id: int = 0, peak_bw_gb_s: float = None):
        self.device = torch.device(f'cuda:{device_id}')
        self.device_id = device_id
        self.peak_bw = peak_bw_gb_s or self.DEFAULT_PEAK_BW

    def _benchmark_copy(self, size_bytes: int, warmup: int = 10, runs: int = 50) -> float:
        """测量 device-to-device copy 时间 (ms)"""
        n_elements = size_bytes // 4  # float32
        src = torch.randn(n_elements, device=self.device)
        dst = torch.empty_like(src)

        for _ in range(warmup):
            dst.copy_(src)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            dst.copy_(src)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / runs * 1000

        del src, dst
        torch.cuda.empty_cache()
        return elapsed

    def benchmark_read(self, sizes_mb: List[int] = None) -> List[MemoryResult]:
        """测试 HBM 读带宽"""
        if sizes_mb is None:
            sizes_mb = [256, 512, 1024]

        results = []
        for size_mb in sizes_mb:
            size_bytes = size_mb * 1024 * 1024
            n_elements = size_bytes // 4

            src = torch.randn(n_elements, device=self.device)
            dummy = torch.zeros(1, device=self.device)

            # 只读不写: sum 操作
            for _ in range(10):
                _ = src.sum()
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(50):
                _ = src.sum()
            torch.cuda.synchronize()
            time_ms = (time.perf_counter() - start) / 50 * 1000

            bw = size_bytes / (time_ms / 1000) / 1e9
            results.append(MemoryResult(
                test_name=f'HBM Read',
                data_size_mb=size_mb,
                time_ms=time_ms,
                bandwidth_gb_s=bw,
                peak_bandwidth_gb_s=self.peak_bw,
                utilization=bw / self.peak_bw * 100,
            ))

            del src
            torch.cuda.empty_cache()

        return results

    def benchmark_copy(self, sizes_mb: List[int] = None) -> List[MemoryResult]:
        """测试 HBM copy 带宽 (读+写)"""
        if sizes_mb is None:
            sizes_mb = [256, 512, 1024]

        results = []
        for size_mb in sizes_mb:
            size_bytes = size_mb * 1024 * 1024
            time_ms = self._benchmark_copy(size_bytes)

            # Copy = 读 + 写，有效带宽 = 2 × size / time
            bw = 2 * size_bytes / (time_ms / 1000) / 1e9
            results.append(MemoryResult(
                test_name=f'HBM Copy (R+W)',
                data_size_mb=size_mb,
                time_ms=time_ms,
                bandwidth_gb_s=bw,
                peak_bandwidth_gb_s=self.peak_bw,
                utilization=bw / self.peak_bw * 100,
            ))

        return results

    def benchmark_h2d(self, sizes_mb: List[int] = None) -> List[MemoryResult]:
        """测试 Host-to-Device 传输带宽 (PCIe)"""
        if sizes_mb is None:
            sizes_mb = [64, 256, 1024]

        results = []
        for size_mb in sizes_mb:
            n_elements = size_mb * 1024 * 1024 // 4

            h_data = torch.randn(n_elements, pin_memory=True)
            d_data = torch.empty(n_elements, device=self.device)

            # Warmup
            for _ in range(5):
                d_data.copy_(h_data)
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(20):
                d_data.copy_(h_data)
            torch.cuda.synchronize()
            time_ms = (time.perf_counter() - start) / 20 * 1000

            bw = size_mb * 1024 * 1024 / (time_ms / 1000) / 1e9
            results.append(MemoryResult(
                test_name='H2D (PCIe)',
                data_size_mb=size_mb,
                time_ms=time_ms,
                bandwidth_gb_s=bw,
                peak_bandwidth_gb_s=64.0,  # PCIe Gen5 x16
                utilization=bw / 64.0 * 100,
            ))

            del h_data, d_data
            torch.cuda.empty_cache()

        return results

    def run_all(self) -> dict:
        """运行全部内存带宽测试"""
        return {
            'read': self.benchmark_read(),
            'copy': self.benchmark_copy(),
            'h2d': self.benchmark_h2d(),
        }

    def print_results(self, results: dict):
        """打印结果"""
        print(f"\n显存带宽测试 (GPU {self.device_id})")
        print(f"{'Test':<18} | {'Size(MB)':>8} | {'Time(ms)':>10} | {'BW(GB/s)':>10} | {'Peak':>10} | {'Util%':>6}")
        print("-" * 75)
        for category, res_list in results.items():
            for r in res_list:
                print(f"{r.test_name:<18} | {r.data_size_mb:>8.0f} | {r.time_ms:>10.3f} | "
                      f"{r.bandwidth_gb_s:>10.1f} | {r.peak_bandwidth_gb_s:>10.1f} | {r.utilization:>5.1f}%")


if __name__ == "__main__":
    bench = MemoryBenchmark(device_id=0)
    results = bench.run_all()
    bench.print_results(results)
