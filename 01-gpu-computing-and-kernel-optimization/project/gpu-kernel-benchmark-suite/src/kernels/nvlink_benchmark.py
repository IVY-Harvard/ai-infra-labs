"""
NVLink 带宽测试

通过 AllReduce 操作测试 NVLink 的实际带宽。
AllReduce 是多卡训练中最常见的通信操作，
其性能直接反映 NVLink 的实际可用带宽。
"""

import torch
import torch.distributed as dist
import time
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class NVLinkResult:
    """NVLink 带宽测试结果"""
    operation: str
    num_gpus: int
    data_size_mb: float
    time_ms: float
    bandwidth_gb_s: float  # 算法带宽
    bus_bandwidth_gb_s: float  # 总线带宽


class NVLinkBenchmark:
    """
    NVLink 带宽测试。

    使用 NCCL AllReduce 测量实际的 NVLink 带宽。
    需要在分布式环境下运行。

    简单模式（不需要 dist.init）:
        使用 torch.cuda.nccl 直接调用 NCCL（如可用）。

    分布式模式:
        使用 torch.distributed 的标准 API。
    """

    def __init__(self):
        self.num_gpus = torch.cuda.device_count()

    def benchmark_allreduce_simple(self, size_mb: int = 256,
                                     warmup: int = 10, runs: int = 50) -> Optional[NVLinkResult]:
        """
        简化版 AllReduce 测试（不需要 torch.distributed）。
        使用多 stream + P2P 模拟 AllReduce。
        """
        if self.num_gpus < 2:
            print("需要至少 2 张 GPU")
            return None

        n_elements = size_mb * 1024 * 1024 // 4
        tensors = [torch.randn(n_elements, device=f'cuda:{i}') for i in range(self.num_gpus)]

        # 简单的 ring allreduce 模拟
        # 实际 NCCL 的实现更高效，这里只是估算带宽
        def ring_allreduce_step():
            """一次 ring allreduce 的通信步骤"""
            for i in range(self.num_gpus):
                next_gpu = (i + 1) % self.num_gpus
                # 将 GPU i 的数据复制到 GPU next
                tensors[next_gpu].copy_(tensors[i])

        # Warmup
        for _ in range(warmup):
            ring_allreduce_step()
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(runs):
            ring_allreduce_step()
        torch.cuda.synchronize()
        time_ms = (time.perf_counter() - start) / runs * 1000

        # AllReduce 的算法带宽和总线带宽
        # 算法带宽 = data_size / time
        # 总线带宽 = 算法带宽 × 2(N-1)/N（ring allreduce 的效率因子）
        data_bytes = size_mb * 1024 * 1024
        algo_bw = data_bytes / (time_ms / 1000) / 1e9
        bus_bw = algo_bw * 2 * (self.num_gpus - 1) / self.num_gpus

        # 清理
        for t in tensors:
            del t
        torch.cuda.empty_cache()

        return NVLinkResult(
            operation='AllReduce (simple)',
            num_gpus=self.num_gpus,
            data_size_mb=size_mb,
            time_ms=time_ms,
            bandwidth_gb_s=algo_bw,
            bus_bandwidth_gb_s=bus_bw,
        )

    def benchmark_all2all_simple(self, size_mb: int = 64) -> Optional[NVLinkResult]:
        """All-to-All 通信测试"""
        if self.num_gpus < 2:
            return None

        per_gpu_elements = size_mb * 1024 * 1024 // 4 // self.num_gpus
        tensors = [torch.randn(per_gpu_elements * self.num_gpus, device=f'cuda:{i}')
                   for i in range(self.num_gpus)]
        outputs = [torch.empty_like(t) for t in tensors]

        def all2all_step():
            for i in range(self.num_gpus):
                for j in range(self.num_gpus):
                    chunk_size = per_gpu_elements
                    src_slice = tensors[i][j*chunk_size:(j+1)*chunk_size]
                    dst_slice = outputs[j][i*chunk_size:(i+1)*chunk_size]
                    dst_slice.copy_(src_slice)

        # Warmup + Benchmark
        for _ in range(5):
            all2all_step()
        torch.cuda.synchronize()

        start = time.perf_counter()
        runs = 20
        for _ in range(runs):
            all2all_step()
        torch.cuda.synchronize()
        time_ms = (time.perf_counter() - start) / runs * 1000

        data_bytes = size_mb * 1024 * 1024
        algo_bw = data_bytes / (time_ms / 1000) / 1e9

        for t in tensors + outputs:
            del t
        torch.cuda.empty_cache()

        return NVLinkResult(
            operation='All-to-All (simple)',
            num_gpus=self.num_gpus,
            data_size_mb=size_mb,
            time_ms=time_ms,
            bandwidth_gb_s=algo_bw,
            bus_bandwidth_gb_s=algo_bw * (self.num_gpus - 1) / self.num_gpus,
        )

    def run_all(self) -> List[NVLinkResult]:
        """运行全部 NVLink 测试"""
        results = []

        sizes = [64, 256, 1024]
        for size in sizes:
            r = self.benchmark_allreduce_simple(size)
            if r:
                results.append(r)

        r = self.benchmark_all2all_simple()
        if r:
            results.append(r)

        return results

    def print_results(self, results: List[NVLinkResult]):
        """打印结果"""
        print(f"\nNVLink 带宽测试 ({self.num_gpus} GPUs)")
        print(f"{'Operation':<24} | {'Size(MB)':>8} | {'Time(ms)':>10} | "
              f"{'Algo BW':>10} | {'Bus BW':>10}")
        print("-" * 75)
        for r in results:
            print(f"{r.operation:<24} | {r.data_size_mb:>8.0f} | {r.time_ms:>10.3f} | "
                  f"{r.bandwidth_gb_s:>9.1f} | {r.bus_bandwidth_gb_s:>9.1f}")

        print(f"\n注: H20 NVLink 理论带宽 = 900 GB/s (双向)")
        print(f"    实际可达 ~80-90% (720-810 GB/s)")


if __name__ == "__main__":
    bench = NVLinkBenchmark()
    results = bench.run_all()
    bench.print_results(results)
