"""
GPU 计算性能基准测试

测试不同精度下的实际算力 (TFLOPS)：
- FP32 (CUDA Core)
- FP16 (Tensor Core)
- INT8 (Tensor Core)

原理：运行大矩阵乘法，测量时间，计算 TFLOPS。
大矩阵乘法是 compute-bound 的，其性能接近硬件算力上限。
"""

import torch
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ComputeResult:
    """单次计算基准测试结果"""
    dtype: str
    M: int
    N: int
    K: int
    time_ms: float
    tflops: float
    peak_tflops: float  # 理论峰值
    utilization: float  # 利用率百分比


class ComputeBenchmark:
    """
    GPU 计算基准测试。

    使用大矩阵乘法来测试峰值算力。
    矩阵足够大时（M=N=K>=4096），GEMM 是 compute-bound，
    实测性能接近硬件算力上限。
    """

    # H20 理论峰值（可通过配置覆盖）
    DEFAULT_PEAKS = {
        'fp32': 44.0,      # TFLOPS
        'tf32': 74.0,      # TFLOPS
        'fp16': 148.0,     # TFLOPS
        'bf16': 148.0,     # TFLOPS
        'int8': 296.0,     # TOPS
    }

    def __init__(self, device_id: int = 0, peaks: Optional[Dict[str, float]] = None):
        self.device = torch.device(f'cuda:{device_id}')
        self.device_id = device_id
        self.peaks = peaks or self.DEFAULT_PEAKS
        self.device_name = torch.cuda.get_device_name(device_id)

    def _benchmark_gemm(self, M: int, N: int, K: int, dtype: torch.dtype,
                        warmup: int = 10, runs: int = 50) -> float:
        """执行 GEMM 并返回平均时间 (ms)"""
        A = torch.randn(M, K, device=self.device, dtype=dtype)
        B = torch.randn(K, N, device=self.device, dtype=dtype)

        # Warmup
        for _ in range(warmup):
            _ = torch.matmul(A, B)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(runs):
            _ = torch.matmul(A, B)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / runs * 1000

        del A, B
        torch.cuda.empty_cache()
        return elapsed

    def benchmark_fp32(self, sizes: List[int] = None) -> List[ComputeResult]:
        """FP32 (TF32 on Ampere+) 计算基准"""
        if sizes is None:
            sizes = [2048, 4096, 8192]

        results = []
        for size in sizes:
            M = N = K = size
            time_ms = self._benchmark_gemm(M, N, K, torch.float32)
            flops = 2 * M * N * K
            tflops = flops / (time_ms / 1000) / 1e12

            # PyTorch 默认开启 TF32，所以实际是 TF32 性能
            peak = self.peaks.get('tf32', self.peaks['fp32'])

            results.append(ComputeResult(
                dtype='fp32 (tf32)',
                M=M, N=N, K=K,
                time_ms=time_ms,
                tflops=tflops,
                peak_tflops=peak,
                utilization=tflops / peak * 100,
            ))
        return results

    def benchmark_fp16(self, sizes: List[int] = None) -> List[ComputeResult]:
        """FP16 Tensor Core 计算基准"""
        if sizes is None:
            sizes = [2048, 4096, 8192]

        results = []
        for size in sizes:
            M = N = K = size
            time_ms = self._benchmark_gemm(M, N, K, torch.float16)
            flops = 2 * M * N * K
            tflops = flops / (time_ms / 1000) / 1e12
            peak = self.peaks['fp16']

            results.append(ComputeResult(
                dtype='fp16',
                M=M, N=N, K=K,
                time_ms=time_ms,
                tflops=tflops,
                peak_tflops=peak,
                utilization=tflops / peak * 100,
            ))
        return results

    def benchmark_int8(self, sizes: List[int] = None) -> List[ComputeResult]:
        """INT8 Tensor Core 计算基准"""
        if sizes is None:
            sizes = [2048, 4096, 8192]

        results = []
        for size in sizes:
            M = N = K = size
            A = torch.randint(-128, 127, (M, K), device=self.device, dtype=torch.int8)
            B = torch.randint(-128, 127, (K, N), device=self.device, dtype=torch.int8)

            # INT8 GEMM 需要通过 torch._int_mm 或 cuBLAS INT8
            # 这里用 FP16 模拟 INT8 的吞吐量估算
            A_fp16 = A.to(torch.float16)
            B_fp16 = B.to(torch.float16)

            # Warmup
            for _ in range(10):
                _ = torch.matmul(A_fp16, B_fp16)
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(50):
                _ = torch.matmul(A_fp16, B_fp16)
            torch.cuda.synchronize()
            time_ms = (time.perf_counter() - start) / 50 * 1000

            ops = 2 * M * N * K
            tops = ops / (time_ms / 1000) / 1e12
            peak = self.peaks['int8']

            results.append(ComputeResult(
                dtype='int8 (simulated)',
                M=M, N=N, K=K,
                time_ms=time_ms,
                tflops=tops,
                peak_tflops=peak,
                utilization=tops / peak * 100,
            ))

            del A, B, A_fp16, B_fp16
            torch.cuda.empty_cache()

        return results

    def run_all(self, sizes: List[int] = None) -> Dict[str, List[ComputeResult]]:
        """运行全部计算基准"""
        return {
            'fp32': self.benchmark_fp32(sizes),
            'fp16': self.benchmark_fp16(sizes),
            'int8': self.benchmark_int8(sizes),
        }

    def print_results(self, results: Dict[str, List[ComputeResult]]):
        """打印结果"""
        print(f"\nGPU {self.device_id}: {self.device_name}")
        print(f"{'Dtype':<16} | {'Size':>8} | {'Time(ms)':>10} | {'TFLOPS':>8} | {'Peak':>8} | {'Util%':>6}")
        print("-" * 70)
        for dtype, res_list in results.items():
            for r in res_list:
                print(f"{r.dtype:<16} | {r.M:>8} | {r.time_ms:>10.3f} | "
                      f"{r.tflops:>8.2f} | {r.peak_tflops:>8.1f} | {r.utilization:>5.1f}%")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("需要 CUDA GPU")
        exit(1)

    bench = ComputeBenchmark(device_id=0)
    results = bench.run_all()
    bench.print_results(results)
