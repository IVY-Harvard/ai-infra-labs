"""
Lab 10: 自动化 Roofline 分析

对常见 AI 算子做 Roofline 分析，自动判断瓶颈类型。
不需要 NSight（纯 PyTorch 测量），适合快速评估。

Usage: python roofline_analysis.py
"""

import torch
import torch.nn.functional as F
import time
from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass
class RooflineResult:
    """Roofline 分析结果"""
    name: str
    time_ms: float
    flops: int
    bytes_accessed: int
    arithmetic_intensity: float
    achieved_tflops: float
    achieved_bandwidth_gb_s: float
    bottleneck: str  # "compute" or "memory"


class RooflineAnalyzer:
    """
    Roofline 分析器。

    通过测量实际执行时间，结合理论 FLOPs 和 bytes，
    判断每个算子的瓶颈类型。
    """

    def __init__(self, device='cuda:0'):
        self.device = torch.device(device)

        # H20 理论峰值（可根据实际 GPU 修改）
        self.peak_fp16_tflops = 148.0  # FP16 Tensor Core
        self.peak_fp32_tflops = 44.0   # FP32 CUDA Core
        self.peak_bandwidth_tb_s = 4.0  # HBM3 带宽

        # 机器平衡点
        self.balance_fp16 = self.peak_fp16_tflops * 1000 / (self.peak_bandwidth_tb_s * 1000)
        self.balance_fp32 = self.peak_fp32_tflops * 1000 / (self.peak_bandwidth_tb_s * 1000)

        print(f"GPU 理论规格:")
        print(f"  FP16 Tensor: {self.peak_fp16_tflops} TFLOPS")
        print(f"  FP32 CUDA:   {self.peak_fp32_tflops} TFLOPS")
        print(f"  HBM 带宽:    {self.peak_bandwidth_tb_s} TB/s")
        print(f"  平衡点 (FP16): {self.balance_fp16:.1f} FLOP/Byte")
        print(f"  平衡点 (FP32): {self.balance_fp32:.1f} FLOP/Byte\n")

    def measure_time(self, fn: Callable, warmup=10, runs=50) -> float:
        """测量执行时间 (ms)"""
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / runs * 1000

    def analyze(self, name: str, fn: Callable, flops: int, bytes_accessed: int,
                dtype_is_fp16: bool = False) -> RooflineResult:
        """分析一个算子"""
        time_ms = self.measure_time(fn)

        ai = flops / bytes_accessed if bytes_accessed > 0 else float('inf')
        achieved_tflops = flops / (time_ms / 1000) / 1e12
        achieved_bw = bytes_accessed / (time_ms / 1000) / 1e9

        balance = self.balance_fp16 if dtype_is_fp16 else self.balance_fp32
        bottleneck = "compute" if ai > balance else "memory"

        return RooflineResult(
            name=name,
            time_ms=time_ms,
            flops=flops,
            bytes_accessed=bytes_accessed,
            arithmetic_intensity=ai,
            achieved_tflops=achieved_tflops,
            achieved_bandwidth_gb_s=achieved_bw,
            bottleneck=bottleneck,
        )


def analyze_common_ops():
    """分析常见 AI 算子的 Roofline 特性"""
    device = 'cuda:0'
    analyzer = RooflineAnalyzer(device)

    results = []

    # ---- 1. Vector Add (极度 memory-bound) ----
    N = 1 << 24  # 16M
    a = torch.randn(N, device=device, dtype=torch.float32)
    b = torch.randn(N, device=device, dtype=torch.float32)
    c = torch.empty_like(a)

    results.append(analyzer.analyze(
        "Vector Add (FP32, 16M)",
        lambda: torch.add(a, b, out=c),
        flops=N,  # N 次加法
        bytes_accessed=3 * N * 4,  # 读 2 + 写 1
    ))

    # ---- 2. ReLU (memory-bound) ----
    x = torch.randn(4096, 4096, device=device)
    results.append(analyzer.analyze(
        "ReLU (4096x4096, FP32)",
        lambda: F.relu(x),
        flops=4096 * 4096,  # 一次比较
        bytes_accessed=2 * 4096 * 4096 * 4,  # 读 + 写
    ))

    # ---- 3. LayerNorm (memory-bound) ----
    x_ln = torch.randn(32, 512, 4096, device=device)
    weight = torch.ones(4096, device=device)
    bias = torch.zeros(4096, device=device)
    results.append(analyzer.analyze(
        "LayerNorm (32,512,4096)",
        lambda: F.layer_norm(x_ln, [4096], weight, bias),
        flops=32 * 512 * 4096 * 5,  # mean, var, normalize, scale, shift
        bytes_accessed=2 * 32 * 512 * 4096 * 4 + 2 * 4096 * 4,
    ))

    # ---- 4. GEMM 大矩阵 (compute-bound) ----
    M, N_g, K = 4096, 4096, 4096
    a_fp16 = torch.randn(M, K, device=device, dtype=torch.float16)
    b_fp16 = torch.randn(K, N_g, device=device, dtype=torch.float16)
    results.append(analyzer.analyze(
        "GEMM (4096x4096x4096, FP16)",
        lambda: torch.matmul(a_fp16, b_fp16),
        flops=2 * M * N_g * K,
        bytes_accessed=(M*K + K*N_g + M*N_g) * 2,  # FP16 = 2 bytes
        dtype_is_fp16=True,
    ))

    # ---- 5. GEMM 小 M (decode, memory-bound) ----
    M_small = 1
    a_small = torch.randn(M_small, K, device=device, dtype=torch.float16)
    results.append(analyzer.analyze(
        "GEMV (1x4096x4096, FP16)",
        lambda: torch.matmul(a_small, b_fp16),
        flops=2 * M_small * N_g * K,
        bytes_accessed=(M_small*K + K*N_g + M_small*N_g) * 2,
        dtype_is_fp16=True,
    ))

    # ---- 6. Softmax (memory-bound) ----
    x_soft = torch.randn(32, 64, 2048, device=device)
    results.append(analyzer.analyze(
        "Softmax (32,64,2048)",
        lambda: F.softmax(x_soft, dim=-1),
        flops=32 * 64 * 2048 * 5,  # max, sub, exp, sum, div
        bytes_accessed=2 * 32 * 64 * 2048 * 4,
    ))

    # ---- 7. Attention (varies) ----
    q = torch.randn(4, 32, 512, 128, device=device, dtype=torch.float16)
    k = torch.randn(4, 32, 512, 128, device=device, dtype=torch.float16)
    v = torch.randn(4, 32, 512, 128, device=device, dtype=torch.float16)
    B, H, N_seq, D = q.shape
    results.append(analyzer.analyze(
        "Attention (B4,H32,N512,D128)",
        lambda: F.scaled_dot_product_attention(q, k, v),
        flops=B * H * (2 * N_seq * N_seq * D + 2 * N_seq * N_seq * D),
        bytes_accessed=B * H * (3 * N_seq * D) * 2,  # Q,K,V + O
        dtype_is_fp16=True,
    ))

    # ---- 打印结果 ----
    print("=" * 90)
    print(f"{'算子':<30} | {'时间(ms)':>8} | {'AI':>8} | {'TFLOPS':>8} | {'BW(GB/s)':>10} | {'瓶颈':>8}")
    print("-" * 90)

    for r in results:
        print(f"{r.name:<30} | {r.time_ms:>8.3f} | {r.arithmetic_intensity:>8.1f} | "
              f"{r.achieved_tflops:>8.2f} | {r.achieved_bandwidth_gb_s:>10.0f} | {r.bottleneck:>8}")

    print("-" * 90)
    print(f"\n关键发现:")
    print(f"  - Vector Add/ReLU: AI < 1, 极度 memory-bound")
    print(f"  - 大 GEMM: AI >> 平衡点, compute-bound (Tensor Core 发挥作用)")
    print(f"  - GEMV (decode): AI ≈ 1, memory-bound (这就是 decode 慢的原因)")
    print(f"  - Attention: 取决于序列长度和 batch size")


def main():
    if not torch.cuda.is_available():
        print("需要 CUDA GPU")
        return

    print("Lab 10: Roofline 分析\n")
    analyze_common_ops()


if __name__ == "__main__":
    main()
