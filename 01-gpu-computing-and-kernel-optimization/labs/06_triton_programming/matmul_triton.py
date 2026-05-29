"""
Lab 06: Triton 矩阵乘法

展示 Triton 的 tiling 编程模式和 autotune 功能。
这是理解 GPU 高性能计算的核心 pattern。

Usage: python matmul_triton.py
"""

import torch
import triton
import triton.language as tl
import time


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=2, num_stages=5),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # 矩阵指针
    a_ptr, b_ptr, c_ptr,
    # 矩阵维度
    M, N, K,
    # 矩阵 stride（行主序中 stride_row = 列数）
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # 编译期常量
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Triton 矩阵乘法 kernel。

    C[M×N] = A[M×K] × B[K×N]

    每个程序实例计算 C 中一个 BLOCK_M × BLOCK_N 的 tile。
    沿 K 维度循环，每次加载 A[BLOCK_M×BLOCK_K] 和 B[BLOCK_K×BLOCK_N]。

    关键理解：
    - 2D Grid: 每个 program 处理 C 的一个 tile
    - K 维度内层循环: 类似 CUDA 的 tiled GEMM
    - Triton 自动处理共享内存和 Tensor Core
    """
    # ---- 计算当前 program 负责的 C tile 位置 ----
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # L2 缓存友好的 tile 排列（swizzle）
    # 让相邻的 program 处理空间上接近的 tile，提高 L2 命中率
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- 计算偏移量 ----
    # A tile 的行偏移: [pid_m * BLOCK_M, pid_m * BLOCK_M + BLOCK_M)
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # B tile 的列偏移: [pid_n * BLOCK_N, pid_n * BLOCK_N + BLOCK_N)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # K 维度偏移（初始）
    offs_k = tl.arange(0, BLOCK_K)

    # 构造 A 和 B 的指针矩阵
    # a_ptrs[i][k] 指向 A[offs_am[i], offs_k[k]]
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # ---- 初始化累加器 ----
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ---- K 维度循环 ----
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # 加载 A tile [BLOCK_M, BLOCK_K]
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] + k * BLOCK_K < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # 加载 B tile [BLOCK_K, BLOCK_N]
        b_mask = (offs_k[:, None] + k * BLOCK_K < K) & (offs_bn[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # 矩阵乘加 — Triton 编译器会自动使用 Tensor Core
        accumulator += tl.dot(a, b)

        # 移动 K 维度的指针
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ---- 存储结果 ----
    c = accumulator.to(tl.float16)  # 转回 FP16

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Triton 矩阵乘法封装"""
    assert a.shape[1] == b.shape[0], "维度不匹配"
    assert a.is_contiguous() and b.is_contiguous()
    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def benchmark():
    """性能对比"""
    print("=" * 60)
    print("Triton 矩阵乘法 vs PyTorch (cuBLAS)")
    print("=" * 60)

    sizes = [512, 1024, 2048, 4096]

    print(f"\n{'M=N=K':>8} | {'PyTorch(ms)':>12} | {'Triton(ms)':>12} | {'比值':>8} | {'TFLOPS':>8}")
    print("-" * 60)

    for size in sizes:
        M, N, K = size, size, size
        a = torch.randn((M, K), device='cuda', dtype=torch.float16)
        b = torch.randn((K, N), device='cuda', dtype=torch.float16)

        # Warmup
        for _ in range(5):
            _ = torch.matmul(a, b)
            _ = matmul_triton(a, b)
        torch.cuda.synchronize()

        # Benchmark PyTorch
        runs = 20
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        pytorch_ms = (time.perf_counter() - start) / runs * 1000

        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            _ = matmul_triton(a, b)
        torch.cuda.synchronize()
        triton_ms = (time.perf_counter() - start) / runs * 1000

        # 计算 TFLOPS
        flops = 2 * M * N * K
        triton_tflops = flops / (triton_ms / 1000) / 1e12
        ratio = pytorch_ms / triton_ms

        print(f"{size:>8} | {pytorch_ms:>12.3f} | {triton_ms:>12.3f} | {ratio:>7.2f}x | {triton_tflops:>7.1f}")

    print(f"\n观察:")
    print(f"  - Triton GEMM 通常能达到 cuBLAS 性能的 85-95%")
    print(f"  - 大矩阵更接近（GPU 更充分利用）")
    print(f"  - Triton 的优势: 50 行 Python vs 1000 行 CUDA")


def correctness_test():
    """正确性验证"""
    print("\n正确性验证:")
    torch.manual_seed(42)

    M, N, K = 512, 512, 512
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)

    triton_result = matmul_triton(a, b)
    torch_result = torch.matmul(a, b)

    # FP16 精度下允许较大的绝对误差
    max_diff = (triton_result - torch_result).abs().max().item()
    relative_diff = max_diff / torch_result.abs().max().item()
    print(f"  最大绝对误差: {max_diff:.4e}")
    print(f"  最大相对误差: {relative_diff:.4e}")
    print(f"  状态: {'PASS' if relative_diff < 0.01 else 'FAIL'}")


if __name__ == "__main__":
    correctness_test()
    benchmark()
