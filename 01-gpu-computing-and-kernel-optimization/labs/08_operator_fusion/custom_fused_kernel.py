"""
Lab 08: 手写融合 Kernel

用 Triton 手写一个融合了 Residual Add + RMSNorm 的 kernel，
对比自动融合和手动融合的效果。

RMSNorm 是 LLaMA 等模型使用的归一化方法，比 LayerNorm 更简单：
RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight

Usage: python custom_fused_kernel.py
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import time
import math


# ============================================================
# 手写 Triton Fused Kernel: Residual Add + RMSNorm
# ============================================================

@triton.jit
def fused_residual_rmsnorm_kernel(
    # 输入输出指针
    output_ptr,
    input_ptr,
    residual_ptr,
    weight_ptr,
    # 维度
    N,  # hidden_size (每行的元素数)
    eps,
    # 步长
    input_row_stride,
    output_row_stride,
    # 编译期常量
    BLOCK_SIZE: tl.constexpr,
):
    """
    融合 Kernel: output = RMSNorm(input + residual) * weight

    每个 program 处理一行（一个 token 的 hidden_dim）。

    融合的好处:
    1. input + residual 的结果不写 HBM，直接在寄存器中做 RMSNorm
    2. RMSNorm 的中间结果（mean(x²)）也不写 HBM
    3. 最终只读 3 个张量（input, residual, weight），写 1 个（output）
    """
    row_idx = tl.program_id(0)

    # 偏移量
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # 计算行的起始位置
    input_row_ptr = input_ptr + row_idx * input_row_stride
    residual_row_ptr = residual_ptr + row_idx * input_row_stride
    output_row_ptr = output_ptr + row_idx * output_row_stride

    # Step 1: 加载 input 和 residual，在寄存器中相加
    x = tl.load(input_row_ptr + col_offsets, mask=mask, other=0.0)
    res = tl.load(residual_row_ptr + col_offsets, mask=mask, other=0.0)
    x = x + res  # Residual Add（在寄存器中！不写 HBM）

    # Step 2: 计算 RMS = sqrt(mean(x²) + eps)
    x_sq = x * x
    mean_sq = tl.sum(x_sq, axis=0) / N
    rms = tl.sqrt(mean_sq + eps)

    # Step 3: Normalize
    x_norm = x / rms

    # Step 4: Scale
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    output = x_norm * weight

    # Step 5: 写回 HBM（整个过程只写这一次）
    tl.store(output_row_ptr + col_offsets, output, mask=mask)


def fused_residual_rmsnorm_triton(x, residual, weight, eps=1e-6):
    """Triton 融合 kernel 封装"""
    assert x.shape == residual.shape
    *batch_dims, N = x.shape

    # 展平成 2D
    x_2d = x.reshape(-1, N)
    res_2d = residual.reshape(-1, N)
    output = torch.empty_like(x_2d)

    n_rows = x_2d.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(N)
    BLOCK_SIZE = min(BLOCK_SIZE, 8192)

    fused_residual_rmsnorm_kernel[(n_rows,)](
        output, x_2d, res_2d, weight,
        N, eps,
        x_2d.stride(0), output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.reshape(x.shape)


# ============================================================
# 非融合版本（对比用）
# ============================================================

def rmsnorm_pytorch(x, weight, eps=1e-6):
    """PyTorch 原生 RMSNorm"""
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def unfused_residual_rmsnorm(x, residual, weight, eps=1e-6):
    """非融合版本: 分步执行"""
    x = x + residual  # Kernel 1: 写 HBM
    x = rmsnorm_pytorch(x, weight, eps)  # Kernels 2-4: 多次读写 HBM
    return x


def benchmark():
    """性能对比"""
    print("=" * 60)
    print("Fused Residual + RMSNorm: 手写 Triton vs PyTorch")
    print("=" * 60)

    configs = [
        (8, 512, 4096, "batch=8, seq=512, hidden=4096"),
        (32, 512, 4096, "batch=32, seq=512, hidden=4096"),
        (4, 2048, 4096, "batch=4, seq=2048, hidden=4096"),
    ]

    for batch, seq, hidden, desc in configs:
        print(f"\n  {desc}")
        x = torch.randn(batch, seq, hidden, device='cuda', dtype=torch.float32)
        res = torch.randn_like(x)
        weight = torch.ones(hidden, device='cuda', dtype=torch.float32)

        # Warmup
        for _ in range(10):
            _ = unfused_residual_rmsnorm(x, res, weight)
            _ = fused_residual_rmsnorm_triton(x, res, weight)
        torch.cuda.synchronize()

        # Unfused PyTorch
        runs = 100
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            _ = unfused_residual_rmsnorm(x, res, weight)
        torch.cuda.synchronize()
        unfused_ms = (time.perf_counter() - start) / runs * 1000

        # Fused Triton
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            _ = fused_residual_rmsnorm_triton(x, res, weight)
        torch.cuda.synchronize()
        fused_ms = (time.perf_counter() - start) / runs * 1000

        # torch.compile
        compiled_fn = torch.compile(unfused_residual_rmsnorm)
        for _ in range(10):
            _ = compiled_fn(x, res, weight)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            _ = compiled_fn(x, res, weight)
        torch.cuda.synchronize()
        compiled_ms = (time.perf_counter() - start) / runs * 1000

        print(f"    Unfused PyTorch:  {unfused_ms:.3f} ms")
        print(f"    torch.compile:    {compiled_ms:.3f} ms (auto-fuse)")
        print(f"    Triton (手写):    {fused_ms:.3f} ms (manual fuse)")
        print(f"    手写 vs unfused:  {unfused_ms/fused_ms:.2f}x 加速")
        print(f"    手写 vs compile:  {compiled_ms/fused_ms:.2f}x")


def correctness_test():
    """验证手写 kernel 的正确性"""
    print("\n正确性验证:")
    torch.manual_seed(42)

    batch, seq, hidden = 4, 64, 1024
    x = torch.randn(batch, seq, hidden, device='cuda')
    res = torch.randn_like(x)
    weight = torch.randn(hidden, device='cuda')

    ref = unfused_residual_rmsnorm(x, res, weight)
    out = fused_residual_rmsnorm_triton(x, res, weight)

    max_diff = (ref - out).abs().max().item()
    print(f"  最大误差: {max_diff:.2e}")
    print(f"  状态: {'PASS' if max_diff < 1e-4 else 'FAIL'}")


if __name__ == "__main__":
    correctness_test()
    benchmark()
