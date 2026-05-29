"""
Lab 06: Triton Fused Softmax

展示 Triton 在算子融合上的优势：
- 一个 kernel 完成整个 softmax（而不是分成 max, sub, exp, sum, div）
- 中间结果在 SRAM 中完成，不写 HBM

Usage: python fused_softmax.py
"""

import torch
import triton
import triton.language as tl
import time


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,  # 一行有多少元素
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused Softmax Kernel。

    每个程序实例处理输入矩阵的一行。
    整个 softmax（max, sub, exp, sum, div）在一个 kernel 中完成。

    为什么比 PyTorch 快：
    1. PyTorch 的 softmax 分成多个 kernel（max, sub, exp, sum, div）
    2. 每个 kernel 都要读写 HBM
    3. Triton: 所有操作在 SRAM 中完成，只读写 HBM 一次
    """
    # 每个 program 处理一行
    row_idx = tl.program_id(0)

    # 计算这一行的起始地址
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # 加载整行数据到 SRAM
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))

    # ---- 以下全在 SRAM 中计算，不访问 HBM ----

    # Step 1: 减去最大值（数值稳定性）
    row_max = tl.max(row, axis=0)
    row = row - row_max

    # Step 2: 计算 exp
    numerator = tl.exp(row)

    # Step 3: 计算 sum
    denominator = tl.sum(numerator, axis=0)

    # Step 4: 归一化
    softmax_output = numerator / denominator

    # ---- 写回 HBM（只写一次） ----
    output_row_start_ptr = output_ptr + row_idx * input_row_stride
    tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)


def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    """Triton Softmax 封装"""
    n_rows, n_cols = x.shape

    # BLOCK_SIZE 必须是 2 的幂且 >= n_cols
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # 限制 BLOCK_SIZE 防止超出共享内存
    # 每个 block 需要 BLOCK_SIZE * 4 bytes (FP32)
    BLOCK_SIZE = min(BLOCK_SIZE, 8192)

    output = torch.empty_like(x)

    # Grid: 每行一个 program 实例
    grid = (n_rows,)

    softmax_kernel[grid](
        output, x,
        x.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def benchmark():
    """性能对比"""
    print("=" * 60)
    print("Fused Softmax: Triton vs PyTorch")
    print("=" * 60)

    configs = [
        (1024, 256),
        (4096, 512),
        (4096, 1024),
        (4096, 2048),
        (4096, 4096),
        (8192, 4096),
    ]

    print(f"\n{'shape':>16} | {'PyTorch(us)':>12} | {'Triton(us)':>12} | {'加速比':>8}")
    print("-" * 55)

    for rows, cols in configs:
        x = torch.randn(rows, cols, device='cuda', dtype=torch.float32)

        # Warmup
        for _ in range(10):
            _ = torch.softmax(x, dim=-1)
            _ = softmax_triton(x)
        torch.cuda.synchronize()

        # PyTorch
        runs = 100
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            _ = torch.softmax(x, dim=-1)
        torch.cuda.synchronize()
        pytorch_us = (time.perf_counter() - start) / runs * 1e6

        # Triton
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            _ = softmax_triton(x)
        torch.cuda.synchronize()
        triton_us = (time.perf_counter() - start) / runs * 1e6

        speedup = pytorch_us / triton_us
        shape_str = f"({rows},{cols})"
        print(f"{shape_str:>16} | {pytorch_us:>12.1f} | {triton_us:>12.1f} | {speedup:>7.2f}x")

    print(f"\n观察:")
    print(f"  - Triton Fused Softmax 通常比 PyTorch 快 1.5-3x")
    print(f"  - 原因: PyTorch 的 softmax 分多步(多个 kernel)，每步读写 HBM")
    print(f"  - Triton: 一个 kernel 完成所有操作，数据全程在 SRAM")
    print(f"  - 这就是算子融合的典型收益")


def correctness_test():
    """正确性验证"""
    print("\n正确性验证:")
    torch.manual_seed(42)
    x = torch.randn(1024, 768, device='cuda', dtype=torch.float32)

    triton_result = softmax_triton(x)
    torch_result = torch.softmax(x, dim=-1)

    max_diff = (triton_result - torch_result).abs().max().item()
    print(f"  最大误差: {max_diff:.2e}")
    print(f"  状态: {'PASS' if max_diff < 1e-5 else 'FAIL'}")

    # 验证 softmax 性质: 每行和为 1
    row_sums = triton_result.sum(dim=-1)
    sum_error = (row_sums - 1.0).abs().max().item()
    print(f"  行和误差: {sum_error:.2e} (应接近 0)")


if __name__ == "__main__":
    correctness_test()
    benchmark()
