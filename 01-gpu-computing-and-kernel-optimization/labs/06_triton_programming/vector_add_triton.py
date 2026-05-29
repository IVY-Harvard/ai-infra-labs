"""
Lab 06: Triton 向量加法

最简单的 Triton kernel，展示核心编程模型：
- 块级编程思维（不是线程级）
- 自动内存管理
- mask 机制处理边界

Usage: python vector_add_triton.py
"""

import torch
import triton
import triton.language as tl
import time


@triton.jit
def vector_add_kernel(
    # 指针参数：Triton 中的张量通过指针传递
    a_ptr,
    b_ptr,
    output_ptr,
    # 元数据
    n_elements,
    # 编译期常量：BLOCK_SIZE 在编译时确定，不同值生成不同的二进制
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton 向量加法 kernel。

    关键区别于 CUDA：
    - 你操作的是"数据块"而非"单个线程"
    - tl.arange 生成一组偏移量，一次处理 BLOCK_SIZE 个元素
    - mask 自动处理边界，不需要手动 if (idx < n)
    """
    # 获取当前程序实例的 ID（类似 CUDA 的 blockIdx）
    # 每个 program_id 处理 BLOCK_SIZE 个元素
    pid = tl.program_id(axis=0)

    # 计算这个程序实例负责的元素偏移量
    # tl.arange(0, BLOCK_SIZE) 返回 [0, 1, 2, ..., BLOCK_SIZE-1]
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 创建 mask：处理最后一个块可能越界的情况
    mask = offsets < n_elements

    # 从 HBM 加载数据（mask=False 的位置不会实际读取）
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    # 计算（在寄存器中完成）
    output = a + b

    # 写回 HBM
    tl.store(output_ptr + offsets, output, mask=mask)


def vector_add_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Triton 向量加法的 Python 封装"""
    # 确保输入在 GPU 上且连续
    assert a.is_cuda and b.is_cuda
    assert a.is_contiguous() and b.is_contiguous()
    assert a.shape == b.shape

    output = torch.empty_like(a)
    n_elements = output.numel()

    # 计算 grid 大小（需要多少个程序实例）
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # 启动 kernel
    vector_add_kernel[grid](a, b, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return output


def benchmark():
    """性能基准测试"""
    print("=" * 60)
    print("Triton 向量加法 vs PyTorch")
    print("=" * 60)

    sizes = [2**i for i in range(16, 26)]

    print(f"\n{'N':>12} | {'PyTorch(us)':>12} | {'Triton(us)':>12} | {'相对性能':>10}")
    print("-" * 55)

    for n in sizes:
        a = torch.randn(n, device='cuda', dtype=torch.float32)
        b = torch.randn(n, device='cuda', dtype=torch.float32)

        # Warmup
        for _ in range(10):
            _ = a + b
            _ = vector_add_triton(a, b)
        torch.cuda.synchronize()

        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            _ = a + b
        torch.cuda.synchronize()
        pytorch_us = (time.perf_counter() - start) / 100 * 1e6

        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            _ = vector_add_triton(a, b)
        torch.cuda.synchronize()
        triton_us = (time.perf_counter() - start) / 100 * 1e6

        ratio = pytorch_us / triton_us
        print(f"{n:>12,} | {pytorch_us:>12.1f} | {triton_us:>12.1f} | {ratio:>9.2f}x")

    print(f"\n注意:")
    print(f"  - 向量加法太简单，Triton 和 PyTorch 性能几乎相同")
    print(f"  - 两者都是 memory-bound，瓶颈都在 HBM 带宽")
    print(f"  - Triton 的优势在更复杂的融合 kernel 中才体现")


def correctness_test():
    """正确性验证"""
    print("\n正确性验证:")
    torch.manual_seed(42)
    a = torch.randn(10007, device='cuda')  # 非对齐大小测试边界处理
    b = torch.randn(10007, device='cuda')

    triton_result = vector_add_triton(a, b)
    torch_result = a + b

    max_diff = (triton_result - torch_result).abs().max().item()
    print(f"  最大误差: {max_diff:.2e}")
    print(f"  状态: {'PASS' if max_diff < 1e-6 else 'FAIL'}")


if __name__ == "__main__":
    correctness_test()
    benchmark()
