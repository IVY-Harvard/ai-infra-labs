"""
Lab 08 - 通信-计算重叠 (Overlap)
==================================
演示如何使用异步通信实现 computation-communication overlap。

核心技术:
  1. 异步 AllReduce: 在计算下一个 bucket 时，上一个 bucket 在后台通信
  2. CUDA Stream: 将通信和计算放在不同的 stream 上
  3. Prefetch: 提前发起下一层的参数收集

运行:
    torchrun --nproc_per_node=4 overlap_comm_compute.py
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return rank, world_size, device


def demo_async_allreduce(rank, world_size, device):
    """
    演示异步 AllReduce 如何与计算重叠。
    """
    if rank == 0:
        print("=" * 60)
        print("Demo 1: 异步 AllReduce Overlap")
        print("=" * 60)

    num_layers = 8
    tensor_size = 4 * 1024 * 1024  # 4M elements per layer
    compute_size = 1024

    # 模拟多层网络的梯度
    gradients = [torch.randn(tensor_size, device=device) for _ in range(num_layers)]
    compute_matrix = torch.randn(compute_size, compute_size, device=device)

    warmup = 3
    n_iters = 10

    # --- 方法 1: 串行（先计算完所有，再通信） ---
    for _ in range(warmup):
        for g in gradients:
            torch.mm(compute_matrix, compute_matrix)
        for g in gradients:
            dist.all_reduce(g)
    torch.cuda.synchronize()

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(n_iters):
        # 所有计算
        results = []
        for i in range(num_layers):
            results.append(torch.mm(compute_matrix, compute_matrix))
        # 所有通信
        for g in gradients:
            dist.all_reduce(g)
    t1.record()
    torch.cuda.synchronize()
    serial_time = t0.elapsed_time(t1) / n_iters

    # --- 方法 2: Overlap（计算和通信交替） ---
    for _ in range(warmup):
        for g in gradients:
            handle = dist.all_reduce(g, async_op=True)
            torch.mm(compute_matrix, compute_matrix)
            handle.wait()
    torch.cuda.synchronize()

    t0.record()
    for _ in range(n_iters):
        handles = []
        for i in range(num_layers):
            # 异步通信当前层
            handle = dist.all_reduce(gradients[i], async_op=True)
            # 同时计算下一层
            torch.mm(compute_matrix, compute_matrix)
            handles.append(handle)
        # 等待所有通信完成
        for h in handles:
            h.wait()
    t1.record()
    torch.cuda.synchronize()
    overlap_time = t0.elapsed_time(t1) / n_iters

    if rank == 0:
        speedup = serial_time / overlap_time
        print(f"  串行: {serial_time:.2f} ms")
        print(f"  重叠: {overlap_time:.2f} ms")
        print(f"  加速: {speedup:.2f}x")
        print(f"  理想加速: 当通信完全被计算隐藏时 ≈ max(compute, comm) / (compute + comm)")


def demo_stream_overlap(rank, world_size, device):
    """
    使用 CUDA Stream 实现细粒度的 overlap。
    """
    if rank == 0:
        print("\n" + "=" * 60)
        print("Demo 2: CUDA Stream Overlap")
        print("=" * 60)

    # 创建独立的通信 stream
    compute_stream = torch.cuda.current_stream(device)
    comm_stream = torch.cuda.Stream(device)

    size = 8 * 1024 * 1024  # 8M elements
    compute_size = 2048
    n_iters = 10
    warmup = 3

    data = torch.randn(size, device=device)
    mat = torch.randn(compute_size, compute_size, device=device)

    # 预热
    for _ in range(warmup):
        dist.all_reduce(data)
        torch.mm(mat, mat)
    torch.cuda.synchronize()

    # --- 无 overlap ---
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(n_iters):
        dist.all_reduce(data)
        torch.mm(mat, mat)
    t1.record()
    torch.cuda.synchronize()
    no_overlap_time = t0.elapsed_time(t1) / n_iters

    # --- 有 overlap (不同 stream) ---
    t0.record()
    for _ in range(n_iters):
        # 在通信 stream 上启动 AllReduce
        with torch.cuda.stream(comm_stream):
            dist.all_reduce(data)

        # 在默认 stream 上做计算 (与通信并行)
        torch.mm(mat, mat)

        # 等待通信完成
        compute_stream.wait_stream(comm_stream)
    t1.record()
    torch.cuda.synchronize()
    with_overlap_time = t0.elapsed_time(t1) / n_iters

    if rank == 0:
        print(f"  无 Overlap: {no_overlap_time:.2f} ms")
        print(f"  有 Overlap: {with_overlap_time:.2f} ms")
        print(f"  加速: {no_overlap_time / with_overlap_time:.2f}x")
        print(f"\n  关键: 使用独立 CUDA stream 让通信和计算在硬件上真正并行")
        print(f"  注意: NVLink 通信不占用 SM，因此与计算可以完全 overlap")


def main():
    rank, world_size, device = setup()
    demo_async_allreduce(rank, world_size, device)
    dist.barrier()
    demo_stream_overlap(rank, world_size, device)
    dist.barrier()
    if rank == 0:
        print("\n通信优化实验完成！")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
