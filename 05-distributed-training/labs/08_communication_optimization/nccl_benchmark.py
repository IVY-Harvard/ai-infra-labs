"""
Lab 08 - NCCL Benchmark
========================
全面测试 NCCL 各种集合通信操作的带宽和延迟。

运行:
    torchrun --nproc_per_node=8 nccl_benchmark.py
"""

import os
import torch
import torch.distributed as dist


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return rank, world_size, device


def benchmark_op(op_name, op_fn, sizes_bytes, rank, world_size, device, warmup=5, n_iters=20):
    """通用 benchmark 函数"""
    results = []

    for size_bytes in sizes_bytes:
        num_elements = size_bytes // 2  # BF16
        tensor = torch.randn(num_elements, dtype=torch.bfloat16, device=device)

        # 预热
        for _ in range(warmup):
            op_fn(tensor)
        torch.cuda.synchronize()

        # 计时
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(n_iters):
            op_fn(tensor)
        t1.record()
        torch.cuda.synchronize()

        time_ms = t0.elapsed_time(t1) / n_iters
        # 算法带宽
        algo_bw = size_bytes / (time_ms / 1000) / 1e9

        # Bus 带宽修正因子
        N = world_size
        if op_name == "AllReduce":
            bus_factor = 2 * (N - 1) / N
        elif op_name in ["AllGather", "ReduceScatter"]:
            bus_factor = (N - 1) / N
        elif op_name == "Broadcast":
            bus_factor = 1
        else:
            bus_factor = 1

        bus_bw = algo_bw * bus_factor
        results.append((size_bytes, time_ms, algo_bw, bus_bw))

    return results


def main():
    rank, world_size, device = setup()

    sizes_bytes = [
        64 * 1024,         # 64 KB
        256 * 1024,        # 256 KB
        1 * 1024 * 1024,   # 1 MB
        4 * 1024 * 1024,   # 4 MB
        16 * 1024 * 1024,  # 16 MB
        64 * 1024 * 1024,  # 64 MB
        256 * 1024 * 1024, # 256 MB
    ]

    operations = {
        "AllReduce": lambda t: dist.all_reduce(t),
        "Broadcast": lambda t: dist.broadcast(t, src=0),
        "ReduceScatter": lambda t: dist.reduce_scatter(
            torch.empty(t.shape[0] // world_size, dtype=t.dtype, device=t.device),
            list(t.chunk(world_size))
        ),
    }

    if rank == 0:
        print("=" * 80)
        print(f"NCCL Benchmark | {world_size} GPUs | {torch.cuda.get_device_name(0)}")
        print("=" * 80)

    for op_name, op_fn in operations.items():
        results = benchmark_op(op_name, op_fn, sizes_bytes, rank, world_size, device)

        if rank == 0:
            print(f"\n--- {op_name} ---")
            print(f"{'Size':<12} {'Time (ms)':<12} {'Algo BW (GB/s)':<16} {'Bus BW (GB/s)':<16}")
            print("-" * 56)
            for size_bytes, time_ms, algo_bw, bus_bw in results:
                size_label = f"{size_bytes/1024/1024:.1f} MB" if size_bytes >= 1024*1024 else f"{size_bytes/1024:.0f} KB"
                print(f"{size_label:<12} {time_ms:<12.3f} {algo_bw:<16.1f} {bus_bw:<16.1f}")

    if rank == 0:
        print(f"\n{'='*80}")
        print(f"分析:")
        print(f"  - 小消息 (<256KB): 受 latency 限制，带宽利用率低")
        print(f"  - 大消息 (>16MB): 接近硬件峰值带宽")
        print(f"  - H20 NVLink Bus BW 理论峰值: ~450 GB/s")
        print(f"  - 实际能达到理论峰值的 80-90%")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
