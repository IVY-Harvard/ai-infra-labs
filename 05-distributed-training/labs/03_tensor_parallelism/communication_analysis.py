"""
Lab 03 - TP 通信量分析
=======================
测量 TP 中 AllReduce 的实际耗时，分析通信占比。

运行:
    torchrun --nproc_per_node=4 communication_analysis.py

输出:
    - 不同 tensor size 的 AllReduce 耗时
    - 有效带宽 vs 理论带宽
    - TP 通信占总训练时间的比例
"""

import os
import time
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


# ---------------------------------------------------------------------------
# 1. AllReduce 带宽测试
# ---------------------------------------------------------------------------

def benchmark_allreduce(rank, world_size, device):
    """测量不同大小 tensor 的 AllReduce 带宽"""
    if rank == 0:
        print("\n" + "=" * 70)
        print("AllReduce 带宽测试")
        print("=" * 70)
        print(f"{'Size':<15} {'Time (ms)':<12} {'Algo BW (GB/s)':<15} {'Bus BW (GB/s)':<15}")
        print("-" * 70)

    # 典型 TP 通信大小: B × S × H
    test_sizes = [
        ("1 MB",   1 * 1024 * 1024 // 2),   # 小消息
        ("4 MB",   4 * 1024 * 1024 // 2),
        ("16 MB",  16 * 1024 * 1024 // 2),   # B=2,S=1024,H=4096,BF16
        ("64 MB",  64 * 1024 * 1024 // 2),   # B=4,S=2048,H=4096,BF16
        ("256 MB", 256 * 1024 * 1024 // 2),  # 大消息
    ]

    tp_group = dist.new_group(list(range(world_size)))
    warmup = 5
    n_iters = 20

    for label, num_elements in test_sizes:
        tensor = torch.randn(num_elements, dtype=torch.bfloat16, device=device)
        size_bytes = tensor.numel() * tensor.element_size()

        # 预热
        for _ in range(warmup):
            dist.all_reduce(tensor, group=tp_group)
        torch.cuda.synchronize()

        # 计时
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(n_iters):
            dist.all_reduce(tensor, group=tp_group)
        t1.record()
        torch.cuda.synchronize()

        elapsed_ms = t0.elapsed_time(t1) / n_iters
        # 算法带宽 = 数据量 / 时间
        algo_bw = size_bytes / (elapsed_ms / 1000) / 1e9
        # Bus 带宽 = 算法带宽 × 修正因子 (AllReduce: 2(N-1)/N)
        bus_bw = algo_bw * 2 * (world_size - 1) / world_size

        if rank == 0:
            print(f"{label:<15} {elapsed_ms:<12.3f} {algo_bw:<15.1f} {bus_bw:<15.1f}")

    if rank == 0:
        print(f"\nBus BW 应接近 NVLink 理论带宽 (~400+ GB/s for H20)")
        print(f"小消息受 latency 影响，带宽利用率低")


# ---------------------------------------------------------------------------
# 2. TP 通信占比分析
# ---------------------------------------------------------------------------

def tp_overhead_analysis(rank, world_size, device):
    """模拟 TP 中的通信与计算，测量通信占比"""
    if rank == 0:
        print("\n" + "=" * 70)
        print("TP 通信开销分析 (模拟一个 Transformer 层)")
        print("=" * 70)

    tp_group = dist.new_group(list(range(world_size)))

    # 典型配置: B=4, S=2048, H=4096, FFN=16384
    B, S, H = 4, 2048, 4096
    FFN = H * 4
    num_layers = 32

    # AllReduce 的 tensor size (per layer, forward)
    comm_tensor = torch.randn(B * S, H, dtype=torch.bfloat16, device=device)
    comm_bytes = comm_tensor.numel() * comm_tensor.element_size()

    # 模拟计算: 矩阵乘法
    # MLP: X @ W1 + X @ W2 (近似)
    x = torch.randn(B * S, H // world_size, dtype=torch.bfloat16, device=device)
    w = torch.randn(H // world_size, FFN // world_size, dtype=torch.bfloat16, device=device)

    warmup = 5
    n_iters = 20

    # --- 通信时间 ---
    for _ in range(warmup):
        dist.all_reduce(comm_tensor, group=tp_group)
    torch.cuda.synchronize()

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(n_iters):
        dist.all_reduce(comm_tensor, group=tp_group)
    t1.record()
    torch.cuda.synchronize()
    comm_time_ms = t0.elapsed_time(t1) / n_iters

    # --- 计算时间 (模拟一层的两个矩阵乘) ---
    for _ in range(warmup):
        torch.mm(x, w)
    torch.cuda.synchronize()

    t0.record()
    for _ in range(n_iters):
        torch.mm(x, w)  # W1 forward
        torch.mm(x, w)  # W2 forward
    t1.record()
    torch.cuda.synchronize()
    compute_time_ms = t0.elapsed_time(t1) / n_iters

    if rank == 0:
        # 每层: 2 次 AllReduce (forward), 2 次 AllReduce (backward)
        total_comm_per_step = num_layers * 4 * comm_time_ms
        # 简化估算 (只算 MLP forward, 实际 backward ≈ 2x forward)
        total_compute_per_step = num_layers * compute_time_ms * 3  # fwd + bwd ≈ 3x fwd

        print(f"配置: B={B}, S={S}, H={H}, TP={world_size}, Layers={num_layers}")
        print(f"  单次 AllReduce ({comm_bytes / 1e6:.0f} MB): {comm_time_ms:.2f} ms")
        print(f"  单层计算 (MLP fwd): {compute_time_ms:.2f} ms")
        print(f"  每步总通信 (4×{num_layers}层): {total_comm_per_step:.0f} ms")
        print(f"  每步总计算 (估算): {total_compute_per_step:.0f} ms")
        print(f"  通信/计算比: {total_comm_per_step / total_compute_per_step:.2f}")
        print(f"\n  > 通信/计算比 < 1 → 通信可以被 overlap 隐藏")
        print(f"  > 通信/计算比 > 1 → 通信成为瓶颈，考虑减小 TP size")


# ---------------------------------------------------------------------------
# 3. 不同 TP size 的扩展性分析
# ---------------------------------------------------------------------------

def scaling_analysis(rank, world_size, device):
    """分析不同 TP size 对吞吐量的影响"""
    if rank == 0:
        print("\n" + "=" * 70)
        print("TP 扩展性分析")
        print("=" * 70)

    tp_group = dist.new_group(list(range(world_size)))

    H = 4096
    B_S = 4 * 2048  # batch * seq_len

    # 模拟不同 TP size 下的矩阵乘法效率
    # TP 越大，每卡的矩阵越小 → GEMM 效率可能下降
    for simulated_tp in [1, 2, 4, 8]:
        local_h = H // simulated_tp
        x = torch.randn(B_S, local_h, dtype=torch.bfloat16, device=device)
        w = torch.randn(local_h, H, dtype=torch.bfloat16, device=device)

        # 预热
        for _ in range(5):
            torch.mm(x, w)
        torch.cuda.synchronize()

        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(20):
            torch.mm(x, w)
        t1.record()
        torch.cuda.synchronize()

        time_ms = t0.elapsed_time(t1) / 20
        flops = 2 * B_S * local_h * H  # GEMM FLOPs
        tflops = flops / (time_ms / 1000) / 1e12

        if rank == 0:
            print(f"  TP={simulated_tp:2d} | 矩阵: [{B_S}×{local_h}] @ [{local_h}×{H}] | "
                  f"Time: {time_ms:.2f}ms | Efficiency: {tflops:.1f} TFLOPS")

    if rank == 0:
        print(f"\n  注意: TP 越大，每卡矩阵越小，TFLOPS 可能下降")
        print(f"  这是因为小矩阵无法充分利用 GPU 的并行单元")


def main():
    rank, world_size, device = setup()
    benchmark_allreduce(rank, world_size, device)
    dist.barrier()
    tp_overhead_analysis(rank, world_size, device)
    dist.barrier()
    scaling_analysis(rank, world_size, device)
    dist.barrier()
    if rank == 0:
        print("\n通信分析完成！")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
