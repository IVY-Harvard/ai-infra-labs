"""
Lab 08 - 手写 Ring AllReduce
==============================
从零实现 Ring AllReduce = ReduceScatter + AllGather。

算法:
  N 个 GPU 组成环，数据分为 N 个 chunk。
  Phase 1 (ReduceScatter): N-1 步，每步发送一个 chunk 给下一个 rank，累加
  Phase 2 (AllGather): N-1 步，每步发送已完成的 chunk 给下一个 rank

通信量: 每 GPU 发送 2(N-1)/N × M ≈ 2M （与 N 无关，带宽最优）

运行:
    torchrun --nproc_per_node=4 ring_allreduce.py
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


def ring_allreduce_manual(tensor: torch.Tensor, rank: int, world_size: int):
    """
    手写 Ring AllReduce。

    输入: 每个 rank 有一份数据 tensor
    输出: 所有 rank 得到所有数据的 sum

    实现步骤:
    1. 将 tensor 分为 world_size 个 chunk
    2. ReduceScatter: N-1 步环形传递+累加
    3. AllGather: N-1 步环形传递
    """
    N = world_size
    assert tensor.numel() % N == 0, "Tensor 大小必须能被 world_size 整除"

    # 分成 N 个 chunk
    chunks = list(tensor.chunk(N))

    # 环的邻居
    left = (rank - 1) % N
    right = (rank + 1) % N

    # ========== Phase 1: ReduceScatter ==========
    # N-1 步，每步:
    #   发送 chunk[(rank - step) % N] 给 right
    #   从 left 接收，累加到 chunk[(rank - step - 1) % N]
    for step in range(N - 1):
        send_idx = (rank - step) % N
        recv_idx = (rank - step - 1) % N

        send_buf = chunks[send_idx].clone()
        recv_buf = torch.empty_like(chunks[recv_idx])

        # 非阻塞 send/recv 避免死锁
        send_req = dist.isend(send_buf, dst=right)
        recv_req = dist.irecv(recv_buf, src=left)

        send_req.wait()
        recv_req.wait()

        # 累加接收到的数据
        chunks[recv_idx] += recv_buf

    # 此时: rank i 持有 chunk[(rank+1)%N] 的完整 reduce 结果
    # (注意索引: 经过 N-1 步后，rank i 的 chunk[(i+1)%N] 包含了所有 rank 的 sum)

    # ========== Phase 2: AllGather ==========
    # N-1 步，每步:
    #   发送已完成的 chunk 给 right
    #   从 left 接收完成的 chunk
    for step in range(N - 1):
        send_idx = (rank - step + 1) % N
        recv_idx = (rank - step) % N

        send_buf = chunks[send_idx].clone()
        recv_buf = torch.empty_like(chunks[recv_idx])

        send_req = dist.isend(send_buf, dst=right)
        recv_req = dist.irecv(recv_buf, src=left)

        send_req.wait()
        recv_req.wait()

        chunks[recv_idx] = recv_buf

    # 组装结果
    result = torch.cat(chunks)
    tensor.copy_(result)
    return tensor


def verify_correctness(rank, world_size, device):
    """验证手写 Ring AllReduce 的正确性"""
    if rank == 0:
        print("=" * 60)
        print("Ring AllReduce 正确性验证")
        print("=" * 60)

    # 每个 rank 创建不同的数据
    torch.manual_seed(rank)
    size = 1024  # 每个 rank 的数据大小
    data = torch.randn(size, device=device)

    # 保存原始数据
    data_copy = data.clone()

    # 手写 Ring AllReduce
    ring_allreduce_manual(data, rank, world_size)

    # NCCL AllReduce 作为参考
    ref_data = data_copy.clone()
    dist.all_reduce(ref_data, op=dist.ReduceOp.SUM)

    # 对比
    diff = (data - ref_data).abs().max().item()
    if rank == 0:
        print(f"  数据大小: {size} elements")
        print(f"  手写 vs NCCL 最大差异: {diff:.2e}")
        assert diff < 1e-4, f"差异过大: {diff}"
        print(f"  PASS: Ring AllReduce 实现正确！")


def benchmark_comparison(rank, world_size, device):
    """对比手写 Ring AllReduce 和 NCCL 的性能"""
    if rank == 0:
        print("\n" + "=" * 60)
        print("性能对比: 手写 Ring vs NCCL")
        print("=" * 60)

    sizes = [1024, 1024*1024, 4*1024*1024]  # 4KB, 4MB, 16MB
    warmup = 3
    n_iters = 10

    for num_elements in sizes:
        size_mb = num_elements * 4 / 1e6  # float32 = 4 bytes

        # 手写 Ring AllReduce
        data = torch.randn(num_elements, device=device)
        for _ in range(warmup):
            ring_allreduce_manual(data.clone(), rank, world_size)
        torch.cuda.synchronize()

        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(n_iters):
            d = data.clone()
            ring_allreduce_manual(d, rank, world_size)
        t1.record()
        torch.cuda.synchronize()
        manual_time = t0.elapsed_time(t1) / n_iters

        # NCCL AllReduce
        data = torch.randn(num_elements, device=device)
        for _ in range(warmup):
            dist.all_reduce(data)
        torch.cuda.synchronize()

        t0.record()
        for _ in range(n_iters):
            dist.all_reduce(data)
        t1.record()
        torch.cuda.synchronize()
        nccl_time = t0.elapsed_time(t1) / n_iters

        if rank == 0:
            print(f"  Size: {size_mb:.1f} MB | Manual: {manual_time:.2f} ms | "
                  f"NCCL: {nccl_time:.2f} ms | Ratio: {manual_time/nccl_time:.1f}x")

    if rank == 0:
        print(f"\n  手写版本慢于 NCCL 因为:")
        print(f"    1. NCCL 使用 kernel fusion 和 CUDA graph")
        print(f"    2. NCCL 利用多个 channel (ring) 并行传输")
        print(f"    3. NCCL 针对 NVLink 拓扑做了优化")
        print(f"  但手写帮助理解算法原理！")


def main():
    rank, world_size, device = setup()
    verify_correctness(rank, world_size, device)
    dist.barrier()
    benchmark_comparison(rank, world_size, device)
    dist.barrier()
    if rank == 0:
        print("\n完成！")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
