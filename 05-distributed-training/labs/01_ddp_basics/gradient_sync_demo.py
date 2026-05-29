"""
Lab 01 - 梯度同步观察实验
===========================
深入观察 DDP 中的梯度同步机制：
  1. AllReduce 前后梯度的变化
  2. Gradient Bucketing 的行为
  3. no_sync 上下文管理器（梯度累积）

运行方式:
    torchrun --nproc_per_node=4 gradient_sync_demo.py

核心知识点:
  - DDP 在 backward() 过程中通过 autograd hook 触发 AllReduce
  - 梯度按 bucket（默认 25MB）分批同步，与反向传播重叠
  - no_sync() 可以跳过同步，实现梯度累积
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return rank, world_size, device


def cleanup():
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# 实验 1: 观察 AllReduce 前后的梯度
# ---------------------------------------------------------------------------

def experiment_1_gradient_before_after(rank, world_size, device):
    """
    用一个简单的线性模型，观察 AllReduce 同步前后梯度的变化。

    关键点:
    - 每个 rank 用不同的输入 → 计算出不同的梯度
    - DDP 在 backward() 中自动 AllReduce → 所有 rank 得到相同的平均梯度
    """
    if rank == 0:
        print("\n" + "=" * 60)
        print("实验 1: AllReduce 前后的梯度对比")
        print("=" * 60)

    # 简单模型
    model = nn.Linear(64, 32, bias=False).to(device)
    ddp_model = DDP(model, device_ids=[device])

    # 每个 rank 用不同的输入数据（乘以 rank+1 制造差异）
    torch.manual_seed(42)
    x = torch.randn(8, 64, device=device) * (rank + 1)
    target = torch.randn(8, 32, device=device)

    # --- 方法: 对比 DDP 和非 DDP 的梯度 ---
    # 非 DDP 的本地梯度
    model_local = nn.Linear(64, 32, bias=False).to(device)
    model_local.weight.data.copy_(ddp_model.module.weight.data)

    loss_local = ((model_local(x) - target) ** 2).mean()
    loss_local.backward()
    local_grad = model_local.weight.grad.clone()

    # DDP 的同步梯度
    loss_ddp = ((ddp_model(x) - target) ** 2).mean()
    loss_ddp.backward()
    synced_grad = ddp_model.module.weight.grad.clone()

    # 收集所有 rank 的本地梯度
    local_grad_list = [torch.zeros_like(local_grad) for _ in range(world_size)]
    dist.all_gather(local_grad_list, local_grad)

    if rank == 0:
        # 手动计算平均梯度
        manual_avg = sum(local_grad_list) / world_size

        print(f"  各 rank 的本地梯度 L2 范数:")
        for r, g in enumerate(local_grad_list):
            print(f"    Rank {r}: {g.norm().item():.6f}")
        print(f"  手动计算平均梯度范数: {manual_avg.norm().item():.6f}")
        print(f"  DDP 同步后梯度范数:   {synced_grad.norm().item():.6f}")
        print(f"  差异 (应为 ~0): {(manual_avg - synced_grad).abs().max().item():.2e}")


# ---------------------------------------------------------------------------
# 实验 2: Gradient Bucketing 行为
# ---------------------------------------------------------------------------

def experiment_2_bucketing(rank, world_size, device):
    """
    观察 DDP 的 bucket 分配和通信时间。

    关键点:
    - DDP 将参数按逆序（从最后一层到第一层）分入 bucket
    - bucket 满后立即触发 AllReduce，与反向传播重叠
    - bucket_cap_mb 控制桶大小
    """
    if rank == 0:
        print("\n" + "=" * 60)
        print("实验 2: Gradient Bucketing 行为")
        print("=" * 60)

    # 多层模型
    model = nn.Sequential(
        nn.Linear(256, 512),   # ~128K params
        nn.ReLU(),
        nn.Linear(512, 512),   # ~256K params
        nn.ReLU(),
        nn.Linear(512, 256),   # ~128K params
        nn.ReLU(),
        nn.Linear(256, 10),    # ~2.5K params
    ).to(device)

    # 对比不同 bucket 大小
    for bucket_mb in [1, 5, 25]:
        ddp_model = DDP(model, device_ids=[device], bucket_cap_mb=bucket_mb)

        x = torch.randn(64, 256, device=device)
        target = torch.randint(0, 10, (64,), device=device)

        # 预热
        for _ in range(3):
            loss = nn.functional.cross_entropy(ddp_model(x), target)
            ddp_model.zero_grad()
            loss.backward()

        # 计时
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)

        t0.record()
        for _ in range(20):
            loss = nn.functional.cross_entropy(ddp_model(x), target)
            ddp_model.zero_grad()
            loss.backward()
        t1.record()
        torch.cuda.synchronize()

        elapsed = t0.elapsed_time(t1) / 20  # 毫秒

        if rank == 0:
            # 打印 bucket 信息
            num_buckets = len(list(ddp_model._module_to_parameters_to_clean_up.keys())) if hasattr(ddp_model, '_module_to_parameters_to_clean_up') else "N/A"
            print(f"  bucket_cap_mb={bucket_mb:2d} | 反向传播平均耗时: {elapsed:.2f} ms")


# ---------------------------------------------------------------------------
# 实验 3: no_sync 与梯度累积
# ---------------------------------------------------------------------------

def experiment_3_no_sync(rank, world_size, device):
    """
    演示 DDP 的梯度累积模式。

    关键点:
    - model.no_sync() 跳过 AllReduce（累积本地梯度）
    - 最后一个 micro-batch 才做同步
    - 等效于更大的 batch size
    """
    if rank == 0:
        print("\n" + "=" * 60)
        print("实验 3: no_sync() 梯度累积")
        print("=" * 60)

    model = nn.Linear(64, 32, bias=False).to(device)
    ddp_model = DDP(model, device_ids=[device])

    accumulation_steps = 4
    torch.manual_seed(rank * 100 + 42)

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    optimizer.zero_grad()

    for micro_step in range(accumulation_steps):
        x = torch.randn(8, 64, device=device)
        target = torch.randn(8, 32, device=device)

        # 前 K-1 步不同步
        if micro_step < accumulation_steps - 1:
            with ddp_model.no_sync():
                loss = ((ddp_model(x) - target) ** 2).mean()
                loss.backward()
                if rank == 0:
                    grad_norm = ddp_model.module.weight.grad.norm().item()
                    print(f"  micro_step {micro_step}: no_sync, local grad norm = {grad_norm:.4f}")
        else:
            # 最后一步做同步
            loss = ((ddp_model(x) - target) ** 2).mean()
            loss.backward()
            if rank == 0:
                grad_norm = ddp_model.module.weight.grad.norm().item()
                print(f"  micro_step {micro_step}: synced,  avg grad norm = {grad_norm:.4f}")

    # 检查所有 rank 的梯度是否一致
    grad = ddp_model.module.weight.grad.clone()
    grad_list = [torch.zeros_like(grad) for _ in range(world_size)]
    dist.all_gather(grad_list, grad)

    if rank == 0:
        all_same = all(torch.allclose(grad_list[0], g) for g in grad_list)
        print(f"  所有 rank 梯度一致: {all_same}")


# ---------------------------------------------------------------------------
# 实验 4: 同步 vs 异步的性能差异
# ---------------------------------------------------------------------------

def experiment_4_sync_overhead(rank, world_size, device):
    """
    测量 DDP 的通信开销占总训练时间的比例。
    对比：DDP backward vs 纯本地 backward（no_sync）
    """
    if rank == 0:
        print("\n" + "=" * 60)
        print("实验 4: DDP 通信开销测量")
        print("=" * 60)

    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
    ).to(device)
    ddp_model = DDP(model, device_ids=[device])

    x = torch.randn(64, 1024, device=device)
    target = torch.randn(64, 1024, device=device)

    n_iters = 50

    # 预热
    for _ in range(5):
        loss = ((ddp_model(x) - target) ** 2).mean()
        ddp_model.zero_grad()
        loss.backward()

    # 有同步的 backward
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(n_iters):
        loss = ((ddp_model(x) - target) ** 2).mean()
        ddp_model.zero_grad()
        loss.backward()
    t1.record()
    torch.cuda.synchronize()
    time_with_sync = t0.elapsed_time(t1) / n_iters

    # 无同步的 backward
    t0.record()
    for _ in range(n_iters):
        with ddp_model.no_sync():
            loss = ((ddp_model(x) - target) ** 2).mean()
            ddp_model.zero_grad()
            loss.backward()
    t1.record()
    torch.cuda.synchronize()
    time_no_sync = t0.elapsed_time(t1) / n_iters

    if rank == 0:
        overhead = time_with_sync - time_no_sync
        overhead_pct = overhead / time_with_sync * 100
        print(f"  有同步 backward: {time_with_sync:.2f} ms")
        print(f"  无同步 backward: {time_no_sync:.2f} ms")
        print(f"  通信开销: {overhead:.2f} ms ({overhead_pct:.1f}%)")
        print(f"  注意: NVLink 互联下开销应较小，跨机会显著增大")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    rank, world_size, device = setup()

    experiment_1_gradient_before_after(rank, world_size, device)
    dist.barrier()

    experiment_2_bucketing(rank, world_size, device)
    dist.barrier()

    experiment_3_no_sync(rank, world_size, device)
    dist.barrier()

    experiment_4_sync_overhead(rank, world_size, device)
    dist.barrier()

    if rank == 0:
        print("\n所有实验完成！")

    cleanup()


if __name__ == "__main__":
    main()
