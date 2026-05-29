"""
Lab 04 - 朴素流水线并行
========================
最简单的流水线：每个 micro-batch 顺序通过所有 stage。
演示 bubble 问题。

运行:
    torchrun --nproc_per_node=4 naive_pipeline.py --micro-batches 4

核心知识:
  - 模型按层切分为 p 个 stage
  - 每个 stage 放在一张 GPU 上
  - micro-batch 通过 P2P Send/Recv 在 stage 间传递
  - 朴素做法: 每个 micro-batch 跑完所有 stage 再跑下一个 → bubble 巨大
"""

import argparse
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


class StageModule(nn.Module):
    """
    一个 pipeline stage 的模型：多个 Transformer 风格的 MLP 层。
    为了简化，不使用 Attention（聚焦 pipeline 调度逻辑）。
    """

    def __init__(self, hidden_size, num_layers_per_stage):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers_per_stage):
            self.layers.append(nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
            ))

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)  # 残差连接
        return x


def naive_pipeline(args):
    rank, world_size, device = setup()

    hidden_size = 512
    layers_per_stage = 4
    micro_batch_size = 8
    seq_len = 128
    num_micro_batches = args.micro_batches

    if rank == 0:
        print(f"[朴素流水线] stages={world_size}, micro-batches={num_micro_batches}")
        print(f"  理论 bubble 率: {(world_size - 1) / (num_micro_batches + world_size - 1):.1%}")

    # 每个 rank 创建自己的 stage
    stage = StageModule(hidden_size, layers_per_stage).to(device)
    optimizer = torch.optim.Adam(stage.parameters(), lr=1e-4)

    # 模拟输入数据（只有 stage 0 需要）
    if rank == 0:
        inputs = [
            torch.randn(micro_batch_size, seq_len, hidden_size, device=device)
            for _ in range(num_micro_batches)
        ]

    # 记录每个 stage 的活动时间
    active_time = 0.0
    total_start = time.perf_counter()

    # ========== 朴素调度: 逐个 micro-batch 处理 ==========
    for mb_idx in range(num_micro_batches):
        # --- Forward ---
        if rank == 0:
            x = inputs[mb_idx].requires_grad_(True)
        else:
            x = torch.empty(micro_batch_size, seq_len, hidden_size, device=device)
            dist.recv(x, src=rank - 1)
            x.requires_grad_(True)

        t_start = time.perf_counter()
        output = stage(x)
        torch.cuda.synchronize()
        active_time += time.perf_counter() - t_start

        if rank < world_size - 1:
            dist.send(output.detach(), dst=rank + 1)

        # --- Backward ---
        if rank == world_size - 1:
            loss = output.mean()
            t_start = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            active_time += time.perf_counter() - t_start
        else:
            grad = torch.empty_like(output)
            dist.recv(grad, src=rank + 1)
            t_start = time.perf_counter()
            output.backward(grad)
            torch.cuda.synchronize()
            active_time += time.perf_counter() - t_start

        if rank > 0:
            dist.send(x.grad, dst=rank - 1)

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    total_time = time.perf_counter() - total_start

    # 收集统计
    active_tensor = torch.tensor([active_time], device=device)
    total_tensor = torch.tensor([total_time], device=device)
    dist.barrier()

    if rank == 0:
        print(f"\n=== 朴素流水线结果 ===")
    for r in range(world_size):
        if rank == r:
            utilization = active_time / total_time * 100
            print(f"  Stage {rank}: 活跃 {active_time:.3f}s / 总 {total_time:.3f}s = {utilization:.1f}%")
        dist.barrier()

    if rank == 0:
        print(f"\n  注意: 各 stage 利用率远低于 100%，大量时间在等待")
        print(f"  GPipe 和 1F1B 通过 micro-batch 流水化解决此问题")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro-batches", type=int, default=4)
    naive_pipeline(parser.parse_args())
