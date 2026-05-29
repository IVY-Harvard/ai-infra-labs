"""
Lab 04 - GPipe 调度
====================
GPipe: 所有 micro-batch 先做前向，再做反向。
  Phase 1: F0, F1, F2, ..., Fm-1 (all forward)
  Phase 2: Bm-1, Bm-2, ..., B0   (all backward)

优点: 实现简单
缺点: 需要保存 m 个 micro-batch 的激活值 → 显存压力大

运行:
    torchrun --nproc_per_node=4 gpipe_schedule.py --micro-batches 8
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
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


def gpipe_schedule(args):
    rank, world_size, device = setup()

    hidden_size = 512
    layers_per_stage = 4
    micro_batch_size = 8
    seq_len = 128
    m = args.micro_batches  # num micro-batches
    p = world_size           # num stages

    if rank == 0:
        print(f"[GPipe] stages={p}, micro-batches={m}")
        print(f"  Bubble 率: {(p - 1) / (m + p - 1):.1%}")

    stage = StageModule(hidden_size, layers_per_stage).to(device)
    optimizer = torch.optim.Adam(stage.parameters(), lr=1e-4)

    # 输入数据
    if rank == 0:
        inputs = [
            torch.randn(micro_batch_size, seq_len, hidden_size, device=device)
            for _ in range(m)
        ]

    # 保存中间激活值（GPipe 需要保存所有 micro-batch 的激活）
    saved_inputs = []    # 每个 micro-batch 的输入（用于反向）
    saved_outputs = []   # 每个 micro-batch 的输出

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # ========== Phase 1: All Forwards ==========
    for mb_idx in range(m):
        if rank == 0:
            x = inputs[mb_idx].clone().requires_grad_(True)
        else:
            x = torch.empty(micro_batch_size, seq_len, hidden_size, device=device)
            dist.recv(x, src=rank - 1)
            x = x.requires_grad_(True)

        saved_inputs.append(x)
        output = stage(x)
        saved_outputs.append(output)

        if rank < world_size - 1:
            dist.send(output.detach(), dst=rank + 1)

    # ========== Phase 2: All Backwards (reverse order) ==========
    for mb_idx in reversed(range(m)):
        output = saved_outputs[mb_idx]
        x = saved_inputs[mb_idx]

        if rank == world_size - 1:
            loss = output.mean()
            loss.backward()
        else:
            grad = torch.empty_like(output)
            dist.recv(grad, src=rank + 1)
            output.backward(grad)

        if rank > 0:
            dist.send(x.grad, dst=rank - 1)

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    # 显存统计
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9

    if rank == 0:
        print(f"\n=== GPipe 结果 ===")
    for r in range(world_size):
        if rank == r:
            print(f"  Stage {rank}: 峰值显存 {peak_mem:.2f} GB | 总时间 {total_time:.3f}s")
        dist.barrier()

    if rank == 0:
        print(f"\n  GPipe 需要保存 {m} 个 micro-batch 的激活值")
        print(f"  当 m 很大时显存压力大 → 需要 activation recomputation")
        print(f"  1F1B 改进: 只需保存 {p} 个激活值")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro-batches", type=int, default=8)
    gpipe_schedule(parser.parse_args())
