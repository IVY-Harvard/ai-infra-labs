"""
Lab 04 - 1F1B (One Forward One Backward) 调度
===============================================
核心思想: warmup 后交替执行 1 Forward + 1 Backward

三个阶段:
  Warmup:   前 p-1 个 micro-batch 做前向，填充流水线
  Steady:   交替 1F + 1B
  Cooldown: 最后 p-1 个 micro-batch 做反向，清空流水线

显存优势: 只需保存 p 个 micro-batch 的激活值 (vs GPipe 的 m 个)

运行:
    torchrun --nproc_per_node=4 1f1b_schedule.py --micro-batches 8
"""

import argparse
import os
import time
from collections import deque

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


def forward_step(stage, rank, world_size, device, inputs, mb_idx,
                  hidden_size, micro_batch_size, seq_len):
    """执行一次前向传播"""
    if rank == 0:
        x = inputs[mb_idx].clone().requires_grad_(True)
    else:
        x = torch.empty(micro_batch_size, seq_len, hidden_size, device=device)
        dist.recv(x, src=rank - 1)
        x = x.requires_grad_(True)

    output = stage(x)

    if rank < world_size - 1:
        dist.send(output.detach(), dst=rank + 1)

    return x, output


def backward_step(stage, rank, world_size, device, x, output):
    """执行一次反向传播"""
    if rank == world_size - 1:
        loss = output.mean()
        loss.backward()
    else:
        grad = torch.empty_like(output)
        dist.recv(grad, src=rank + 1)
        output.backward(grad)

    if rank > 0:
        dist.send(x.grad, dst=rank - 1)


def one_f_one_b_schedule(args):
    rank, world_size, device = setup()

    hidden_size = 512
    layers_per_stage = 4
    micro_batch_size = 8
    seq_len = 128
    m = args.micro_batches
    p = world_size

    if rank == 0:
        print(f"[1F1B] stages={p}, micro-batches={m}")
        print(f"  Bubble 率: {(p - 1) / (m + p - 1):.1%}")
        print(f"  最大同时保存激活数: {p} (vs GPipe 的 {m})")

    assert m >= p, f"micro-batches ({m}) 必须 >= stages ({p})"

    stage = StageModule(hidden_size, layers_per_stage).to(device)
    optimizer = torch.optim.Adam(stage.parameters(), lr=1e-4)

    if rank == 0:
        inputs = [
            torch.randn(micro_batch_size, seq_len, hidden_size, device=device)
            for _ in range(m)
        ]
    else:
        inputs = None

    # 激活值队列（FIFO: 先进先出，先前向的先反向）
    input_queue = deque()
    output_queue = deque()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # ========== Warmup Phase ==========
    # 前 (p - rank - 1) 个 micro-batch 做前向（每个 stage warmup 的数量不同）
    num_warmup = min(p - rank - 1, m)  # 简化: stage 0 warmup p-1 个
    num_warmup = p - 1  # 统一 warmup p-1 个（简化实现）

    fwd_idx = 0
    bwd_idx = 0

    for i in range(num_warmup):
        if fwd_idx < m:
            x, output = forward_step(
                stage, rank, world_size, device, inputs, fwd_idx,
                hidden_size, micro_batch_size, seq_len
            )
            input_queue.append(x)
            output_queue.append(output)
            fwd_idx += 1

    # ========== Steady Phase ==========
    # 交替 1F + 1B
    while fwd_idx < m:
        # 一次前向
        x, output = forward_step(
            stage, rank, world_size, device, inputs, fwd_idx,
            hidden_size, micro_batch_size, seq_len
        )
        input_queue.append(x)
        output_queue.append(output)
        fwd_idx += 1

        # 一次反向
        if output_queue:
            bwd_x = input_queue.popleft()
            bwd_output = output_queue.popleft()
            backward_step(stage, rank, world_size, device, bwd_x, bwd_output)
            bwd_idx += 1

    # ========== Cooldown Phase ==========
    # 做完剩余的反向
    while output_queue:
        bwd_x = input_queue.popleft()
        bwd_output = output_queue.popleft()
        backward_step(stage, rank, world_size, device, bwd_x, bwd_output)
        bwd_idx += 1

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9

    if rank == 0:
        print(f"\n=== 1F1B 结果 ===")
    for r in range(world_size):
        if rank == r:
            print(f"  Stage {rank}: 峰值显存 {peak_mem:.2f} GB | 总时间 {total_time:.3f}s")
        dist.barrier()

    if rank == 0:
        print(f"\n  1F1B 激活值峰值 ≈ {p} 个 micro-batch（远小于 GPipe 的 {m} 个）")
        print(f"  Bubble 率与 GPipe 相同，但显存大幅节省")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro-batches", type=int, default=8)
    one_f_one_b_schedule(parser.parse_args())
