"""
并行模块测试
=============
测试 TP / PP / DP 各模块的正确性。

运行:
    torchrun --nproc_per_node=4 tests/test_parallel.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parallel.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    TPParallelMLP,
    TPTransformerLayer,
)


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return rank, world_size, device


def test_column_parallel_linear(rank, world_size, device):
    """测试列并行 Linear 的前向和反向"""
    tp_group = dist.new_group(list(range(world_size)))
    in_features, out_features = 256, 512

    col = ColumnParallelLinear(
        in_features, out_features, world_size, rank, tp_group,
        bias=True, gather_output=True,
    ).to(device)

    x = torch.randn(4, 32, in_features, device=device, requires_grad=True)
    y = col(x)

    assert y.shape == (4, 32, out_features), f"Shape mismatch: {y.shape}"
    loss = y.sum()
    loss.backward()
    assert x.grad is not None, "No gradient on input"

    if rank == 0:
        print("  [PASS] ColumnParallelLinear")


def test_row_parallel_linear(rank, world_size, device):
    """测试行并行 Linear"""
    tp_group = dist.new_group(list(range(world_size)))
    in_features, out_features = 512, 256

    row = RowParallelLinear(
        in_features, out_features, world_size, rank, tp_group,
        bias=True, input_is_parallel=True,
    ).to(device)

    # 输入是分片的
    x = torch.randn(4, 32, in_features // world_size, device=device, requires_grad=True)
    y = row(x)

    assert y.shape == (4, 32, out_features), f"Shape mismatch: {y.shape}"
    loss = y.sum()
    loss.backward()

    if rank == 0:
        print("  [PASS] RowParallelLinear")


def test_tp_mlp(rank, world_size, device):
    """测试 TP MLP"""
    tp_group = dist.new_group(list(range(world_size)))
    hidden_size = 256
    ffn_size = 1024

    mlp = TPParallelMLP(hidden_size, ffn_size, world_size, rank, tp_group).to(device)
    x = torch.randn(4, 32, hidden_size, device=device, requires_grad=True)
    y = mlp(x)

    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    y.sum().backward()

    if rank == 0:
        print("  [PASS] TPParallelMLP")


def test_tp_transformer_layer(rank, world_size, device):
    """测试 TP Transformer 层"""
    tp_group = dist.new_group(list(range(world_size)))
    hidden_size = 256
    num_heads = 8
    ffn_size = 1024

    layer = TPTransformerLayer(
        hidden_size, num_heads, ffn_size, world_size, rank, tp_group
    ).to(device)

    x = torch.randn(4, 32, hidden_size, device=device, requires_grad=True)
    y = layer(x)

    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None

    if rank == 0:
        print("  [PASS] TPTransformerLayer")


def main():
    rank, world_size, device = setup()

    if rank == 0:
        print(f"并行模块测试 (TP size = {world_size})")

    test_column_parallel_linear(rank, world_size, device)
    dist.barrier()
    test_row_parallel_linear(rank, world_size, device)
    dist.barrier()
    test_tp_mlp(rank, world_size, device)
    dist.barrier()
    test_tp_transformer_layer(rank, world_size, device)
    dist.barrier()

    if rank == 0:
        print("\n所有并行测试通过！")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
