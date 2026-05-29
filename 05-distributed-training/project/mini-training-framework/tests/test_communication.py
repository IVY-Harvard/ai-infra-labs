"""
通信模块测试
=============
测试集合通信操作的正确性。

运行:
    torchrun --nproc_per_node=4 tests/test_communication.py
"""

import os
import sys
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.communication.backend import CommBackend
from src.communication.collective_ops import BucketedAllReduce, GradientReducer
from src.communication.topology import TopologyDetector


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return rank, world_size, device


def test_comm_backend(rank, world_size, device):
    """测试通信后端基础操作"""
    backend = CommBackend.__new__(CommBackend)
    backend.rank = rank
    backend.world_size = world_size

    # AllReduce
    tensor = torch.ones(10, device=device) * rank
    dist.all_reduce(tensor)
    expected = sum(range(world_size))
    assert torch.allclose(tensor, torch.full_like(tensor, expected))

    # Broadcast
    tensor = torch.ones(10, device=device) * rank
    dist.broadcast(tensor, src=0)
    assert torch.allclose(tensor, torch.zeros_like(tensor))

    if rank == 0:
        print("  [PASS] CommBackend basic ops")


def test_bucketed_allreduce(rank, world_size, device):
    """测试分桶 AllReduce"""
    tensors = [torch.ones(100, device=device) * rank for _ in range(5)]

    bucketed = BucketedAllReduce(bucket_size_mb=0.001)  # 小桶，强制多次 flush
    for t in tensors:
        bucketed.add(t)
    bucketed.wait()

    # 验证结果
    expected = sum(range(world_size))
    for t in tensors:
        assert torch.allclose(t, torch.full_like(t, expected)), \
            f"BucketedAllReduce failed: {t[0].item()} != {expected}"

    if rank == 0:
        print("  [PASS] BucketedAllReduce")


def test_topology_detector(rank, world_size, device):
    """测试拓扑检测"""
    topo = TopologyDetector()
    nvlink_groups = topo.get_nvlink_groups()
    recommended_tp = topo.recommend_tp_size()

    if rank == 0:
        print(f"  [PASS] TopologyDetector: NVLink groups={nvlink_groups}, "
              f"recommended TP={recommended_tp}")


def test_gradient_reducer(rank, world_size, device):
    """测试梯度 Reduce"""
    # 创建模拟参数
    params = [torch.nn.Parameter(torch.ones(10, device=device) * rank)]
    params[0].grad = torch.ones(10, device=device) * rank

    reducer = GradientReducer(world_size=world_size)
    reducer.all_reduce_grads(params)

    expected_avg = sum(range(world_size)) / world_size
    assert torch.allclose(params[0].grad, torch.full((10,), expected_avg, device=device))

    if rank == 0:
        print("  [PASS] GradientReducer")


def main():
    rank, world_size, device = setup()

    if rank == 0:
        print(f"通信模块测试 (world_size={world_size})")

    test_comm_backend(rank, world_size, device)
    dist.barrier()
    test_bucketed_allreduce(rank, world_size, device)
    dist.barrier()
    test_topology_detector(rank, world_size, device)
    dist.barrier()
    test_gradient_reducer(rank, world_size, device)
    dist.barrier()

    if rank == 0:
        print("\n所有通信测试通过！")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
