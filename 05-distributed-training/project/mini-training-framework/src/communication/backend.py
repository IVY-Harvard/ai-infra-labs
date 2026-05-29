"""
通信后端抽象
=============
统一的通信接口，封装 NCCL 后端。
"""

from typing import Optional, List
from enum import Enum

import torch
import torch.distributed as dist


class CommBackend:
    """
    通信后端抽象层。
    封装 PyTorch distributed 的底层通信操作。
    """

    def __init__(self, backend: str = "nccl"):
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        self.backend = backend
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    @property
    def is_initialized(self) -> bool:
        return dist.is_initialized()

    def barrier(self, group: Optional[dist.ProcessGroup] = None):
        dist.barrier(group=group)

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
        group: Optional[dist.ProcessGroup] = None,
        async_op: bool = False,
    ):
        return dist.all_reduce(tensor, op=op, group=group, async_op=async_op)

    def all_gather(
        self,
        output_tensors: List[torch.Tensor],
        input_tensor: torch.Tensor,
        group: Optional[dist.ProcessGroup] = None,
        async_op: bool = False,
    ):
        return dist.all_gather(output_tensors, input_tensor, group=group, async_op=async_op)

    def reduce_scatter(
        self,
        output: torch.Tensor,
        input_list: List[torch.Tensor],
        op: dist.ReduceOp = dist.ReduceOp.SUM,
        group: Optional[dist.ProcessGroup] = None,
        async_op: bool = False,
    ):
        return dist.reduce_scatter(output, input_list, op=op, group=group, async_op=async_op)

    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int,
        group: Optional[dist.ProcessGroup] = None,
        async_op: bool = False,
    ):
        return dist.broadcast(tensor, src=src, group=group, async_op=async_op)

    def send(self, tensor: torch.Tensor, dst: int, group: Optional[dist.ProcessGroup] = None):
        return dist.send(tensor, dst=dst, group=group)

    def recv(self, tensor: torch.Tensor, src: int, group: Optional[dist.ProcessGroup] = None):
        return dist.recv(tensor, src=src, group=group)

    def isend(self, tensor: torch.Tensor, dst: int, group: Optional[dist.ProcessGroup] = None):
        return dist.isend(tensor, dst=dst, group=group)

    def irecv(self, tensor: torch.Tensor, src: int, group: Optional[dist.ProcessGroup] = None):
        return dist.irecv(tensor, src=src, group=group)

    def new_group(self, ranks: List[int]) -> dist.ProcessGroup:
        return dist.new_group(ranks)

    def destroy(self):
        if dist.is_initialized():
            dist.destroy_process_group()
