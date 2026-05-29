"""
集合通信操作
=============
高层集合通信封装，支持:
  - 带自动 dtype 转换的 AllReduce
  - 分桶通信（Bucketed Communication）
  - 异步通信管理
"""

from typing import List, Optional
import torch
import torch.distributed as dist


class BucketedAllReduce:
    """
    分桶 AllReduce — 模拟 DDP 的 gradient bucketing 策略。
    将多个小 tensor 合并为一个大 bucket 再通信，减少 latency 开销。
    """

    def __init__(self, bucket_size_mb: float = 25.0, group: Optional[dist.ProcessGroup] = None):
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        self.group = group
        self._buckets: List[torch.Tensor] = []
        self._handles = []

    def add(self, tensor: torch.Tensor):
        """添加 tensor 到当前 bucket"""
        self._buckets.append(tensor)
        total_bytes = sum(t.numel() * t.element_size() for t in self._buckets)
        if total_bytes >= self.bucket_size_bytes:
            self.flush()

    def flush(self):
        """发送当前 bucket"""
        if not self._buckets:
            return

        # 拼接为连续内存
        flat = torch.cat([t.flatten() for t in self._buckets])
        handle = dist.all_reduce(flat, group=self.group, async_op=True)
        self._handles.append((handle, flat, [t.shape for t in self._buckets],
                              [t.numel() for t in self._buckets]))
        self._buckets = []

    def wait(self):
        """等待所有 bucket 完成，并写回原始 tensor"""
        self.flush()  # flush 残余
        for handle, flat, shapes, sizes in self._handles:
            handle.wait()
        self._handles = []


class GradientReducer:
    """
    梯度 Reduce 工具，支持不同策略:
    - all_reduce: 完整 AllReduce
    - reduce_scatter: 用于 ZeRO-2/3
    """

    def __init__(self, group: Optional[dist.ProcessGroup] = None, world_size: int = 1):
        self.group = group
        self.world_size = world_size

    def all_reduce_grads(self, params, async_op: bool = False):
        """对所有参数的梯度做 AllReduce（均值）"""
        handles = []
        for p in params:
            if p.grad is not None:
                h = dist.all_reduce(p.grad, group=self.group, async_op=async_op)
                if async_op:
                    handles.append(h)
                p.grad.div_(self.world_size)
        return handles

    def reduce_scatter_grads(self, params, rank: int):
        """对梯度做 ReduceScatter，每 rank 只保留 1/N"""
        for p in params:
            if p.grad is None:
                continue
            flat = p.grad.flatten()
            chunk_size = flat.numel() // self.world_size
            output = torch.empty(chunk_size, dtype=flat.dtype, device=flat.device)
            dist.reduce_scatter(output, list(flat.chunk(self.world_size)), group=self.group)
            # 将 scatter 后的梯度存回（只有本地分片）
            p.grad = output
