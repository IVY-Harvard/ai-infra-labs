"""
流水线并行 — 1F1B 调度实现
==============================
实现 1F1B (One Forward One Backward) 调度。

核心流程:
  1. Warmup:   前 p-1 个 micro-batch 做前向
  2. Steady:   交替 1F + 1B
  3. Cooldown: 做完剩余反向
"""

from typing import List, Optional, Callable
from collections import deque

import torch
import torch.nn as nn
import torch.distributed as dist


class PipelineStage(nn.Module):
    """
    Pipeline Stage: 模型的一个子集（若干层）。
    """

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class PipelineSchedule1F1B:
    """
    1F1B 流水线调度器。

    参数:
        stage: 当前 stage 的模型
        pp_rank: 当前在 pipeline 中的位置
        pp_size: pipeline stage 总数
        pp_group: PP 通信组
        num_micro_batches: micro-batch 数量
        micro_batch_size: 每个 micro-batch 的 batch size
    """

    def __init__(
        self,
        stage: PipelineStage,
        pp_rank: int,
        pp_size: int,
        pp_group: dist.ProcessGroup,
        pp_ranks: List[int],
        num_micro_batches: int = 8,
    ):
        self.stage = stage
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.pp_group = pp_group
        self.pp_ranks = pp_ranks
        self.num_micro_batches = num_micro_batches

        self.is_first_stage = (pp_rank == 0)
        self.is_last_stage = (pp_rank == pp_size - 1)

        # PP 邻居的 global rank
        self.prev_rank = pp_ranks[pp_rank - 1] if pp_rank > 0 else None
        self.next_rank = pp_ranks[pp_rank + 1] if pp_rank < pp_size - 1 else None

    def _forward_step(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """执行一次前向"""
        return self.stage(input_tensor)

    def _send_forward(self, tensor: torch.Tensor):
        """发送激活值到下一个 stage"""
        if not self.is_last_stage:
            dist.send(tensor.detach().contiguous(), dst=self.next_rank)

    def _recv_forward(self, shape, dtype, device) -> torch.Tensor:
        """从上一个 stage 接收激活值"""
        if self.is_first_stage:
            return None
        tensor = torch.empty(shape, dtype=dtype, device=device)
        dist.recv(tensor, src=self.prev_rank)
        return tensor.requires_grad_(True)

    def _send_backward(self, tensor: torch.Tensor):
        """发送梯度到上一个 stage"""
        if not self.is_first_stage:
            dist.send(tensor.contiguous(), dst=self.prev_rank)

    def _recv_backward(self, shape, dtype, device) -> torch.Tensor:
        """从下一个 stage 接收梯度"""
        if self.is_last_stage:
            return None
        tensor = torch.empty(shape, dtype=dtype, device=device)
        dist.recv(tensor, src=self.next_rank)
        return tensor

    def run(
        self,
        data_iter,
        loss_fn: Callable,
        device: torch.device,
        activation_shape: tuple,
        dtype: torch.dtype = torch.bfloat16,
    ) -> float:
        """
        执行 1F1B 调度。

        参数:
            data_iter: 数据迭代器（只有 stage 0 需要）
            loss_fn: 损失函数（只有最后一个 stage 使用）
            device: 设备
            activation_shape: 激活值的形状 (micro_batch_size, seq_len, hidden_size)
            dtype: 数据类型

        返回:
            平均 loss (仅最后一个 stage 有效)
        """
        m = self.num_micro_batches
        p = self.pp_size
        num_warmup = p - 1

        # 存储激活值的队列
        input_queue = deque()
        output_queue = deque()
        total_loss = 0.0
        fwd_idx = 0
        bwd_idx = 0

        # ========== Warmup ==========
        for _ in range(min(num_warmup, m)):
            if self.is_first_stage:
                batch = next(data_iter)
                input_tensor = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            else:
                input_tensor = self._recv_forward(activation_shape, dtype, device)

            output_tensor = self._forward_step(input_tensor)
            self._send_forward(output_tensor)

            input_queue.append(input_tensor)
            output_queue.append(output_tensor)
            fwd_idx += 1

        # ========== Steady State ==========
        while fwd_idx < m:
            # Forward
            if self.is_first_stage:
                batch = next(data_iter)
                input_tensor = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            else:
                input_tensor = self._recv_forward(activation_shape, dtype, device)

            output_tensor = self._forward_step(input_tensor)
            self._send_forward(output_tensor)
            input_queue.append(input_tensor)
            output_queue.append(output_tensor)
            fwd_idx += 1

            # Backward
            if output_queue:
                bwd_input = input_queue.popleft()
                bwd_output = output_queue.popleft()

                if self.is_last_stage:
                    loss = loss_fn(bwd_output)
                    loss.backward()
                    total_loss += loss.item()
                else:
                    grad = self._recv_backward(activation_shape, dtype, device)
                    bwd_output.backward(grad)

                if not self.is_first_stage and bwd_input.grad is not None:
                    self._send_backward(bwd_input.grad)
                bwd_idx += 1

        # ========== Cooldown ==========
        while output_queue:
            bwd_input = input_queue.popleft()
            bwd_output = output_queue.popleft()

            if self.is_last_stage:
                loss = loss_fn(bwd_output)
                loss.backward()
                total_loss += loss.item()
            else:
                grad = self._recv_backward(activation_shape, dtype, device)
                bwd_output.backward(grad)

            if not self.is_first_stage and bwd_input.grad is not None:
                self._send_backward(bwd_input.grad)
            bwd_idx += 1

        avg_loss = total_loss / m if self.is_last_stage else 0.0
        return avg_loss
