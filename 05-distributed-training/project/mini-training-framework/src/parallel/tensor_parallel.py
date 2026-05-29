"""
张量并行封装 — Megatron-style Column/Row Parallel
===================================================
提供 TP 的核心组件:
  - ColumnParallelLinear: 按列切分权重
  - RowParallelLinear: 按行切分权重
  - f/g 通信操作符
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional
import math


# ---------------------------------------------------------------------------
# 通信操作符
# ---------------------------------------------------------------------------

class _CopyToTPRegion(torch.autograd.Function):
    """f 操作符: forward=identity, backward=AllReduce"""
    @staticmethod
    def forward(ctx, input_, tp_group):
        ctx.tp_group = tp_group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=ctx.tp_group)
        return grad_output, None


class _ReduceFromTPRegion(torch.autograd.Function):
    """g 操作符: forward=AllReduce, backward=identity"""
    @staticmethod
    def forward(ctx, input_, tp_group):
        dist.all_reduce(input_, op=dist.ReduceOp.SUM, group=tp_group)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _ReduceScatterFromTPRegion(torch.autograd.Function):
    """ReduceScatter (用于 Sequence Parallelism)"""
    @staticmethod
    def forward(ctx, input_, tp_group, tp_size):
        ctx.tp_group = tp_group
        ctx.tp_size = tp_size
        B, S, H = input_.shape
        output = torch.empty(B, S // tp_size, H, dtype=input_.dtype, device=input_.device)
        dist.reduce_scatter(output, list(input_.chunk(tp_size, dim=1)), group=tp_group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        gather_list = [torch.empty_like(grad_output) for _ in range(ctx.tp_size)]
        dist.all_gather(gather_list, grad_output, group=ctx.tp_group)
        return torch.cat(gather_list, dim=1), None, None


class _AllGatherFromTPRegion(torch.autograd.Function):
    """AllGather (用于 Sequence Parallelism)"""
    @staticmethod
    def forward(ctx, input_, tp_group, tp_size):
        ctx.tp_group = tp_group
        ctx.tp_size = tp_size
        gather_list = [torch.empty_like(input_) for _ in range(tp_size)]
        dist.all_gather(gather_list, input_.contiguous(), group=tp_group)
        return torch.cat(gather_list, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        B, S, H = grad_output.shape
        output = torch.empty(B, S // ctx.tp_size, H,
                             dtype=grad_output.dtype, device=grad_output.device)
        dist.reduce_scatter(output, list(grad_output.chunk(ctx.tp_size, dim=1)),
                            group=ctx.tp_group)
        return output, None, None


# ---------------------------------------------------------------------------
# TP Linear 层
# ---------------------------------------------------------------------------

class ColumnParallelLinear(nn.Module):
    """
    列并行 Linear: 按列切分输出维度。

    原始: Linear(in_features, out_features)
    每 GPU: Linear(in_features, out_features // tp_size)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int,
        tp_rank: int,
        tp_group: dist.ProcessGroup,
        bias: bool = True,
        gather_output: bool = False,
    ):
        super().__init__()
        assert out_features % tp_size == 0
        self.out_features_per_part = out_features // tp_size
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.gather_output = gather_output

        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_part, in_features)
        )
        self.bias = nn.Parameter(
            torch.empty(self.out_features_per_part)
        ) if bias else None

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _CopyToTPRegion.apply(x, self.tp_group)
        output = F.linear(x, self.weight, self.bias)
        if self.gather_output:
            output_list = [torch.empty_like(output) for _ in range(self.tp_size)]
            dist.all_gather(output_list, output, group=self.tp_group)
            output = torch.cat(output_list, dim=-1)
        return output


class RowParallelLinear(nn.Module):
    """
    行并行 Linear: 按行切分输入维度。

    原始: Linear(in_features, out_features)
    每 GPU: Linear(in_features // tp_size, out_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int,
        tp_rank: int,
        tp_group: dist.ProcessGroup,
        bias: bool = True,
        input_is_parallel: bool = True,
    ):
        super().__init__()
        assert in_features % tp_size == 0
        self.in_features_per_part = in_features // tp_size
        self.tp_group = tp_group
        self.input_is_parallel = input_is_parallel

        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_part)
        )
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = F.linear(x, self.weight)
        output = _ReduceFromTPRegion.apply(output, self.tp_group)
        if self.bias is not None:
            output = output + self.bias
        return output


# ---------------------------------------------------------------------------
# TP Transformer 组件
# ---------------------------------------------------------------------------

class TPParallelMLP(nn.Module):
    """TP MLP: Column(W1) → GeLU → Row(W2) → AllReduce"""

    def __init__(self, hidden_size, ffn_size, tp_size, tp_rank, tp_group, dropout=0.1):
        super().__init__()
        self.w1 = ColumnParallelLinear(hidden_size, ffn_size, tp_size, tp_rank, tp_group)
        self.w2 = RowParallelLinear(ffn_size, hidden_size, tp_size, tp_rank, tp_group)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.gelu(self.w1(x))))


class TPParallelAttention(nn.Module):
    """TP Self-Attention: Column(Wq,Wk,Wv) → Attention → Row(Wo) → AllReduce"""

    def __init__(self, hidden_size, num_heads, tp_size, tp_rank, tp_group, dropout=0.1):
        super().__init__()
        assert num_heads % tp_size == 0
        self.num_heads_per_part = num_heads // tp_size
        self.head_dim = hidden_size // num_heads
        self.tp_group = tp_group

        self.wq = ColumnParallelLinear(hidden_size, hidden_size, tp_size, tp_rank, tp_group, bias=False)
        self.wk = ColumnParallelLinear(hidden_size, hidden_size, tp_size, tp_rank, tp_group, bias=False)
        self.wv = ColumnParallelLinear(hidden_size, hidden_size, tp_size, tp_rank, tp_group, bias=False)
        self.wo = RowParallelLinear(hidden_size, hidden_size, tp_size, tp_rank, tp_group, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, S, _ = x.shape
        q = self.wq(x).view(B, S, self.num_heads_per_part, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, S, self.num_heads_per_part, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, S, self.num_heads_per_part, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        if mask is not None:
            scores = scores.masked_fill(mask == float("-inf"), float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, -1)
        return self.wo(context)


class TPTransformerLayer(nn.Module):
    """完整的 TP Transformer 层"""

    def __init__(self, hidden_size, num_heads, ffn_size, tp_size, tp_rank, tp_group, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = TPParallelAttention(hidden_size, num_heads, tp_size, tp_rank, tp_group, dropout)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = TPParallelMLP(hidden_size, ffn_size, tp_size, tp_rank, tp_group, dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x
