"""
Lab 03 - TP Transformer Block
===============================
用手写的列并行和行并行组合搭建 Transformer 的 MLP 和 Self-Attention 块。

布局:
  MLP:       列并行(W1) → GeLU → 行并行(W2) → AllReduce → 残差
  Attention: 列并行(Wq,Wk,Wv) → Attention → 行并行(Wo) → AllReduce → 残差

每层通信: 前向 2 次 AllReduce (MLP + Attention)
         反向 2 次 AllReduce (MLP 输入梯度 + Attention 输入梯度)

运行:
    torchrun --nproc_per_node=4 tp_transformer_block.py
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# ---------------------------------------------------------------------------
# 通信操作符（复用 column/row_parallel_linear.py 中的实现）
# ---------------------------------------------------------------------------

class _CopyToParallelRegion(torch.autograd.Function):
    """f: forward=identity, backward=AllReduce"""
    @staticmethod
    def forward(ctx, input_, tp_group):
        ctx.tp_group = tp_group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=ctx.tp_group)
        return grad_output, None


class _ReduceFromParallelRegion(torch.autograd.Function):
    """g: forward=AllReduce, backward=identity"""
    @staticmethod
    def forward(ctx, input_, tp_group):
        ctx.tp_group = tp_group
        dist.all_reduce(input_, op=dist.ReduceOp.SUM, group=tp_group)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


# ---------------------------------------------------------------------------
# TP Linear 层
# ---------------------------------------------------------------------------

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_size, tp_rank, tp_group, bias=True):
        super().__init__()
        assert out_features % tp_size == 0
        self.out_per_part = out_features // tp_size
        self.tp_group = tp_group
        self.weight = nn.Parameter(torch.empty(self.out_per_part, in_features))
        self.bias = nn.Parameter(torch.empty(self.out_per_part)) if bias else None
        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        x = _CopyToParallelRegion.apply(x, self.tp_group)
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_size, tp_rank, tp_group, bias=True):
        super().__init__()
        assert in_features % tp_size == 0
        self.in_per_part = in_features // tp_size
        self.tp_group = tp_group
        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_part))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        output = F.linear(x, self.weight)
        output = _ReduceFromParallelRegion.apply(output, self.tp_group)
        if self.bias is not None:
            output = output + self.bias
        return output


# ---------------------------------------------------------------------------
# TP MLP Block
# ---------------------------------------------------------------------------

class TPParallelMLP(nn.Module):
    """
    Tensor Parallel MLP:
        h = GeLU(x @ W1)  — W1 列并行
        y = h @ W2         — W2 行并行 → AllReduce

    GeLU 在列并行输出上本地计算（逐元素操作可以在分片上独立做）
    """

    def __init__(self, hidden_size, ffn_size, tp_size, tp_rank, tp_group, dropout=0.1):
        super().__init__()
        # W1: [hidden_size, ffn_size] → 列并行
        self.w1 = ColumnParallelLinear(hidden_size, ffn_size, tp_size, tp_rank, tp_group)
        # W2: [ffn_size, hidden_size] → 行并行
        self.w2 = RowParallelLinear(ffn_size, hidden_size, tp_size, tp_rank, tp_group)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.gelu(self.w1(x))    # 列并行 + GeLU(本地)
        h = self.w2(h)             # 行并行 + AllReduce
        h = self.dropout(h)
        return h


# ---------------------------------------------------------------------------
# TP Self-Attention Block
# ---------------------------------------------------------------------------

class TPParallelSelfAttention(nn.Module):
    """
    Tensor Parallel Multi-Head Attention:
        Q, K, V = x @ Wq, x @ Wk, x @ Wv  — 列并行（每卡负责 heads/tp_size 个 head）
        attn = softmax(Q @ K^T / sqrt(d)) @ V  — 本地计算（head 维度独立）
        out = attn @ Wo  — 行并行 → AllReduce
    """

    def __init__(self, hidden_size, num_heads, tp_size, tp_rank, tp_group, dropout=0.1):
        super().__init__()
        assert num_heads % tp_size == 0, \
            f"num_heads ({num_heads}) 必须能被 tp_size ({tp_size}) 整除"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.tp_size = tp_size
        self.heads_per_partition = num_heads // tp_size
        self.head_dim = hidden_size // num_heads

        # Q, K, V 投影: 列并行
        self.wq = ColumnParallelLinear(hidden_size, hidden_size, tp_size, tp_rank, tp_group, bias=False)
        self.wk = ColumnParallelLinear(hidden_size, hidden_size, tp_size, tp_rank, tp_group, bias=False)
        self.wv = ColumnParallelLinear(hidden_size, hidden_size, tp_size, tp_rank, tp_group, bias=False)

        # 输出投影: 行并行
        self.wo = RowParallelLinear(hidden_size, hidden_size, tp_size, tp_rank, tp_group, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, S, _ = x.shape

        # Q, K, V 投影 (列并行，无通信)
        q = self.wq(x)  # [B, S, hidden_size // tp_size]
        k = self.wk(x)
        v = self.wv(x)

        # Reshape: [B, S, heads_per_part, head_dim] → [B, heads_per_part, S, head_dim]
        q = q.view(B, S, self.heads_per_partition, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.heads_per_partition, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.heads_per_partition, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention (本地计算)
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        if mask is not None:
            scores = scores.masked_fill(mask == float("-inf"), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)

        # Reshape back: [B, heads_per_part, S, head_dim] → [B, S, hidden_size // tp_size]
        context = context.transpose(1, 2).contiguous().view(B, S, -1)

        # 输出投影 (行并行, AllReduce)
        output = self.wo(context)
        return output


# ---------------------------------------------------------------------------
# 完整 TP Transformer Block
# ---------------------------------------------------------------------------

class TPTransformerBlock(nn.Module):
    """
    [LayerNorm] → [TP Attention] → [Residual]
                                  ↓
    [LayerNorm] → [TP MLP] → [Residual]

    每层通信: 前向 2 次 AllReduce, 反向 2 次 AllReduce
    """

    def __init__(self, hidden_size, num_heads, ffn_size, tp_size, tp_rank, tp_group, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = TPParallelSelfAttention(hidden_size, num_heads, tp_size, tp_rank, tp_group, dropout)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = TPParallelMLP(hidden_size, ffn_size, tp_size, tp_rank, tp_group, dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------

def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    tp_group = dist.new_group(list(range(world_size)))

    hidden_size = 512
    num_heads = 8
    ffn_size = hidden_size * 4
    batch_size = 4
    seq_len = 64

    if rank == 0:
        print(f"TP Transformer Block 测试 (TP={world_size})")
        print(f"  hidden_size={hidden_size}, num_heads={num_heads}")
        print(f"  heads/GPU = {num_heads // world_size}")

    block = TPTransformerBlock(
        hidden_size, num_heads, ffn_size, world_size, rank, tp_group
    ).to(device)

    # 每卡的参数量
    local_params = sum(p.numel() for p in block.parameters())
    total_params = torch.tensor([local_params], device=device)
    dist.all_reduce(total_params)

    if rank == 0:
        print(f"  每卡参数: {local_params / 1e6:.2f}M")
        print(f"  总参数量: {total_params.item() / 1e6:.2f}M (约为 TP=1 的 {total_params.item() / local_params:.1f}x)")

    # 前向 + 反向
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, requires_grad=True)
    mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)

    output = block(x, mask)
    loss = output.sum()
    loss.backward()

    if rank == 0:
        print(f"  输出 shape: {output.shape}")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  输入梯度 norm: {x.grad.norm().item():.4f}")
        print("\n  TP Transformer Block 前向+反向通过！")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
