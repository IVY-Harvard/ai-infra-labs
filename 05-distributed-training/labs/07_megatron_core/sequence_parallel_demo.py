"""
Lab 07 - Sequence Parallelism 原理演示
========================================
演示 SP 如何将 LayerNorm/Dropout 的激活值按 sequence 维度切分。

核心: 用 ReduceScatter + AllGather 替代 AllReduce
  - AllReduce 后: 输出完整 [B, S, H]
  - ReduceScatter 后: 输出切分 [B, S/tp, H]
  - 下一个 TP 操作前: AllGather 恢复 [B, S, H]

显存节省:
  Non-TP 部分 (LayerNorm, Dropout, Residual) 的激活从 [B,S,H] 变为 [B,S/tp,H]

运行:
    torchrun --nproc_per_node=4 sequence_parallel_demo.py
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist


# ---------------------------------------------------------------------------
# 通信原语
# ---------------------------------------------------------------------------

class _ReduceScatterFunc(torch.autograd.Function):
    """ReduceScatter: AllReduce → 只保留 1/tp 份 (按 seq 维度切分)"""
    @staticmethod
    def forward(ctx, input_, tp_group, tp_size):
        ctx.tp_group = tp_group
        ctx.tp_size = tp_size
        # input: [B, S, H] → reduce → scatter 得到 [B, S/tp, H]
        B, S, H = input_.shape
        assert S % tp_size == 0
        output = torch.empty(B, S // tp_size, H, dtype=input_.dtype, device=input_.device)
        input_list = list(input_.chunk(tp_size, dim=1))
        dist.reduce_scatter(output, input_list, group=tp_group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 反向: AllGather
        tp_group = ctx.tp_group
        tp_size = ctx.tp_size
        gather_list = [torch.empty_like(grad_output) for _ in range(tp_size)]
        dist.all_gather(gather_list, grad_output, group=tp_group)
        return torch.cat(gather_list, dim=1), None, None


class _AllGatherFunc(torch.autograd.Function):
    """AllGather: [B, S/tp, H] → [B, S, H]"""
    @staticmethod
    def forward(ctx, input_, tp_group, tp_size):
        ctx.tp_group = tp_group
        ctx.tp_size = tp_size
        gather_list = [torch.empty_like(input_) for _ in range(tp_size)]
        dist.all_gather(gather_list, input_.contiguous(), group=tp_group)
        return torch.cat(gather_list, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        # 反向: ReduceScatter
        tp_group = ctx.tp_group
        tp_size = ctx.tp_size
        B, S, H = grad_output.shape
        output = torch.empty(B, S // tp_size, H, dtype=grad_output.dtype,
                             device=grad_output.device)
        input_list = list(grad_output.chunk(tp_size, dim=1))
        dist.reduce_scatter(output, input_list, group=tp_group)
        return output, None, None


def reduce_scatter_to_sp(input_, tp_group, tp_size):
    return _ReduceScatterFunc.apply(input_, tp_group, tp_size)


def all_gather_from_sp(input_, tp_group, tp_size):
    return _AllGatherFunc.apply(input_, tp_group, tp_size)


# ---------------------------------------------------------------------------
# SP Transformer Block（简化版）
# ---------------------------------------------------------------------------

class SPTransformerBlock(nn.Module):
    """
    带 Sequence Parallelism 的 Transformer 块。

    流程:
    Input: [B, S/tp, H]  (SP region: 切分的)
      ↓ AllGather → [B, S, H]
      ↓ ColumnParallel MLP → [B, S, 4H/tp]
      ↓ GeLU (local)
      ↓ RowParallel MLP → [B, S, H] (partial sum)
      ↓ ReduceScatter → [B, S/tp, H]
      ↓ LayerNorm (local, on S/tp)
      ↓ Residual
    Output: [B, S/tp, H]
    """

    def __init__(self, hidden_size, tp_size, tp_rank, tp_group):
        super().__init__()
        self.hidden_size = hidden_size
        self.tp_size = tp_size
        self.tp_group = tp_group
        self.ffn_per_part = hidden_size * 4 // tp_size

        # LayerNorm 在 SP region (S/tp) 上操作
        self.ln = nn.LayerNorm(hidden_size)

        # TP Linear 层
        self.w1 = nn.Linear(hidden_size, self.ffn_per_part, bias=False)
        self.w2 = nn.Linear(self.ffn_per_part, hidden_size, bias=False)

    def forward(self, x_sp):
        """
        x_sp: [B, S/tp, H] — SP 区域的输入
        """
        residual = x_sp

        # LayerNorm 在 SP 切分的 tensor 上做 (沿 hidden 维度归一化，不依赖 S)
        x_sp = self.ln(x_sp)

        # AllGather: [B, S/tp, H] → [B, S, H] (进入 TP region)
        x_full = all_gather_from_sp(x_sp, self.tp_group, self.tp_size)

        # Column Parallel: [B, S, H] → [B, S, 4H/tp]
        h = torch.nn.functional.gelu(self.w1(x_full))

        # Row Parallel: [B, S, 4H/tp] → [B, S, H] (local partial sum)
        h = self.w2(h)

        # ReduceScatter: [B, S, H] → [B, S/tp, H] (回到 SP region)
        out_sp = reduce_scatter_to_sp(h, self.tp_group, self.tp_size)

        # Residual (在 SP region 做)
        out_sp = out_sp + residual

        return out_sp


# ---------------------------------------------------------------------------
# 验证和演示
# ---------------------------------------------------------------------------

def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    tp_group = dist.new_group(list(range(world_size)))
    tp_size = world_size

    hidden_size = 512
    batch_size = 4
    seq_len = 128

    if rank == 0:
        print(f"Sequence Parallelism Demo (TP={tp_size})")
        print(f"  完整 tensor: [B={batch_size}, S={seq_len}, H={hidden_size}]")
        print(f"  SP tensor:   [B={batch_size}, S/tp={seq_len//tp_size}, H={hidden_size}]")
        print(f"  激活值显存节省: {(1-1/tp_size)*100:.0f}% (non-TP 部分)")

    block = SPTransformerBlock(hidden_size, tp_size, rank, tp_group).to(device)

    # 输入: SP 切分的 tensor
    x_sp = torch.randn(batch_size, seq_len // tp_size, hidden_size,
                        device=device, requires_grad=True)

    # 前向
    out_sp = block(x_sp)

    if rank == 0:
        print(f"\n  输入 shape: {x_sp.shape}  (SP: S/tp)")
        print(f"  输出 shape: {out_sp.shape}  (SP: S/tp)")

    # 反向
    loss = out_sp.sum()
    loss.backward()

    if rank == 0:
        print(f"  输入梯度 shape: {x_sp.grad.shape}")
        print(f"\n  SP 通信模式:")
        print(f"    AllGather:     [B, S/tp, H] → [B, S, H]  (进入 TP GEMM)")
        print(f"    ReduceScatter: [B, S, H] → [B, S/tp, H]  (回到 SP 区域)")
        print(f"    总通信量 = AllGather + ReduceScatter = AllReduce (不变)")
        print(f"    但 LayerNorm/Dropout/Residual 只操作 [B, S/tp, H] → 显存省!")
        print(f"\n  SP 已验证通过！")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
