"""
Lab 03 - 手写行并行 Linear
============================
从零实现 Megatron-style Row Parallel Linear。

核心理解:
  W ∈ R^{k×h}，按行切为 t 份: W_i ∈ R^{k/t × h}
  输入也按列切分: X_i ∈ R^{b × k/t}（来自上一层列并行的输出）
  GPU i 计算: Y_i = X_i @ W_i  (本地)
  最终: Y = AllReduce(Y_0 + Y_1 + ... + Y_{t-1})

通信:
  前向: AllReduce 输出（求和）
  反向: 无额外通信（梯度直接传回列并行层）

Megatron 的 g 操作符:
  前向 = AllReduce
  反向 = identity

运行:
    torchrun --nproc_per_node=4 row_parallel_linear.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# ---------------------------------------------------------------------------
# 1. g 操作符：前向 AllReduce，反向 identity
# ---------------------------------------------------------------------------

class _ReduceFromParallelRegion(torch.autograd.Function):
    """
    Megatron 的 g 操作符
    - 前向: AllReduce（求和各 rank 的部分输出）
    - 反向: 直接传递梯度（identity）
    """

    @staticmethod
    def forward(ctx, input_, tp_group):
        ctx.tp_group = tp_group
        dist.all_reduce(input_, op=dist.ReduceOp.SUM, group=tp_group)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        # identity: 梯度直接传回
        return grad_output, None


def reduce_from_parallel_region(input_, tp_group):
    return _ReduceFromParallelRegion.apply(input_, tp_group)


# ---------------------------------------------------------------------------
# 2. 行并行 Linear
# ---------------------------------------------------------------------------

class RowParallelLinear(nn.Module):
    """
    手写行并行 Linear。

    原始: Linear(in_features, out_features)
    行并行: 每 GPU 持有 Linear(in_features // tp_size, out_features)

    参数:
        in_features: 输入维度 (按 TP 切分)
        out_features: 输出维度 (不切分)
        tp_size: 张量并行大小
        tp_rank: 当前 rank 在 TP group 中的编号
        tp_group: TP 通信组
        input_is_parallel: 输入是否已经是并行分片
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int,
        tp_rank: int,
        tp_group,
        bias: bool = True,
        input_is_parallel: bool = True,
    ):
        super().__init__()
        assert in_features % tp_size == 0, \
            f"in_features ({in_features}) 必须能被 tp_size ({tp_size}) 整除"

        self.in_features_per_partition = in_features // tp_size
        self.out_features = out_features
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group
        self.input_is_parallel = input_is_parallel

        # 每个 rank 只存 in_features // tp_size 行的权重
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )
        if bias:
            # bias 只需要存一份完整的，AllReduce 后加
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, in_features // tp_size] — 输入分片
        output: [batch, seq_len, out_features] — 完整输出
        """
        if not self.input_is_parallel:
            # 如果输入不是分片的，先 scatter
            chunks = x.chunk(self.tp_size, dim=-1)
            x = chunks[self.tp_rank]

        # 本地矩阵乘法: 得到部分和
        output_local = F.linear(x, self.weight)

        # g 操作符: AllReduce 求和得到完整输出
        output = reduce_from_parallel_region(output_local, self.tp_group)

        # bias 在 AllReduce 之后加（只加一次）
        if self.bias is not None:
            output = output + self.bias

        return output


# ---------------------------------------------------------------------------
# 3. 验证正确性
# ---------------------------------------------------------------------------

def verify_row_parallel():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    tp_group = dist.new_group(list(range(world_size)))

    in_features = 512
    out_features = 256
    batch_size = 4
    seq_len = 32

    # ---------- 行并行版本 ----------
    row_linear = RowParallelLinear(
        in_features, out_features, world_size, rank, tp_group,
        bias=True, input_is_parallel=True,
    ).to(device)

    # ---------- 参考 Linear ----------
    torch.manual_seed(42)
    if rank == 0:
        ref_linear = nn.Linear(in_features, out_features).to(device)
        # 按行切分权重
        weight_chunks = ref_linear.weight.data.chunk(world_size, dim=1)
    else:
        weight_chunks = [None] * world_size

    # 分发权重
    local_weight = torch.empty(out_features, in_features // world_size, device=device)
    dist.scatter(local_weight, weight_chunks if rank == 0 else None, src=0)
    row_linear.weight.data.copy_(local_weight)

    # 分发 bias (只有 rank 0 有完整 bias)
    if rank == 0:
        full_bias = ref_linear.bias.data.clone()
    else:
        full_bias = torch.empty(out_features, device=device)
    dist.broadcast(full_bias, src=0)
    row_linear.bias.data.copy_(full_bias)

    # 相同输入
    torch.manual_seed(42)
    x_full = torch.randn(batch_size, seq_len, in_features, device=device)

    # 行并行: 输入按列切分
    x_local = x_full.chunk(world_size, dim=-1)[rank]

    # 前向
    y_tp = row_linear(x_local)

    if rank == 0:
        y_ref = ref_linear(x_full)
        diff = (y_tp - y_ref).abs().max().item()
        print(f"[行并行验证] 前向差异: {diff:.2e}")
        assert diff < 1e-4, f"前向结果差异过大: {diff}"
        print("  PASS: 行并行前向与标准 Linear 一致")

    # 反向
    loss_tp = y_tp.sum()
    loss_tp.backward()

    if rank == 0:
        loss_ref = y_ref.sum()
        loss_ref.backward()

        local_grad = row_linear.weight.grad
        ref_grad_chunk = ref_linear.weight.grad[:, :in_features // world_size]
        grad_diff = (local_grad - ref_grad_chunk).abs().max().item()
        print(f"  权重梯度差异: {grad_diff:.2e}")
        assert grad_diff < 1e-3, f"梯度差异过大"
        print("  PASS: 行并行反向梯度正确")

    dist.barrier()
    if rank == 0:
        print("\n行并行 Linear 验证通过！")

    dist.destroy_process_group()


if __name__ == "__main__":
    verify_row_parallel()
