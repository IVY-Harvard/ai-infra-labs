"""
Lab 03 - 手写列并行 Linear
============================
从零实现 Megatron-style Column Parallel Linear。

核心理解:
  W ∈ R^{h×k}，按列切为 t 份: W_i ∈ R^{h × k/t}
  GPU i 计算: Y_i = X @ W_i    (无通信)
  Y_i 是输出的第 i 个列分片

通信:
  前向: 无（输入 X 每卡都有完整副本）
  反向: 需要 AllReduce ∂L/∂X（因为每卡只有部分 ∂L/∂X）

Megatron 的 f 操作符:
  前向 = identity（直接传递）
  反向 = AllReduce（聚合梯度）

运行:
    torchrun --nproc_per_node=4 column_parallel_linear.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# ---------------------------------------------------------------------------
# 1. f 操作符：前向 identity，反向 AllReduce
# ---------------------------------------------------------------------------

class _CopyToParallelRegion(torch.autograd.Function):
    """
    Megatron 的 f 操作符
    - 前向: 直接传递输入（identity）
    - 反向: AllReduce 梯度（因为每个 rank 只有部分梯度）
    """

    @staticmethod
    def forward(ctx, input_, tp_group):
        ctx.tp_group = tp_group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        # 所有 rank 的梯度求和
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=ctx.tp_group)
        return grad_output, None


def copy_to_parallel_region(input_, tp_group):
    return _CopyToParallelRegion.apply(input_, tp_group)


# ---------------------------------------------------------------------------
# 2. 列并行 Linear
# ---------------------------------------------------------------------------

class ColumnParallelLinear(nn.Module):
    """
    手写列并行 Linear。

    原始: Linear(in_features, out_features)
    列并行: 每 GPU 持有 Linear(in_features, out_features // tp_size)

    参数:
        in_features: 输入维度 (不切分)
        out_features: 输出维度 (按 TP 切分)
        tp_size: 张量并行大小
        tp_rank: 当前 rank 在 TP group 中的编号
        tp_group: TP 通信组
        gather_output: 是否在前向后 AllGather 输出
                       (如果后面接行并行层，通常为 False)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int,
        tp_rank: int,
        tp_group,
        bias: bool = True,
        gather_output: bool = False,
    ):
        super().__init__()
        assert out_features % tp_size == 0, \
            f"out_features ({out_features}) 必须能被 tp_size ({tp_size}) 整除"

        self.in_features = in_features
        self.out_features_per_partition = out_features // tp_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group
        self.gather_output = gather_output

        # 每个 rank 只存 out_features // tp_size 列的权重
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features_per_partition)
            )
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, in_features] — 完整输入
        output: [batch, seq_len, out_features // tp_size] — 列分片输出
        """
        # f 操作符: 前向 identity，反向 AllReduce
        x = copy_to_parallel_region(x, self.tp_group)

        # 本地矩阵乘法
        output = F.linear(x, self.weight, self.bias)

        if self.gather_output:
            # AllGather 所有分片，得到完整输出
            output_list = [torch.empty_like(output) for _ in range(self.tp_size)]
            dist.all_gather(output_list, output, group=self.tp_group)
            output = torch.cat(output_list, dim=-1)

        return output


# ---------------------------------------------------------------------------
# 3. 验证正确性
# ---------------------------------------------------------------------------

def verify_column_parallel():
    """验证列并行的计算结果与单卡相同"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    tp_group = dist.new_group(list(range(world_size)))

    in_features = 256
    out_features = 512
    batch_size = 4
    seq_len = 32

    # ---------- 列并行版本 ----------
    col_linear = ColumnParallelLinear(
        in_features, out_features, world_size, rank, tp_group,
        bias=True, gather_output=True,  # gather 以便与标准版本对比
    ).to(device)

    # ---------- 标准 Linear 作为参考 ----------
    if rank == 0:
        ref_linear = nn.Linear(in_features, out_features).to(device)
        # 将标准 Linear 的权重分发给各 rank
        weight_chunks = ref_linear.weight.data.chunk(world_size, dim=0)
        bias_chunks = ref_linear.bias.data.chunk(world_size, dim=0)
    else:
        weight_chunks = [None] * world_size
        bias_chunks = [None] * world_size

    # 分发权重
    local_weight = torch.empty(out_features // world_size, in_features, device=device)
    local_bias = torch.empty(out_features // world_size, device=device)
    dist.scatter(local_weight, weight_chunks if rank == 0 else None, src=0)
    dist.scatter(local_bias, bias_chunks if rank == 0 else None, src=0)

    col_linear.weight.data.copy_(local_weight)
    col_linear.bias.data.copy_(local_bias)

    # 相同输入
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, in_features, device=device)

    # 前向
    y_tp = col_linear(x)

    if rank == 0:
        y_ref = ref_linear(x)
        diff = (y_tp - y_ref).abs().max().item()
        print(f"[列并行验证] 前向差异: {diff:.2e}")
        assert diff < 1e-5, f"前向结果差异过大: {diff}"
        print("  PASS: 列并行前向与标准 Linear 一致")

    # 反向
    loss_tp = y_tp.sum()
    loss_tp.backward()

    if rank == 0:
        loss_ref = y_ref.sum()
        loss_ref.backward()

        # 收集所有 rank 的权重梯度
        # rank 0 的权重梯度对应 ref_linear.weight.grad 的前 1/tp_size 行
        local_grad = col_linear.weight.grad
        ref_grad_chunk = ref_linear.weight.grad[:out_features // world_size]
        grad_diff = (local_grad - ref_grad_chunk).abs().max().item()
        print(f"  权重梯度差异: {grad_diff:.2e}")
        assert grad_diff < 1e-4, f"梯度差异过大"
        print("  PASS: 列并行反向梯度正确")

    dist.barrier()
    if rank == 0:
        print("\n列并行 Linear 验证通过！")

    dist.destroy_process_group()


if __name__ == "__main__":
    verify_column_parallel()
