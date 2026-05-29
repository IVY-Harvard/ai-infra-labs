"""
Lab 05 - 3D 并行配置生成器
============================
创建 TP / PP / DP 三维并行组，并验证通信正确性。

核心概念:
  总 GPU 数 N = TP_size × PP_size × DP_size
  GPU 排列成 3D 网格: [DP, PP, TP]

  8 GPU, TP=4, PP=2, DP=1:
    PP=0: [GPU 0, 1, 2, 3] — TP group 0
    PP=1: [GPU 4, 5, 6, 7] — TP group 1

    TP groups: {0,1,2,3}, {4,5,6,7}
    PP groups: {0,4}, {1,5}, {2,6}, {3,7}
    DP groups: {0}, {1}, ... (DP=1，无数据并行)

运行:
    torchrun --nproc_per_node=8 hybrid_parallel_config.py --tp 4 --pp 2
    torchrun --nproc_per_node=8 hybrid_parallel_config.py --tp 4 --pp 1 --dp 2
"""

import argparse
import os
import torch
import torch.distributed as dist


class ParallelConfig:
    """
    3D 并行配置管理器。
    负责创建和管理 TP / PP / DP 三组通信组。
    """

    def __init__(self, tp_size: int, pp_size: int, dp_size: int):
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.dp_size = dp_size

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        assert tp_size * pp_size * dp_size == self.world_size, \
            f"TP({tp_size}) × PP({pp_size}) × DP({dp_size}) = {tp_size*pp_size*dp_size} != world_size({self.world_size})"

        # 计算当前 rank 在 3D 网格中的坐标
        # 排列: rank = dp_idx * (tp_size * pp_size) + pp_idx * tp_size + tp_idx
        self.tp_rank = self.rank % tp_size
        self.pp_rank = (self.rank // tp_size) % pp_size
        self.dp_rank = self.rank // (tp_size * pp_size)

        # 创建通信组
        self.tp_group = None
        self.pp_group = None
        self.dp_group = None
        self._create_groups()

    def _create_groups(self):
        """创建所有必要的通信组"""
        tp_size = self.tp_size
        pp_size = self.pp_size
        dp_size = self.dp_size

        # --- TP 组: 相邻的 tp_size 个 rank ---
        # 同一个 PP stage、同一个 DP replica 内的 GPU
        for dp in range(dp_size):
            for pp in range(pp_size):
                ranks = [
                    dp * (tp_size * pp_size) + pp * tp_size + tp
                    for tp in range(tp_size)
                ]
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    self.tp_group = group
                    self._tp_ranks = ranks

        # --- PP 组: 同一 TP rank、同一 DP replica 的不同 stage ---
        for dp in range(dp_size):
            for tp in range(tp_size):
                ranks = [
                    dp * (tp_size * pp_size) + pp * tp_size + tp
                    for pp in range(pp_size)
                ]
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    self.pp_group = group
                    self._pp_ranks = ranks

        # --- DP 组: 同一 TP rank、同一 PP stage 的不同 replica ---
        for pp in range(pp_size):
            for tp in range(tp_size):
                ranks = [
                    dp * (tp_size * pp_size) + pp * tp_size + tp
                    for dp in range(dp_size)
                ]
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    self.dp_group = group
                    self._dp_ranks = ranks

    def get_pp_prev_rank(self):
        """获取 PP 中上一个 stage 的 rank"""
        if self.pp_rank == 0:
            return None
        return self._pp_ranks[self.pp_rank - 1]

    def get_pp_next_rank(self):
        """获取 PP 中下一个 stage 的 rank"""
        if self.pp_rank == self.pp_size - 1:
            return None
        return self._pp_ranks[self.pp_rank + 1]

    def summary(self):
        return (
            f"Rank {self.rank:2d}: "
            f"TP({self.tp_rank}/{self.tp_size}) "
            f"PP({self.pp_rank}/{self.pp_size}) "
            f"DP({self.dp_rank}/{self.dp_size}) | "
            f"TP group={self._tp_ranks} "
            f"PP group={self._pp_ranks} "
            f"DP group={self._dp_ranks}"
        )


def verify_groups(config: ParallelConfig, device):
    """验证通信组工作正常"""

    # 验证 TP AllReduce
    tp_tensor = torch.ones(4, device=device) * config.rank
    dist.all_reduce(tp_tensor, group=config.tp_group)
    expected_tp_sum = sum(config._tp_ranks)
    assert tp_tensor[0].item() == expected_tp_sum, \
        f"TP AllReduce 失败: got {tp_tensor[0].item()}, expected {expected_tp_sum}"

    # 验证 PP P2P
    if config.pp_size > 1:
        if config.pp_rank < config.pp_size - 1:
            send_tensor = torch.tensor([config.rank * 100.0], device=device)
            dist.send(send_tensor, dst=config.get_pp_next_rank())

        if config.pp_rank > 0:
            recv_tensor = torch.zeros(1, device=device)
            dist.recv(recv_tensor, src=config.get_pp_prev_rank())
            expected = config.get_pp_prev_rank() * 100.0
            assert recv_tensor.item() == expected

    # 验证 DP AllReduce (if DP > 1)
    if config.dp_size > 1:
        dp_tensor = torch.ones(4, device=device) * config.rank
        dist.all_reduce(dp_tensor, group=config.dp_group)
        expected_dp_sum = sum(config._dp_ranks)
        assert dp_tensor[0].item() == expected_dp_sum

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--dp", type=int, default=None)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dp_size = args.dp or (world_size // (args.tp * args.pp))

    if rank == 0:
        print(f"3D 并行配置: TP={args.tp}, PP={args.pp}, DP={dp_size}")
        print(f"总 GPU: {world_size}\n")

    config = ParallelConfig(args.tp, args.pp, dp_size)

    # 逐个 rank 打印
    for r in range(world_size):
        if rank == r:
            print(config.summary())
        dist.barrier()

    # 验证
    ok = verify_groups(config, device)
    dist.barrier()

    if rank == 0:
        print(f"\n所有通信组验证通过！")
        print(f"\n通信需求分析:")
        print(f"  TP AllReduce: 在 {{{','.join(map(str, config._tp_ranks))}}} 间 → 需要 NVLink")
        if config.pp_size > 1:
            print(f"  PP P2P: {config._pp_ranks[0]} ↔ {config._pp_ranks[1]} → 可用低带宽")
        if config.dp_size > 1:
            print(f"  DP AllReduce: 在 {{{','.join(map(str, config._dp_ranks))}}} 间 → 可 overlap")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
