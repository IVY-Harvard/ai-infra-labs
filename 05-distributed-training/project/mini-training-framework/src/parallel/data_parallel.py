"""
数据并行封装 — DDP / FSDP 统一接口
=====================================
提供数据并行的统一封装，支持:
  - DDP (DistributedDataParallel)
  - FSDP (FullyShardedDataParallel) with FULL_SHARD / SHARD_GRAD_OP
"""

import functools
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


class DataParallelWrapper:
    """
    数据并行统一封装。

    用法:
        wrapper = DataParallelWrapper(strategy="fsdp", dp_group=dp_group)
        model = wrapper.wrap(model, device)
    """

    STRATEGIES = {
        "ddp": "DDP",
        "fsdp_full": "FSDP FULL_SHARD (ZeRO-3)",
        "fsdp_grad_op": "FSDP SHARD_GRAD_OP (ZeRO-2)",
        "none": "No data parallel",
    }

    def __init__(
        self,
        strategy: str = "ddp",
        dp_group: Optional[dist.ProcessGroup] = None,
        mixed_precision: bool = True,
        transformer_layer_cls: Optional[set] = None,
    ):
        assert strategy in self.STRATEGIES, \
            f"Unknown strategy: {strategy}. Choose from {list(self.STRATEGIES.keys())}"
        self.strategy = strategy
        self.dp_group = dp_group
        self.mixed_precision = mixed_precision
        self.transformer_layer_cls = transformer_layer_cls

    def wrap(self, model: nn.Module, device: torch.device) -> nn.Module:
        """将模型包装为数据并行版本"""
        if self.strategy == "none":
            return model.to(device)

        elif self.strategy == "ddp":
            model = model.to(device)
            return DDP(
                model,
                device_ids=[device.index],
                process_group=self.dp_group,
            )

        elif self.strategy.startswith("fsdp"):
            sharding = {
                "fsdp_full": ShardingStrategy.FULL_SHARD,
                "fsdp_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            }[self.strategy]

            mp_policy = None
            if self.mixed_precision:
                mp_policy = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                )

            wrap_policy = None
            if self.transformer_layer_cls:
                wrap_policy = functools.partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=self.transformer_layer_cls,
                )

            return FSDP(
                model,
                sharding_strategy=sharding,
                mixed_precision=mp_policy,
                auto_wrap_policy=wrap_policy,
                process_group=self.dp_group,
                device_id=device,
            )

        raise ValueError(f"Unknown strategy: {self.strategy}")

    def clip_grad_norm(self, model: nn.Module, max_norm: float):
        """统一的梯度裁剪接口"""
        if isinstance(model, FSDP):
            model.clip_grad_norm_(max_norm)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
