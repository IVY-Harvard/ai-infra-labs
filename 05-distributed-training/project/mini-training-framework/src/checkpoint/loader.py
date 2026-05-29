"""
Checkpoint 加载器
==================
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Dict, Any


class CheckpointLoader:
    """
    分布式 Checkpoint 加载器。
    支持:
      - 相同 world_size 加载
      - 不同 world_size 的 resharding (简化版)
    """

    def __init__(self, save_dir: str):
        self.save_dir = save_dir

    def load(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        rank: int,
        world_size: int,
        step: Optional[int] = None,
        device: str = "cpu",
    ) -> int:
        """
        加载 checkpoint。

        返回: 恢复的 step 编号，-1 表示无 checkpoint
        """
        if step is None:
            step = self._find_latest_step()
        if step < 0:
            return -1

        step_dir = os.path.join(self.save_dir, f"step_{step:06d}")
        if not os.path.exists(step_dir):
            return -1

        # 检查 world_size 是否匹配
        meta_path = os.path.join(step_dir, "meta.pt")
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location="cpu")
            saved_world_size = meta.get("world_size", world_size)
        else:
            saved_world_size = world_size

        if saved_world_size == world_size:
            # 直接加载对应 rank
            ckpt_path = os.path.join(step_dir, f"rank_{rank:03d}.pt")
            if not os.path.exists(ckpt_path):
                return -1
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        else:
            # Resharding: 简化处理，加载 rank 0 的参数并 broadcast
            ckpt_path = os.path.join(step_dir, "rank_000.pt")
            if not os.path.exists(ckpt_path):
                return -1
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            # broadcast 到所有 rank
            for p in model.parameters():
                dist.broadcast(p.data, src=0)

        return step

    def _find_latest_step(self) -> int:
        """找到最新的 checkpoint step"""
        if not os.path.exists(self.save_dir):
            return -1
        dirs = [
            d for d in os.listdir(self.save_dir)
            if d.startswith("step_") and os.path.isdir(os.path.join(self.save_dir, d))
        ]
        if not dirs:
            return -1
        latest = max(dirs, key=lambda d: int(d.split("_")[1]))
        return int(latest.split("_")[1])
