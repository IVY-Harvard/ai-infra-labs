"""
Checkpoint 保存器
==================
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, Any, Optional


class CheckpointSaver:
    """
    分布式 Checkpoint 保存器。
    每个 rank 保存自己的 state，支持并行 I/O。
    """

    def __init__(self, save_dir: str, max_keep: int = 3):
        self.save_dir = save_dir
        self.max_keep = max_keep
        os.makedirs(save_dir, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        rank: int,
        world_size: int,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """保存 checkpoint"""
        step_dir = os.path.join(self.save_dir, f"step_{step:06d}")
        os.makedirs(step_dir, exist_ok=True)

        ckpt = {
            "step": step,
            "rank": rank,
            "world_size": world_size,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if extra:
            ckpt.update(extra)

        save_path = os.path.join(step_dir, f"rank_{rank:03d}.pt")
        torch.save(ckpt, save_path)

        # rank 0 保存元信息
        if rank == 0:
            meta = {
                "step": step,
                "world_size": world_size,
            }
            torch.save(meta, os.path.join(step_dir, "meta.pt"))

        # 清理旧 checkpoint
        if rank == 0:
            self._cleanup()

    def _cleanup(self):
        """保留最近 max_keep 个 checkpoint"""
        dirs = sorted([
            d for d in os.listdir(self.save_dir)
            if d.startswith("step_") and os.path.isdir(os.path.join(self.save_dir, d))
        ])
        while len(dirs) > self.max_keep:
            old_dir = os.path.join(self.save_dir, dirs.pop(0))
            import shutil
            shutil.rmtree(old_dir, ignore_errors=True)

    def get_latest_step(self) -> int:
        """获取最新 checkpoint 的 step"""
        dirs = [
            d for d in os.listdir(self.save_dir)
            if d.startswith("step_") and os.path.isdir(os.path.join(self.save_dir, d))
        ]
        if not dirs:
            return -1
        latest = max(dirs, key=lambda d: int(d.split("_")[1]))
        return int(latest.split("_")[1])
