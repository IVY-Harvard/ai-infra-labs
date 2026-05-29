"""
异步 Checkpoint 保存器
=======================
使用后台线程保存，不阻塞训练。
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional

import torch
import torch.nn as nn


class AsyncCheckpointSaver:
    """
    异步 Checkpoint 保存。
    核心思路: snapshot (CPU copy) → 后台线程写磁盘
    """

    def __init__(self, save_dir: str, max_keep: int = 3, num_workers: int = 2):
        self.save_dir = save_dir
        self.max_keep = max_keep
        os.makedirs(save_dir, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self._pending = []
        self._lock = threading.Lock()

    def save_async(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        rank: int,
        world_size: int,
    ):
        """
        异步保存: 先 snapshot 到 CPU，再后台写入。
        snapshot 阻塞时间很短 (几 ms)，写磁盘在后台完成。
        """
        # CPU snapshot (快速, 阻塞主线程)
        state = self._snapshot(model, optimizer, step, rank, world_size)

        # 后台写入 (慢, 不阻塞)
        step_dir = os.path.join(self.save_dir, f"step_{step:06d}")
        os.makedirs(step_dir, exist_ok=True)
        save_path = os.path.join(step_dir, f"rank_{rank:03d}.pt")

        future = self.executor.submit(self._write_worker, state, save_path)
        with self._lock:
            self._pending.append(future)

        # rank 0 也异步写 meta
        if rank == 0:
            meta = {"step": step, "world_size": world_size}
            meta_path = os.path.join(step_dir, "meta.pt")
            self.executor.submit(self._write_worker, meta, meta_path)

    def _snapshot(self, model, optimizer, step, rank, world_size) -> Dict[str, Any]:
        """快照: 将所有 tensor copy 到 CPU"""
        model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        opt_state = {"param_groups": optimizer.state_dict()["param_groups"]}
        opt_state["state"] = {}
        for k, v in optimizer.state_dict()["state"].items():
            opt_state["state"][k] = {
                sk: sv.cpu().clone() if isinstance(sv, torch.Tensor) else sv
                for sk, sv in v.items()
            }

        return {
            "step": step,
            "rank": rank,
            "world_size": world_size,
            "model_state_dict": model_state,
            "optimizer_state_dict": opt_state,
        }

    @staticmethod
    def _write_worker(data: dict, path: str):
        """后台线程写入磁盘"""
        torch.save(data, path)

    def wait_all(self):
        """等待所有异步保存完成"""
        with self._lock:
            for f in self._pending:
                f.result()
            self._pending.clear()

    def shutdown(self):
        self.wait_all()
        self.executor.shutdown(wait=True)
