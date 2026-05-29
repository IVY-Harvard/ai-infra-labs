"""Checkpoint 管理器

支持:
- 同步 / 异步保存
- 增量 checkpoint（仅保存变化的 shard）
- 自动上传到远端存储
- 断点续训恢复
"""

import os
import time
import json
import hashlib
import shutil
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from ..storage.backend import StorageBackend


@dataclass
class CheckpointMeta:
    """Checkpoint 元信息"""
    step: int
    timestamp: float
    local_path: str
    remote_key: Optional[str] = None
    shard_checksums: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    is_incremental: bool = False
    base_step: Optional[int] = None


class CheckpointManager:
    """Checkpoint 管理器

    典型用法：
        ckpt_mgr = CheckpointManager(
            local_dir="/nvme/checkpoints",
            backend=s3_backend,
            keep_local=3,
        )
        # 训练循环中
        ckpt_mgr.save(step=1000, state_dict=model.state_dict(), metrics={...})
        # 恢复
        state = ckpt_mgr.load_latest()
    """

    def __init__(self, local_dir: str,
                 backend: StorageBackend = None,
                 keep_local: int = 3,
                 async_upload: bool = True):
        self.local_dir = local_dir
        self.backend = backend
        self.keep_local = keep_local
        self.async_upload = async_upload

        self.history: List[CheckpointMeta] = []
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2)

        os.makedirs(local_dir, exist_ok=True)
        self._load_history()

    def _load_history(self):
        """从本地目录恢复历史记录"""
        meta_path = os.path.join(self.local_dir, "ckpt_history.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                records = json.load(f)
            self.history = [
                CheckpointMeta(**r) for r in records
            ]

    def _save_history(self):
        """持久化历史记录"""
        meta_path = os.path.join(self.local_dir, "ckpt_history.json")
        records = []
        for m in self.history:
            records.append({
                "step": m.step,
                "timestamp": m.timestamp,
                "local_path": m.local_path,
                "remote_key": m.remote_key,
                "shard_checksums": m.shard_checksums,
                "metrics": m.metrics,
                "is_incremental": m.is_incremental,
                "base_step": m.base_step,
            })
        with open(meta_path, "w") as f:
            json.dump(records, f, indent=2)

    def save(self, step: int, state_dict: Dict,
             metrics: Dict[str, float] = None,
             incremental: bool = False) -> CheckpointMeta:
        """保存 checkpoint

        Args:
            step: 当前训练步数
            state_dict: 模型状态字典
            metrics: 训练指标（loss, lr 等）
            incremental: 是否增量保存（仅保存与上一次不同的 shard）
        """
        ckpt_dir = os.path.join(self.local_dir, f"step_{step:08d}")
        os.makedirs(ckpt_dir, exist_ok=True)

        t_start = time.perf_counter()

        # 分 shard 保存
        shard_checksums = {}
        base_step = None

        if incremental and self.history:
            base_meta = self.history[-1]
            base_step = base_meta.step
            saved_count = 0
            skipped_count = 0

            for key, tensor_data in state_dict.items():
                data = self._serialize_tensor(tensor_data)
                checksum = hashlib.sha256(data).hexdigest()[:16]

                if checksum == base_meta.shard_checksums.get(key):
                    # 未变化，创建符号链接或跳过
                    skipped_count += 1
                    shard_checksums[key] = checksum
                    continue

                shard_path = os.path.join(ckpt_dir, f"{key}.shard")
                with open(shard_path, "wb") as f:
                    f.write(data)
                shard_checksums[key] = checksum
                saved_count += 1
        else:
            for key, tensor_data in state_dict.items():
                data = self._serialize_tensor(tensor_data)
                shard_path = os.path.join(ckpt_dir, f"{key}.shard")
                with open(shard_path, "wb") as f:
                    f.write(data)
                shard_checksums[key] = hashlib.sha256(data).hexdigest()[:16]

        duration = time.perf_counter() - t_start

        meta = CheckpointMeta(
            step=step,
            timestamp=time.time(),
            local_path=ckpt_dir,
            shard_checksums=shard_checksums,
            metrics=metrics or {},
            is_incremental=incremental and bool(self.history),
            base_step=base_step,
        )

        with self._lock:
            self.history.append(meta)
            self._save_history()

        # 异步上传到远端
        if self.backend and self.async_upload:
            self._executor.submit(self._upload_checkpoint, meta)
        elif self.backend:
            self._upload_checkpoint(meta)

        # 清理旧 checkpoint
        self._cleanup_local()

        return meta

    def _serialize_tensor(self, tensor) -> bytes:
        """序列化 tensor 数据"""
        if isinstance(tensor, bytes):
            return tensor
        # 模拟序列化（实际使用 torch.save 或 safetensors）
        import pickle
        return pickle.dumps(tensor)

    def _upload_checkpoint(self, meta: CheckpointMeta):
        """上传 checkpoint 到远端存储"""
        remote_key = f"checkpoints/step_{meta.step:08d}"

        ckpt_dir = meta.local_path
        if not os.path.isdir(ckpt_dir):
            return

        for filename in os.listdir(ckpt_dir):
            filepath = os.path.join(ckpt_dir, filename)
            with open(filepath, "rb") as f:
                data = f.read()
            self.backend.put(f"{remote_key}/{filename}", data)

        meta.remote_key = remote_key
        with self._lock:
            self._save_history()

    def _cleanup_local(self):
        """保留最近 N 个本地 checkpoint"""
        if len(self.history) <= self.keep_local:
            return

        to_remove = self.history[:-self.keep_local]
        for meta in to_remove:
            if os.path.isdir(meta.local_path):
                shutil.rmtree(meta.local_path, ignore_errors=True)

    def load_latest(self) -> Optional[Dict]:
        """加载最新 checkpoint"""
        if not self.history:
            return None
        return self.load_step(self.history[-1].step)

    def load_step(self, step: int) -> Optional[Dict]:
        """加载指定步数的 checkpoint"""
        meta = next((m for m in self.history if m.step == step), None)
        if meta is None:
            return None

        # 优先从本地加载
        if os.path.isdir(meta.local_path):
            return self._load_from_local(meta)

        # 从远端下载
        if self.backend and meta.remote_key:
            return self._load_from_remote(meta)

        return None

    def _load_from_local(self, meta: CheckpointMeta) -> Dict:
        """从本地目录恢复"""
        import pickle
        state_dict = {}
        for filename in os.listdir(meta.local_path):
            if filename.endswith(".shard"):
                key = filename[:-6]  # 去掉 .shard
                filepath = os.path.join(meta.local_path, filename)
                with open(filepath, "rb") as f:
                    state_dict[key] = pickle.loads(f.read())
        return state_dict

    def _load_from_remote(self, meta: CheckpointMeta) -> Dict:
        """从远端存储下载并恢复"""
        import pickle
        state_dict = {}
        # 简化实现：列举远端 key
        # 实际应使用 backend.list() 接口
        for key, checksum in meta.shard_checksums.items():
            remote_path = f"{meta.remote_key}/{key}.shard"
            data = self.backend.get(remote_path)
            if data:
                state_dict[key] = pickle.loads(data)
        return state_dict

    def list_checkpoints(self) -> List[Dict]:
        """列出所有 checkpoint"""
        return [
            {
                "step": m.step,
                "timestamp": m.timestamp,
                "local_exists": os.path.isdir(m.local_path),
                "remote_key": m.remote_key,
                "metrics": m.metrics,
                "incremental": m.is_incremental,
            }
            for m in self.history
        ]
