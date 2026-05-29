"""分发引擎"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass

from ..storage.backend import StorageBackend
from ..storage.cache_manager import CacheManager


@dataclass
class DistributionTask:
    task_id: str
    model_name: str
    version: str
    target_nodes: List[str]
    status: str = "pending"
    progress: float = 0.0
    created_at: float = 0


class Distributor:
    """模型分发引擎

    分发策略：
    1. 检查目标节点缓存
    2. 未命中则从存储后端拉取
    3. 支持 P2P 加速（多节点场景）
    """

    def __init__(self, backend: StorageBackend,
                 cache: CacheManager = None):
        self.backend = backend
        self.cache = cache
        self.tasks: Dict[str, DistributionTask] = {}

    def distribute(self, model_key: str,
                   target_nodes: List[str] = None) -> Dict:
        """分发模型到目标节点"""
        import uuid
        task_id = str(uuid.uuid4())[:8]

        task = DistributionTask(
            task_id=task_id,
            model_name=model_key,
            version="latest",
            target_nodes=target_nodes or ["local"],
            created_at=time.time(),
        )
        self.tasks[task_id] = task

        # 检查缓存
        if self.cache:
            cached_path = self.cache.get(model_key)
            if cached_path:
                task.status = "completed"
                task.progress = 1.0
                return {"task_id": task_id, "status": "cache_hit",
                        "path": cached_path}

        # 从后端下载
        task.status = "downloading"
        data = self.backend.get(model_key)
        if data is None:
            task.status = "failed"
            return {"task_id": task_id, "status": "not_found"}

        # 缓存到本地
        if self.cache:
            local_path = self.cache.put(model_key, data)
        else:
            local_path = None

        task.status = "completed"
        task.progress = 1.0

        return {
            "task_id": task_id,
            "status": "completed",
            "size_bytes": len(data),
            "path": local_path,
        }

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        task = self.tasks.get(task_id)
        if task:
            return {
                "task_id": task.task_id,
                "status": task.status,
                "progress": task.progress,
            }
        return None
