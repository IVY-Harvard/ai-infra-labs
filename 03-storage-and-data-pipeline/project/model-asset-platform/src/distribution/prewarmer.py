"""模型预热管理器"""

import time
import threading
from typing import Dict, List
from dataclasses import dataclass
from queue import PriorityQueue
from concurrent.futures import ThreadPoolExecutor

from ..storage.backend import StorageBackend
from ..storage.cache_manager import CacheManager


@dataclass
class PrewarmTask:
    priority: int
    model_key: str
    deadline: float
    status: str = "pending"

    def __lt__(self, other):
        return self.priority < other.priority


class Prewarmer:
    """模型预热管理 — 提前将模型缓存到本地"""

    def __init__(self, backend: StorageBackend,
                 cache: CacheManager,
                 max_workers: int = 3):
        self.backend = backend
        self.cache = cache
        self.queue = PriorityQueue()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.completed: List[Dict] = []

    def submit(self, model_key: str, deadline_seconds: float = 60,
               priority: int = None):
        """提交预热任务"""
        if priority is None:
            priority = max(0, int(deadline_seconds / 10))

        task = PrewarmTask(
            priority=priority,
            model_key=model_key,
            deadline=time.time() + deadline_seconds,
        )
        self.queue.put(task)

    def _execute(self, task: PrewarmTask):
        """执行预热"""
        if self.cache.get(task.model_key):
            task.status = "already_cached"
            return

        data = self.backend.get(task.model_key)
        if data:
            self.cache.put(task.model_key, data)
            task.status = "completed"
        else:
            task.status = "not_found"

        self.completed.append({
            "model_key": task.model_key,
            "status": task.status,
            "timestamp": time.time(),
        })

    def process_queue(self):
        """处理队列中的预热任务"""
        while not self.queue.empty():
            task = self.queue.get()
            if time.time() > task.deadline:
                task.status = "expired"
                continue
            self.executor.submit(self._execute, task)

    def status(self) -> Dict:
        return {
            "pending": self.queue.qsize(),
            "completed": len(self.completed),
        }
