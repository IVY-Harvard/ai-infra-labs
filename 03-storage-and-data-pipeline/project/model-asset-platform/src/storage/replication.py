"""数据复制管理"""

import os
import time
import hashlib
import threading
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

from .backend import StorageBackend


class ReplicationManager:
    """数据复制管理器 — 确保数据在多个后端间同步"""

    def __init__(self, primary: StorageBackend,
                 replicas: List[StorageBackend] = None,
                 max_workers: int = 4):
        self.primary = primary
        self.replicas = replicas or []
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.replication_log: List[Dict] = []

    def replicate(self, key: str) -> Dict:
        """将主存储中的对象复制到所有副本"""
        data = self.primary.get(key)
        if data is None:
            return {"key": key, "status": "not_found"}

        results = {"key": key, "primary": True, "replicas": []}

        for i, replica in enumerate(self.replicas):
            try:
                success = replica.put(key, data)
                results["replicas"].append({
                    "index": i, "success": success
                })
            except Exception as e:
                results["replicas"].append({
                    "index": i, "success": False, "error": str(e)
                })

        self.replication_log.append({
            "key": key,
            "timestamp": time.time(),
            "results": results,
        })

        return results

    def replicate_async(self, key: str):
        """异步复制"""
        self.executor.submit(self.replicate, key)

    def verify_consistency(self, key: str) -> Dict:
        """验证主副本数据一致性"""
        primary_meta = self.primary.get_metadata(key)
        if primary_meta is None:
            return {"key": key, "status": "not_in_primary"}

        results = {"key": key, "primary_size": primary_meta.size, "replicas": []}

        for i, replica in enumerate(self.replicas):
            replica_meta = replica.get_metadata(key)
            if replica_meta is None:
                results["replicas"].append({"index": i, "consistent": False,
                                           "reason": "missing"})
            elif replica_meta.size != primary_meta.size:
                results["replicas"].append({"index": i, "consistent": False,
                                           "reason": "size_mismatch"})
            else:
                results["replicas"].append({"index": i, "consistent": True})

        return results

    def sync_all(self, prefix: str = "") -> Dict:
        """同步所有对象"""
        keys = self.primary.list_objects(prefix)
        results = {"total": len(keys), "success": 0, "failed": 0}

        for key in keys:
            result = self.replicate(key)
            if all(r.get("success") for r in result.get("replicas", [])):
                results["success"] += 1
            else:
                results["failed"] += 1

        return results
