"""
多级缓存管理器

L1: 本地 NVMe SSD
L2: 共享存储 / NFS
L3: 远端对象存储
"""

import os
import time
import json
import shutil
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass, asdict
from threading import Lock


@dataclass
class CacheEntry:
    key: str
    local_path: str
    size_bytes: int
    last_access: float
    access_count: int = 0
    tier: str = "l1"


class CacheManager:
    """多级缓存管理器"""

    def __init__(self, l1_dir: str, l2_dir: str = None,
                 l1_max_bytes: int = 500 * 1024**3):
        self.l1_dir = Path(l1_dir)
        self.l2_dir = Path(l2_dir) if l2_dir else None
        self.l1_max_bytes = l1_max_bytes
        self.l1_dir.mkdir(parents=True, exist_ok=True)
        if self.l2_dir:
            self.l2_dir.mkdir(parents=True, exist_ok=True)

        self.index: Dict[str, CacheEntry] = {}
        self.lock = Lock()
        self._load_index()

    def _index_path(self) -> Path:
        return self.l1_dir / ".cache_index.json"

    def _load_index(self):
        idx_path = self._index_path()
        if idx_path.exists():
            with open(idx_path) as f:
                data = json.load(f)
            for key, entry in data.items():
                self.index[key] = CacheEntry(**entry)

    def _save_index(self):
        with open(self._index_path(), "w") as f:
            json.dump({k: asdict(v) for k, v in self.index.items()}, f)

    def get(self, key: str) -> Optional[str]:
        """获取缓存文件路径"""
        with self.lock:
            if key in self.index:
                entry = self.index[key]
                if os.path.exists(entry.local_path):
                    entry.last_access = time.time()
                    entry.access_count += 1
                    self._save_index()
                    return entry.local_path
                else:
                    del self.index[key]
            return None

    def put(self, key: str, data: bytes) -> str:
        """写入缓存"""
        with self.lock:
            self._ensure_space(len(data))
            local_path = str(self.l1_dir / key.replace("/", "_"))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            with open(local_path, "wb") as f:
                f.write(data)

            self.index[key] = CacheEntry(
                key=key,
                local_path=local_path,
                size_bytes=len(data),
                last_access=time.time(),
                access_count=1,
                tier="l1",
            )
            self._save_index()
            return local_path

    def evict(self, key: str) -> bool:
        """手动驱逐缓存"""
        with self.lock:
            if key in self.index:
                entry = self.index[key]
                if os.path.exists(entry.local_path):
                    os.remove(entry.local_path)
                del self.index[key]
                self._save_index()
                return True
            return False

    def _ensure_space(self, needed_bytes: int):
        """确保有足够空间（LRU 淘汰）"""
        current = sum(e.size_bytes for e in self.index.values())
        while current + needed_bytes > self.l1_max_bytes and self.index:
            oldest_key = min(self.index, key=lambda k: self.index[k].last_access)
            entry = self.index[oldest_key]
            if os.path.exists(entry.local_path):
                os.remove(entry.local_path)
            current -= entry.size_bytes
            del self.index[oldest_key]

    def usage(self) -> Dict:
        """获取缓存使用信息"""
        total_bytes = sum(e.size_bytes for e in self.index.values())
        return {
            "entries": len(self.index),
            "total_bytes": total_bytes,
            "total_mb": total_bytes / 1024 / 1024,
            "max_mb": self.l1_max_bytes / 1024 / 1024,
            "usage_pct": total_bytes / self.l1_max_bytes * 100,
        }
