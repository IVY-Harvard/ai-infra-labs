"""
多级缓存模型加载器

实现 L1(本地SSD) → L2(节点缓存) → L3(远端存储) 的多级加载策略。
每次加载优先从最快的层级获取，未命中时逐级回退。

用法：
    python multi_level_cache.py --cache-dir /nvme/model-cache --model-name llama-7b
"""

import os
import time
import json
import shutil
import hashlib
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List


@dataclass
class CacheEntry:
    """缓存条目"""
    model_name: str
    version: str
    format: str
    local_path: str
    size_bytes: int
    last_access: float
    access_count: int = 0
    checksum: str = ""


class MultiLevelCache:
    """多级缓存管理器

    层级：
    - L1: 本地 NVMe SSD（最快，容量有限）
    - L2: 节点间共享存储 / NFS（中等速度，容量中等）
    - L3: 远端对象存储 S3/MinIO（最慢，容量无限）

    策略：
    - 加载时：L1 → L2 → L3，命中即返回
    - 加载后：将数据缓存到 L1（和 L2）
    - 淘汰：LRU 或 LFU，当空间不足时淘汰最冷数据
    """

    def __init__(self, l1_dir: str, l2_dir: str = None,
                 l1_max_gb: float = 500, l2_max_gb: float = 2000):
        self.l1_dir = Path(l1_dir)
        self.l2_dir = Path(l2_dir) if l2_dir else None
        self.l1_max_bytes = int(l1_max_gb * 1024**3)
        self.l2_max_bytes = int(l2_max_gb * 1024**3)

        self.l1_dir.mkdir(parents=True, exist_ok=True)
        if self.l2_dir:
            self.l2_dir.mkdir(parents=True, exist_ok=True)

        self.cache_index: Dict[str, CacheEntry] = {}
        self._load_index()

    def _cache_key(self, model_name: str, version: str,
                   fmt: str = "safetensors") -> str:
        return f"{model_name}/{version}/{fmt}"

    def _load_index(self):
        """加载缓存索引"""
        index_path = self.l1_dir / ".cache_index.json"
        if index_path.exists():
            with open(index_path) as f:
                data = json.load(f)
                for key, entry in data.items():
                    self.cache_index[key] = CacheEntry(**entry)

    def _save_index(self):
        """保存缓存索引"""
        index_path = self.l1_dir / ".cache_index.json"
        data = {k: asdict(v) for k, v in self.cache_index.items()}
        with open(index_path, "w") as f:
            json.dump(data, f, indent=2)

    def load_model(self, model_name: str, version: str,
                   fmt: str = "safetensors",
                   remote_path: str = None) -> Optional[str]:
        """加载模型（多级缓存查找）

        Returns:
            本地文件路径（已缓存到 L1）
        """
        cache_key = self._cache_key(model_name, version, fmt)
        print(f"\n加载模型: {cache_key}")

        # L1: 本地 SSD
        l1_path = self._check_l1(cache_key)
        if l1_path:
            print(f"  ✓ L1 命中 (本地 SSD): {l1_path}")
            self._update_access(cache_key)
            return l1_path

        # L2: 节点共享存储
        l2_path = self._check_l2(cache_key)
        if l2_path:
            print(f"  ✓ L2 命中 (共享存储): {l2_path}")
            # 拷贝到 L1
            l1_path = self._promote_to_l1(cache_key, l2_path)
            return l1_path

        # L3: 远端存储
        if remote_path:
            print(f"  ○ L1/L2 未命中，从远端下载: {remote_path}")
            l1_path = self._download_to_l1(cache_key, remote_path)
            return l1_path

        print(f"  ✗ 未找到模型")
        return None

    def _check_l1(self, cache_key: str) -> Optional[str]:
        """检查 L1 缓存"""
        if cache_key in self.cache_index:
            entry = self.cache_index[cache_key]
            if os.path.exists(entry.local_path):
                return entry.local_path
        # 检查文件是否存在
        l1_path = self.l1_dir / cache_key
        if l1_path.exists():
            return str(l1_path)
        return None

    def _check_l2(self, cache_key: str) -> Optional[str]:
        """检查 L2 缓存"""
        if self.l2_dir is None:
            return None
        l2_path = self.l2_dir / cache_key
        if l2_path.exists():
            return str(l2_path)
        return None

    def _promote_to_l1(self, cache_key: str, source_path: str) -> str:
        """将数据提升到 L1"""
        dest_path = self.l1_dir / cache_key
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # 检查 L1 空间
        self._ensure_l1_space(os.path.getsize(source_path))

        t0 = time.perf_counter()
        shutil.copy2(source_path, dest_path)
        copy_time = time.perf_counter() - t0

        size_mb = os.path.getsize(source_path) / 1024 / 1024
        print(f"  → 提升到 L1: {size_mb:.1f}MB in {copy_time:.2f}s "
              f"({size_mb/copy_time:.1f} MB/s)")

        self._register_cache(cache_key, str(dest_path))
        return str(dest_path)

    def _download_to_l1(self, cache_key: str, remote_path: str) -> str:
        """从远端下载到 L1"""
        dest_path = self.l1_dir / cache_key
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # 模拟下载（实际环境用 boto3 / s3cmd）
        t0 = time.perf_counter()

        if os.path.exists(remote_path):
            # 本地模拟
            self._ensure_l1_space(os.path.getsize(remote_path))
            shutil.copy2(remote_path, dest_path)
        else:
            print(f"  [模拟] 从 {remote_path} 下载...")
            # 创建空文件作为占位
            dest_path.touch()

        download_time = time.perf_counter() - t0

        if dest_path.exists() and dest_path.stat().st_size > 0:
            size_mb = dest_path.stat().st_size / 1024 / 1024
            print(f"  → 下载到 L1: {size_mb:.1f}MB in {download_time:.2f}s")

        self._register_cache(cache_key, str(dest_path))
        return str(dest_path)

    def _register_cache(self, cache_key: str, local_path: str):
        """注册缓存条目"""
        size = os.path.getsize(local_path) if os.path.exists(local_path) else 0
        self.cache_index[cache_key] = CacheEntry(
            model_name=cache_key.split("/")[0],
            version=cache_key.split("/")[1] if "/" in cache_key else "v1",
            format=cache_key.split("/")[2] if cache_key.count("/") >= 2 else "bin",
            local_path=local_path,
            size_bytes=size,
            last_access=time.time(),
            access_count=1,
        )
        self._save_index()

    def _update_access(self, cache_key: str):
        """更新访问时间"""
        if cache_key in self.cache_index:
            self.cache_index[cache_key].last_access = time.time()
            self.cache_index[cache_key].access_count += 1
            self._save_index()

    def _ensure_l1_space(self, needed_bytes: int):
        """确保 L1 有足够空间（LRU 淘汰）"""
        current_usage = sum(
            e.size_bytes for e in self.cache_index.values()
            if e.local_path.startswith(str(self.l1_dir))
        )

        while current_usage + needed_bytes > self.l1_max_bytes:
            # 淘汰最久未访问的
            oldest_key = min(
                self.cache_index,
                key=lambda k: self.cache_index[k].last_access,
            )
            entry = self.cache_index[oldest_key]
            if os.path.exists(entry.local_path):
                os.remove(entry.local_path)
                current_usage -= entry.size_bytes
                print(f"  [LRU] 淘汰: {oldest_key} ({entry.size_bytes/1024/1024:.1f}MB)")
            del self.cache_index[oldest_key]

    def status(self):
        """打印缓存状态"""
        print(f"\n{'='*50}")
        print("缓存状态:")
        total_size = sum(e.size_bytes for e in self.cache_index.values())
        print(f"  L1 目录: {self.l1_dir}")
        print(f"  缓存条目: {len(self.cache_index)}")
        print(f"  总大小: {total_size/1024/1024:.1f}MB / "
              f"{self.l1_max_bytes/1024**3:.0f}GB")

        for key, entry in sorted(self.cache_index.items(),
                                  key=lambda x: x[1].last_access, reverse=True):
            age = time.time() - entry.last_access
            print(f"  - {key}: {entry.size_bytes/1024/1024:.1f}MB, "
                  f"访问{entry.access_count}次, {age:.0f}s ago")


def main():
    parser = argparse.ArgumentParser(description="多级缓存模型加载器")
    parser.add_argument("--cache-dir", type=str, default="/tmp/model-cache/l1")
    parser.add_argument("--l2-dir", type=str, default="/tmp/model-cache/l2")
    parser.add_argument("--model-name", type=str, default="llama-7b")
    parser.add_argument("--version", type=str, default="v1.0")
    args = parser.parse_args()

    cache = MultiLevelCache(
        l1_dir=args.cache_dir,
        l2_dir=args.l2_dir,
        l1_max_gb=10,  # 演示用小值
    )

    # 演示加载流程
    result = cache.load_model(
        model_name=args.model_name,
        version=args.version,
        remote_path=f"/tmp/remote/{args.model_name}/{args.version}/model.bin",
    )

    # 第二次加载（应命中 L1）
    result = cache.load_model(
        model_name=args.model_name,
        version=args.version,
    )

    cache.status()


if __name__ == "__main__":
    main()
