"""
P2P 模型加载器

模拟 BitTorrent 风格的模型分发：
1. 将模型文件切分为固定大小的块（chunk）
2. 种子节点持有全部块
3. 请求节点优先从已有块的邻居节点拉取
4. 下载完成的块立即可供其他节点使用

本地演示版：使用多线程模拟多节点行为。

用法：
    python p2p_model_loader.py --model-path /path/to/model.bin --num-peers 4
"""

import os
import time
import hashlib
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional


@dataclass
class Chunk:
    """数据块"""
    index: int
    offset: int
    size: int
    checksum: str
    data: Optional[bytes] = None


@dataclass
class Peer:
    """模拟的对等节点"""
    peer_id: str
    available_chunks: Set[int] = field(default_factory=set)
    download_speed_mbps: float = 100.0  # 模拟带宽
    downloaded_bytes: int = 0
    upload_bytes: int = 0


class P2PModelLoader:
    """P2P 模型分发加载器

    分发策略：
    1. Rarest First: 优先下载最稀有的块（加速全网络可用性）
    2. Local First: 优先从最近的节点下载
    3. Parallel Download: 同时从多个节点下载不同块
    """

    def __init__(self, chunk_size_mb: int = 64):
        self.chunk_size = chunk_size_mb * 1024 * 1024
        self.chunks: List[Chunk] = []
        self.peers: Dict[str, Peer] = {}
        self.lock = threading.Lock()

    def prepare_model(self, model_path: str) -> List[Chunk]:
        """将模型文件切分为块"""
        file_size = os.path.getsize(model_path)
        num_chunks = (file_size + self.chunk_size - 1) // self.chunk_size

        print(f"模型文件: {model_path}")
        print(f"文件大小: {file_size / 1024 / 1024:.1f}MB")
        print(f"块大小: {self.chunk_size / 1024 / 1024:.0f}MB")
        print(f"总块数: {num_chunks}")

        self.chunks = []
        with open(model_path, "rb") as f:
            for i in range(num_chunks):
                offset = i * self.chunk_size
                f.seek(offset)
                data = f.read(self.chunk_size)
                checksum = hashlib.md5(data).hexdigest()

                chunk = Chunk(
                    index=i,
                    offset=offset,
                    size=len(data),
                    checksum=checksum,
                    data=data,
                )
                self.chunks.append(chunk)

        return self.chunks

    def register_seeder(self, peer_id: str = "seeder"):
        """注册种子节点（拥有全部块）"""
        peer = Peer(
            peer_id=peer_id,
            available_chunks=set(range(len(self.chunks))),
        )
        self.peers[peer_id] = peer
        print(f"种子节点注册: {peer_id} (拥有全部 {len(self.chunks)} 块)")

    def simulate_download(self, requester_id: str,
                          output_path: str,
                          strategy: str = "rarest_first") -> float:
        """模拟从 P2P 网络下载模型

        Args:
            requester_id: 请求节点 ID
            output_path: 输出路径
            strategy: 下载策略 (rarest_first / sequential)

        Returns:
            下载耗时(秒)
        """
        peer = Peer(peer_id=requester_id)
        self.peers[requester_id] = peer

        needed_chunks = set(range(len(self.chunks)))
        downloaded_chunks: Dict[int, bytes] = {}

        print(f"\n节点 {requester_id} 开始下载 ({len(needed_chunks)} 块)...")

        t_start = time.perf_counter()

        while needed_chunks:
            # 选择下一个要下载的块
            if strategy == "rarest_first":
                chunk_idx = self._select_rarest(needed_chunks)
            else:
                chunk_idx = min(needed_chunks)

            # 找到拥有该块的节点
            source_peer = self._find_source(chunk_idx, requester_id)
            if source_peer is None:
                print(f"  块 {chunk_idx} 无可用源！")
                break

            # 模拟传输
            chunk = self.chunks[chunk_idx]
            transfer_time = chunk.size / (source_peer.download_speed_mbps
                                          * 1024 * 1024)
            # 不实际 sleep，用累加模拟
            downloaded_chunks[chunk_idx] = chunk.data

            with self.lock:
                peer.available_chunks.add(chunk_idx)
                peer.downloaded_bytes += chunk.size
                source_peer.upload_bytes += chunk.size

            needed_chunks.remove(chunk_idx)

        download_time = time.perf_counter() - t_start

        # 写入文件
        with open(output_path, "wb") as f:
            for i in sorted(downloaded_chunks.keys()):
                f.write(downloaded_chunks[i])

        total_mb = peer.downloaded_bytes / 1024 / 1024
        print(f"  下载完成: {total_mb:.1f}MB in {download_time:.2f}s "
              f"({total_mb/download_time:.1f} MB/s)")

        return download_time

    def _select_rarest(self, needed: Set[int]) -> int:
        """选择最稀有的块（拥有该块的节点最少）"""
        chunk_counts = {}
        for idx in needed:
            count = sum(
                1 for p in self.peers.values()
                if idx in p.available_chunks
            )
            chunk_counts[idx] = count

        return min(chunk_counts, key=chunk_counts.get)

    def _find_source(self, chunk_idx: int,
                     requester_id: str) -> Optional[Peer]:
        """找到拥有指定块的节点"""
        for peer in self.peers.values():
            if peer.peer_id != requester_id and chunk_idx in peer.available_chunks:
                return peer
        return None

    def simulate_multi_peer_download(self, num_peers: int,
                                     output_dir: str) -> Dict:
        """模拟多个节点同时下载"""
        os.makedirs(output_dir, exist_ok=True)
        results = {}

        self.register_seeder()

        for i in range(num_peers):
            peer_id = f"node_{i+1}"
            output_path = os.path.join(output_dir, f"{peer_id}_model.bin")

            download_time = self.simulate_download(
                peer_id, output_path, strategy="rarest_first"
            )
            results[peer_id] = download_time

        # 打印对比
        print(f"\n{'='*50}")
        print("P2P 分发结果:")
        for peer_id, t in results.items():
            total_mb = self.peers[peer_id].downloaded_bytes / 1024 / 1024
            print(f"  {peer_id}: {t:.2f}s ({total_mb/t:.1f} MB/s)")

        return results


def create_test_model(path: str, size_mb: int = 256):
    """创建测试用的模型文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = os.urandom(64 * 1024 * 1024)
    remaining = size_mb * 1024 * 1024

    with open(path, "wb") as f:
        while remaining > 0:
            write_size = min(len(data), remaining)
            f.write(data[:write_size])
            remaining -= write_size

    print(f"创建测试模型: {path} ({size_mb}MB)")


def main():
    parser = argparse.ArgumentParser(description="P2P 模型加载器")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-size", type=int, default=256,
                       help="测试模型大小(MB)")
    parser.add_argument("--chunk-size", type=int, default=64,
                       help="块大小(MB)")
    parser.add_argument("--num-peers", type=int, default=4,
                       help="模拟节点数")
    parser.add_argument("--output-dir", type=str, default="/tmp/p2p_test")
    args = parser.parse_args()

    # 准备模型文件
    if args.model_path is None:
        args.model_path = os.path.join(args.output_dir, "source_model.bin")
        create_test_model(args.model_path, args.model_size)

    # P2P 分发
    loader = P2PModelLoader(chunk_size_mb=args.chunk_size)
    loader.prepare_model(args.model_path)
    loader.simulate_multi_peer_download(args.num_peers, args.output_dir)


if __name__ == "__main__":
    main()
