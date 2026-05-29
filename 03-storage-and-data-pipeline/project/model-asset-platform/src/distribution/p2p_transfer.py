"""P2P 传输模块"""

import hashlib
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field


@dataclass
class Chunk:
    index: int
    size: int
    checksum: str
    data: Optional[bytes] = None


@dataclass
class PeerNode:
    node_id: str
    address: str
    available_chunks: Set[int] = field(default_factory=set)


class P2PTransfer:
    """P2P 传输 — 将模型文件分块后在节点间互传"""

    def __init__(self, chunk_size_mb: int = 64):
        self.chunk_size = chunk_size_mb * 1024 * 1024
        self.peers: Dict[str, PeerNode] = {}
        self.chunks: List[Chunk] = []

    def split_file(self, data: bytes) -> List[Chunk]:
        """将文件数据切分为块"""
        self.chunks = []
        offset = 0
        index = 0
        while offset < len(data):
            end = min(offset + self.chunk_size, len(data))
            chunk_data = data[offset:end]
            self.chunks.append(Chunk(
                index=index,
                size=len(chunk_data),
                checksum=hashlib.md5(chunk_data).hexdigest(),
                data=chunk_data,
            ))
            offset = end
            index += 1
        return self.chunks

    def register_peer(self, node_id: str, address: str,
                      available_chunks: Set[int] = None):
        """注册对等节点"""
        self.peers[node_id] = PeerNode(
            node_id=node_id,
            address=address,
            available_chunks=available_chunks or set(),
        )

    def get_chunk(self, chunk_index: int,
                  requester: str) -> Optional[bytes]:
        """获取指定块（模拟 P2P 传输）"""
        if 0 <= chunk_index < len(self.chunks):
            return self.chunks[chunk_index].data
        return None

    def assemble(self, chunks: Dict[int, bytes]) -> bytes:
        """将块组装为完整文件"""
        result = b""
        for i in sorted(chunks.keys()):
            result += chunks[i]
        return result
