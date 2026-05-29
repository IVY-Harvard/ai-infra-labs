"""
拓扑感知通信
=============
根据 GPU 互联拓扑（NVLink / PCIe / Network）选择最优通信策略。
"""

import subprocess
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import torch.distributed as dist


@dataclass
class GPULink:
    """描述两个 GPU 之间的链接"""
    src: int
    dst: int
    link_type: str  # "nvlink", "pcie", "network"
    bandwidth_gbps: float


class TopologyDetector:
    """
    检测 GPU 拓扑并提供通信策略建议。
    """

    def __init__(self):
        self.num_gpus = torch.cuda.device_count()
        self.links: List[GPULink] = []
        self._detect()

    def _detect(self):
        """通过 P2P 访问能力推断拓扑"""
        for i in range(self.num_gpus):
            for j in range(i + 1, self.num_gpus):
                can_p2p = torch.cuda.can_device_access_peer(i, j)
                if can_p2p:
                    self.links.append(GPULink(i, j, "nvlink", 450.0))
                    self.links.append(GPULink(j, i, "nvlink", 450.0))
                else:
                    self.links.append(GPULink(i, j, "pcie", 64.0))
                    self.links.append(GPULink(j, i, "pcie", 64.0))

    def get_nvlink_groups(self) -> List[List[int]]:
        """获取 NVLink 互联的 GPU 组"""
        # 使用 Union-Find 算法分组
        parent = list(range(self.num_gpus))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for link in self.links:
            if link.link_type == "nvlink":
                union(link.src, link.dst)

        groups: Dict[int, List[int]] = {}
        for i in range(self.num_gpus):
            root = find(i)
            groups.setdefault(root, []).append(i)

        return list(groups.values())

    def recommend_tp_size(self) -> int:
        """推荐 TP 大小（NVLink 组内的 GPU 数量）"""
        nvlink_groups = self.get_nvlink_groups()
        if nvlink_groups:
            return len(nvlink_groups[0])
        return 1

    def get_bandwidth(self, src: int, dst: int) -> float:
        """获取两个 GPU 间的带宽"""
        for link in self.links:
            if link.src == src and link.dst == dst:
                return link.bandwidth_gbps
        return 0.0


class TopologyAwareGroupCreator:
    """
    拓扑感知的通信组创建器。
    确保 TP 通信只在 NVLink 组内发生。
    """

    def __init__(self, topology: TopologyDetector):
        self.topology = topology

    def create_optimal_groups(self, tp_size: int, pp_size: int, dp_size: int):
        """
        创建拓扑最优的通信组。
        规则: TP 放 NVLink 内，PP 跨 NVLink 组。
        """
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        assert tp_size * pp_size * dp_size == world_size

        nvlink_groups = self.topology.get_nvlink_groups()
        nvlink_size = len(nvlink_groups[0]) if nvlink_groups else world_size

        if tp_size > nvlink_size:
            if rank == 0:
                print(f"  警告: TP size ({tp_size}) > NVLink group size ({nvlink_size})")
                print(f"  TP 通信将走 PCIe，性能可能下降")

        # 标准映射: [DP, PP, TP]
        # TP 放在连续 rank (NVLink 内)
        tp_rank = rank % tp_size
        pp_rank = (rank // tp_size) % pp_size
        dp_rank = rank // (tp_size * pp_size)

        return {
            "tp_rank": tp_rank,
            "pp_rank": pp_rank,
            "dp_rank": dp_rank,
            "nvlink_group_size": nvlink_size,
        }
