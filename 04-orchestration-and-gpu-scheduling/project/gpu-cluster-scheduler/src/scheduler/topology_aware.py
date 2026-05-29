"""
GPU 拓扑感知调度模块

根据节点的 GPU NVLink/NVSwitch 拓扑信息，选择最优的 GPU 组合。
目标：让请求多 GPU 的 Job 获得 NVLink 互连的 GPU，最大化通信带宽。
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GPUTopologyInfo:
    """节点 GPU 拓扑信息"""
    node_name: str
    gpu_count: int
    nvlink_groups: list[list[int]]    # NVLink 互连组
    pcie_groups: list[list[int]]      # PCIe Switch 分组
    numa_mapping: dict[int, list[int]]  # NUMA node → GPU indices
    allocated_gpus: set[int] = field(default_factory=set)

    def get_available(self) -> list[int]:
        return [i for i in range(self.gpu_count) if i not in self.allocated_gpus]


class TopologyScorer:
    """GPU 拓扑感知评分器"""

    # 拓扑质量分数
    SCORE_SAME_NVLINK_GROUP = 100    # 全部在同一 NVLink 组
    SCORE_SAME_NUMA = 80             # 全部在同一 NUMA 节点
    SCORE_SAME_PCIE = 60             # 同一 PCIe Switch
    SCORE_CROSS_NUMA = 30            # 跨 NUMA（最差）
    SCORE_INSUFFICIENT = 0           # GPU 不足

    def __init__(self):
        self._topologies: dict[str, GPUTopologyInfo] = {}

    def register_node(self, topo: GPUTopologyInfo):
        """注册节点拓扑信息"""
        self._topologies[topo.node_name] = topo

    def update_allocation(self, node_name: str, allocated: set[int]):
        """更新 GPU 分配状态"""
        if node_name in self._topologies:
            self._topologies[node_name].allocated_gpus = allocated

    def score(self, node_name: str, gpu_count: int) -> float:
        """
        为节点的 GPU 拓扑打分。

        Returns:
            0-100 分，越高表示 GPU 拓扑越适合该请求
        """
        topo = self._topologies.get(node_name)
        if topo is None:
            return 50.0  # 没有拓扑信息，给中等分

        available = topo.get_available()
        if len(available) < gpu_count:
            return self.SCORE_INSUFFICIENT

        # 尝试各级拓扑
        # 优先级 1: 同一 NVLink 组
        for group in topo.nvlink_groups:
            avail_in_group = [g for g in group if g in available]
            if len(avail_in_group) >= gpu_count:
                return self.SCORE_SAME_NVLINK_GROUP

        # 优先级 2: 同一 NUMA 节点
        for numa_id, gpus in topo.numa_mapping.items():
            avail_in_numa = [g for g in gpus if g in available]
            if len(avail_in_numa) >= gpu_count:
                return self.SCORE_SAME_NUMA

        # 优先级 3: 同一 PCIe Switch
        for pcie_group in topo.pcie_groups:
            avail_in_pcie = [g for g in pcie_group if g in available]
            if len(avail_in_pcie) >= gpu_count:
                return self.SCORE_SAME_PCIE

        # 优先级 4: 跨 NUMA（有足够的空闲 GPU）
        if len(available) >= gpu_count:
            return self.SCORE_CROSS_NUMA

        return self.SCORE_INSUFFICIENT

    def select_gpus(self, node_name: str, gpu_count: int) -> list[int]:
        """
        为 Job 选择具体的 GPU indices。

        按拓扑优先级选择最佳组合：
        1. 尝试在同一 NVLink 组内找连续的 GPU
        2. 退而求其次在同一 NUMA 内找
        3. 最后跨 NUMA 分配
        """
        topo = self._topologies.get(node_name)
        if topo is None:
            # 没有拓扑信息，返回前 N 个可用 GPU
            return list(range(gpu_count))

        available = set(topo.get_available())
        if len(available) < gpu_count:
            return []

        # 策略 1: 同一 NVLink 组
        for group in topo.nvlink_groups:
            candidates = [g for g in group if g in available]
            if len(candidates) >= gpu_count:
                selected = candidates[:gpu_count]
                logger.info(f"  拓扑选择: NVLink 组内 GPU={selected}")
                return selected

        # 策略 2: 同一 NUMA 节点
        for numa_id, gpus in topo.numa_mapping.items():
            candidates = [g for g in gpus if g in available]
            if len(candidates) >= gpu_count:
                selected = candidates[:gpu_count]
                logger.info(f"  拓扑选择: NUMA-{numa_id} 内 GPU={selected}")
                return selected

        # 策略 3: 同一 PCIe Switch（尽量少跨 Switch）
        selected = []
        for pcie_group in topo.pcie_groups:
            candidates = [g for g in pcie_group if g in available and g not in selected]
            selected.extend(candidates)
            if len(selected) >= gpu_count:
                selected = selected[:gpu_count]
                logger.info(f"  拓扑选择: PCIe 组合 GPU={selected}")
                return selected

        # 策略 4: 任意可用 GPU
        selected = sorted(available)[:gpu_count]
        logger.info(f"  拓扑选择: 跨 NUMA GPU={selected}")
        return selected


def build_h20_topology(node_name: str, allocated: set[int] = None) -> GPUTopologyInfo:
    """
    构建 H20 8-GPU 标准拓扑。

    H20 拓扑：
      - GPU 0-3: NVLink 互连（NVSwitch 域 A）, NUMA 0
      - GPU 4-7: NVLink 互连（NVSwitch 域 B）, NUMA 1
      - 两组之间通过 PCIe
    """
    return GPUTopologyInfo(
        node_name=node_name,
        gpu_count=8,
        nvlink_groups=[[0, 1, 2, 3], [4, 5, 6, 7]],
        pcie_groups=[[0, 1], [2, 3], [4, 5], [6, 7]],
        numa_mapping={0: [0, 1, 2, 3], 1: [4, 5, 6, 7]},
        allocated_gpus=allocated or set(),
    )
