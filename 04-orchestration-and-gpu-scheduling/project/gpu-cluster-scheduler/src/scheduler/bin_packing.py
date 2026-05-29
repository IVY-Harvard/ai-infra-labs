"""
GPU Bin Packing 调度策略

将 GPU 工作负载集中分配到尽量少的节点，减少碎片化。
"""

import logging

logger = logging.getLogger(__name__)


class BinPackingScorer:
    """Bin Packing 评分器"""

    # 常见 GPU 请求粒度
    COMMON_SIZES = [1, 2, 4, 8]

    def __init__(self, fragmentation_penalty: float = 0.3):
        self.fragmentation_penalty = fragmentation_penalty

    def score(self, allocated: int, total: int, requested: int) -> float:
        """
        为节点打 Bin Packing 分数。

        核心逻辑：分配后利用率越高分数越高（集中分配）。
        同时考虑碎片化惩罚。

        Args:
            allocated: 已分配的 GPU 数量
            total: 总 GPU 数量
            requested: 请求的 GPU 数量

        Returns:
            0-100 分
        """
        available = total - allocated
        if available < requested:
            return 0.0

        # 基础分：分配后利用率 × 100
        new_allocated = allocated + requested
        utilization = new_allocated / total
        base_score = utilization * 100

        # 碎片化惩罚
        remaining = total - new_allocated
        penalty = self._calc_fragmentation_penalty(remaining)

        return max(0.0, min(100.0, base_score - penalty))

    def _calc_fragmentation_penalty(self, remaining: int) -> float:
        """
        计算碎片化惩罚分。

        如果剩余 GPU 不是常见粒度的整数倍，
        说明碎片化严重，给予惩罚。
        """
        if remaining == 0:
            return 0.0

        # 检查是否能被常见粒度整除
        for size in self.COMMON_SIZES:
            if remaining % size == 0:
                return 0.0

        # 贪心计算浪费
        temp = remaining
        for size in sorted(self.COMMON_SIZES, reverse=True):
            while temp >= size:
                temp -= size

        waste_ratio = temp / remaining if remaining > 0 else 0
        return waste_ratio * self.fragmentation_penalty * 100

    def rank_nodes(
        self,
        nodes: list[dict],
        requested_gpus: int,
    ) -> list[tuple[str, float]]:
        """
        为多个节点打分并排序。

        Args:
            nodes: [{"name": str, "allocated": int, "total": int}]
            requested_gpus: 请求 GPU 数

        Returns:
            [(node_name, score)] 降序
        """
        scored = []
        for node in nodes:
            s = self.score(node["allocated"], node["total"], requested_gpus)
            scored.append((node["name"], s))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
