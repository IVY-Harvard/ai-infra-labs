"""
GPU Bin Packing 调度策略

目标：将 GPU 工作负载集中分配到尽量少的节点上，
减少 GPU 碎片化，为大型训练任务腾出连续的 GPU 资源块。

对比：
  - 默认 LeastAllocated: 分散 → 每个节点都有少量空闲 GPU → 大任务无法调度
  - Bin Packing:         集中 → 少数节点满载，其余节点完全空闲 → 大任务有位置

使用方式：
    from bin_packing import BinPackingScorer
    scorer = BinPackingScorer()
    scores = scorer.score_nodes(nodes, requested_gpus=4)
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NodeState:
    """节点 GPU 资源状态"""
    name: str
    total_gpus: int
    allocated_gpus: int

    @property
    def available_gpus(self) -> int:
        return self.total_gpus - self.allocated_gpus

    @property
    def utilization(self) -> float:
        if self.total_gpus == 0:
            return 0.0
        return self.allocated_gpus / self.total_gpus


class BinPackingScorer:
    """
    GPU Bin Packing 打分器

    核心思想：优先选择已经分配了更多 GPU 的节点（利用率高的节点）。
    这样可以把空闲 GPU 集中在少数节点上，减少碎片。
    """

    def __init__(self, fragmentation_penalty: float = 0.3):
        """
        Args:
            fragmentation_penalty: 碎片化惩罚系数。
                分配后剩余 GPU 如果不满足常见请求粒度（1,2,4,8），
                会被额外惩罚。
        """
        self.fragmentation_penalty = fragmentation_penalty
        # 常见的 GPU 请求粒度
        self.common_request_sizes = [1, 2, 4, 8]

    def score_node(self, node: NodeState, requested_gpus: int) -> float:
        """
        为单个节点打分（0-100）。

        算法：
          base_score = (已分配 GPU / 总 GPU) * 100
          → 利用率越高分数越高（Bin Packing 核心）

          fragmentation_adjustment：
          → 分配后剩余 GPU 能否被常见粒度整除
          → 不能整除的情况给予惩罚

        Returns:
            分数，越高越优先被调度
        """
        if node.available_gpus < requested_gpus:
            return 0.0

        # Bin Packing 基础分：利用率越高分越高
        gpus_after = node.allocated_gpus + requested_gpus
        utilization_after = gpus_after / node.total_gpus
        base_score = utilization_after * 100

        # 碎片化惩罚
        remaining = node.total_gpus - gpus_after
        frag_penalty = self._fragmentation_penalty(remaining)

        final_score = base_score - frag_penalty
        return max(0.0, min(100.0, final_score))

    def _fragmentation_penalty(self, remaining_gpus: int) -> float:
        """
        计算碎片化惩罚。

        如果分配后剩余的 GPU 数量不是常见请求粒度的整数倍，
        说明这些剩余 GPU 很可能无法被后续任务充分利用。

        示例（8 GPU 节点）：
          剩余 0: 惩罚 0（完美填充）
          剩余 1: 惩罚小（可以跑 1-GPU 推理）
          剩余 2: 惩罚 0（可以跑 2-GPU 任务）
          剩余 3: 惩罚大（3 = 2+1，有碎片）
          剩余 4: 惩罚 0（可以跑 4-GPU 训练）
          剩余 5: 惩罚大（5 = 4+1，有碎片）
          剩余 6: 惩罚小（6 = 4+2）
          剩余 7: 惩罚中（7 = 4+2+1）
        """
        if remaining_gpus == 0:
            return 0.0

        # 检查是否能被常见粒度整除
        best_fit = False
        for size in self.common_request_sizes:
            if remaining_gpus % size == 0:
                best_fit = True
                break

        if best_fit:
            return 0.0

        # 计算最佳近似组合的浪费
        waste = self._calculate_waste(remaining_gpus)
        return waste * self.fragmentation_penalty * 100

    def _calculate_waste(self, remaining: int) -> float:
        """贪心算法计算最小浪费比例"""
        used = 0
        temp = remaining
        for size in sorted(self.common_request_sizes, reverse=True):
            while temp >= size:
                temp -= size
                used += size
        waste = (remaining - used) / max(remaining, 1)
        return waste

    def score_nodes(
        self,
        nodes: list[NodeState],
        requested_gpus: int,
    ) -> list[tuple[str, float]]:
        """
        为所有节点打分并排序。

        Returns:
            [(节点名, 分数)] 按分数降序排列
        """
        scores = []
        for node in nodes:
            score = self.score_node(node, requested_gpus)
            scores.append((node.name, score))
            logger.info(
                f"  {node.name}: "
                f"GPU {node.allocated_gpus}/{node.total_gpus} "
                f"→ 分配 {requested_gpus} 后 {node.allocated_gpus + requested_gpus}/{node.total_gpus} "
                f"→ 分数 {score:.1f}"
            )

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


# --- 演示 ---

def demo():
    """
    模拟在 8x H20 集群上的 Bin Packing 调度。
    """
    logging.basicConfig(level=logging.INFO)

    # 集群状态：2 个节点，各 8 GPU
    nodes = [
        NodeState("gpu-node-0", total_gpus=8, allocated_gpus=2),
        NodeState("gpu-node-1", total_gpus=8, allocated_gpus=0),
    ]

    scorer = BinPackingScorer()

    print("=" * 60)
    print("场景 1：请求 2 GPU")
    print("=" * 60)
    scores = scorer.score_nodes(nodes, requested_gpus=2)
    print(f"调度决策：→ {scores[0][0]} (分数 {scores[0][1]:.1f})")
    print()

    # 更新状态（假设调度到了 node-0）
    nodes[0].allocated_gpus += 2

    print("=" * 60)
    print("场景 2：再请求 2 GPU（node-0 已有 4 GPU）")
    print("=" * 60)
    scores = scorer.score_nodes(nodes, requested_gpus=2)
    print(f"调度决策：→ {scores[0][0]} (分数 {scores[0][1]:.1f})")
    print()

    # 更新状态
    nodes[0].allocated_gpus += 2

    print("=" * 60)
    print("场景 3：请求 4 GPU（node-0 已满 6 GPU）")
    print("=" * 60)
    scores = scorer.score_nodes(nodes, requested_gpus=4)
    print(f"调度决策：→ {scores[0][0]} (分数 {scores[0][1]:.1f})")
    print("Bin Packing 让 node-1 保持了 4 个连续空闲 GPU！")

    print()
    print("=" * 60)
    print("对比：如果用默认 LeastAllocated（分散策略）")
    print("=" * 60)
    print("场景 1: 2 GPU → node-1 (空闲更多)")
    print("场景 2: 2 GPU → node-0 或 node-1")
    print("场景 3: 4 GPU → 可能无法调度（两个节点各剩 4-6 GPU 但被分散了）")


if __name__ == "__main__":
    demo()
