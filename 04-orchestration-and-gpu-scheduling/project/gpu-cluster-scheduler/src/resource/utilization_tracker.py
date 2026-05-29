"""
GPU 利用率追踪器

持续采集和记录 GPU 利用率指标，用于：
  1. 调度决策参考（把任务调度到利用率低的 GPU 上）
  2. 识别闲置 GPU（触发资源回收）
  3. 性能报告和容量规划
"""

import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GPUMetrics:
    """单个 GPU 的实时指标"""
    node_name: str
    gpu_index: int
    utilization_percent: float = 0.0   # SM 利用率
    memory_used_mb: int = 0
    memory_total_mb: int = 0
    temperature_celsius: float = 0.0
    power_watts: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def memory_utilization(self) -> float:
        if self.memory_total_mb == 0:
            return 0.0
        return self.memory_used_mb / self.memory_total_mb * 100


@dataclass
class NodeUtilizationSummary:
    """节点利用率摘要"""
    node_name: str
    avg_gpu_utilization: float
    avg_memory_utilization: float
    max_gpu_utilization: float
    min_gpu_utilization: float
    idle_gpu_count: int              # 利用率 < 5% 的 GPU 数量
    total_gpu_count: int
    collection_time: datetime = field(default_factory=datetime.now)


class UtilizationTracker:
    """GPU 利用率追踪器"""

    IDLE_THRESHOLD = 5.0   # 利用率 < 5% 视为闲置

    def __init__(self, history_minutes: int = 60):
        """
        Args:
            history_minutes: 保留多少分钟的历史数据
        """
        self.history_minutes = history_minutes
        # {(node_name, gpu_index): [GPUMetrics]}
        self._history: dict[tuple[str, int], list[GPUMetrics]] = defaultdict(list)

    def record(self, metrics: GPUMetrics):
        """记录一条 GPU 指标"""
        key = (metrics.node_name, metrics.gpu_index)
        self._history[key].append(metrics)
        self._trim_history(key)

    def record_batch(self, metrics_list: list[GPUMetrics]):
        """批量记录"""
        for m in metrics_list:
            self.record(m)

    def get_node_summary(self, node_name: str) -> Optional[NodeUtilizationSummary]:
        """获取节点利用率摘要（基于最近 5 分钟的数据）"""
        cutoff = datetime.now() - timedelta(minutes=5)
        gpu_utils = []
        mem_utils = []
        gpu_count = 0

        for (nname, gpu_idx), history in self._history.items():
            if nname != node_name:
                continue
            gpu_count += 1

            recent = [m for m in history if m.timestamp > cutoff]
            if recent:
                avg_util = sum(m.utilization_percent for m in recent) / len(recent)
                avg_mem = sum(m.memory_utilization for m in recent) / len(recent)
                gpu_utils.append(avg_util)
                mem_utils.append(avg_mem)

        if not gpu_utils:
            return None

        idle_count = sum(1 for u in gpu_utils if u < self.IDLE_THRESHOLD)

        return NodeUtilizationSummary(
            node_name=node_name,
            avg_gpu_utilization=sum(gpu_utils) / len(gpu_utils),
            avg_memory_utilization=sum(mem_utils) / len(mem_utils),
            max_gpu_utilization=max(gpu_utils),
            min_gpu_utilization=min(gpu_utils),
            idle_gpu_count=idle_count,
            total_gpu_count=gpu_count,
        )

    def get_idle_gpus(self) -> list[tuple[str, int]]:
        """获取所有闲置的 GPU"""
        cutoff = datetime.now() - timedelta(minutes=5)
        idle = []

        for (node_name, gpu_idx), history in self._history.items():
            recent = [m for m in history if m.timestamp > cutoff]
            if recent:
                avg_util = sum(m.utilization_percent for m in recent) / len(recent)
                if avg_util < self.IDLE_THRESHOLD:
                    idle.append((node_name, gpu_idx))

        return idle

    def get_cluster_utilization(self) -> dict:
        """获取集群整体利用率"""
        nodes = set(nname for nname, _ in self._history.keys())
        summaries = []
        for node in nodes:
            s = self.get_node_summary(node)
            if s:
                summaries.append(s)

        if not summaries:
            return {"avg_utilization": 0, "total_gpus": 0, "idle_gpus": 0}

        total_gpus = sum(s.total_gpu_count for s in summaries)
        idle_gpus = sum(s.idle_gpu_count for s in summaries)
        avg_util = sum(s.avg_gpu_utilization * s.total_gpu_count for s in summaries) / total_gpus

        return {
            "avg_utilization": round(avg_util, 1),
            "total_gpus": total_gpus,
            "idle_gpus": idle_gpus,
            "active_gpus": total_gpus - idle_gpus,
            "nodes": [
                {
                    "name": s.node_name,
                    "avg_util": round(s.avg_gpu_utilization, 1),
                    "idle": s.idle_gpu_count,
                    "total": s.total_gpu_count,
                }
                for s in summaries
            ],
        }

    def _trim_history(self, key: tuple[str, int]):
        """清理过期历史数据"""
        cutoff = datetime.now() - timedelta(minutes=self.history_minutes)
        self._history[key] = [
            m for m in self._history[key] if m.timestamp > cutoff
        ]
