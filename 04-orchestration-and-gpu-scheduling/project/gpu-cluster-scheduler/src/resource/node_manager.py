"""
节点状态管理

维护集群中所有节点的状态信息，包括健康状态、资源使用、标签和 Taint。
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class NodeStatus:
    """节点状态"""
    name: str
    ready: bool = True
    gpu_healthy: bool = True
    last_heartbeat: Optional[datetime] = None
    conditions: dict[str, str] = field(default_factory=dict)
    # 例如: {"MemoryPressure": "False", "DiskPressure": "False"}


class NodeManager:
    """节点管理器"""

    def __init__(self, health_check_timeout_sec: int = 60):
        self._nodes: dict[str, NodeStatus] = {}
        self._health_timeout = health_check_timeout_sec

    def register_node(self, name: str, initial_status: Optional[NodeStatus] = None):
        """注册节点"""
        self._nodes[name] = initial_status or NodeStatus(name=name)
        logger.info(f"节点注册: {name}")

    def update_heartbeat(self, name: str):
        """更新节点心跳"""
        if name in self._nodes:
            self._nodes[name].last_heartbeat = datetime.now()

    def mark_node_unhealthy(self, name: str, reason: str):
        """标记节点不健康"""
        if name in self._nodes:
            self._nodes[name].ready = False
            self._nodes[name].conditions["GPUHealth"] = reason
            logger.warning(f"节点 {name} 标记为不健康: {reason}")

    def mark_node_healthy(self, name: str):
        """标记节点恢复健康"""
        if name in self._nodes:
            self._nodes[name].ready = True
            self._nodes[name].gpu_healthy = True
            self._nodes[name].conditions.pop("GPUHealth", None)
            logger.info(f"节点 {name} 恢复健康")

    def get_healthy_nodes(self) -> list[str]:
        """获取所有健康节点"""
        now = datetime.now()
        healthy = []
        for name, status in self._nodes.items():
            if not status.ready:
                continue
            # 检查心跳超时
            if status.last_heartbeat:
                elapsed = (now - status.last_heartbeat).total_seconds()
                if elapsed > self._health_timeout:
                    logger.warning(f"节点 {name} 心跳超时 ({elapsed:.0f}s)")
                    continue
            healthy.append(name)
        return healthy

    def get_node_status(self, name: str) -> Optional[NodeStatus]:
        """获取节点状态"""
        return self._nodes.get(name)

    def get_all_status(self) -> dict[str, NodeStatus]:
        """获取所有节点状态"""
        return dict(self._nodes)

    def check_stale_nodes(self) -> list[str]:
        """检查心跳超时的节点"""
        stale = []
        now = datetime.now()
        for name, status in self._nodes.items():
            if status.last_heartbeat:
                elapsed = (now - status.last_heartbeat).total_seconds()
                if elapsed > self._health_timeout:
                    stale.append(name)
        return stale
