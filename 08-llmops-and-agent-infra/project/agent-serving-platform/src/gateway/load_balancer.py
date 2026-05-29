"""负载均衡器 - 多实例负载分配"""
import random
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional


@dataclass
class ServiceInstance:
    instance_id: str
    host: str
    port: int
    weight: int = 1
    healthy: bool = True
    active_connections: int = 0
    last_health_check: float = 0
    total_requests: int = 0
    total_errors: int = 0


class LoadBalancer:
    """
    负载均衡器
    支持：轮询、加权轮询、最少连接、一致性哈希
    """

    def __init__(self, strategy: str = "weighted_round_robin"):
        self.strategy = strategy
        self.instances: dict[str, list[ServiceInstance]] = defaultdict(list)
        self._rr_index: dict[str, int] = defaultdict(int)

    def register_instance(self, service_name: str, instance: ServiceInstance):
        """注册服务实例"""
        self.instances[service_name].append(instance)

    def select_instance(self, service_name: str,
                        key: str = None) -> Optional[ServiceInstance]:
        """选择一个健康的实例"""
        healthy = [i for i in self.instances.get(service_name, []) if i.healthy]
        if not healthy:
            return None

        if self.strategy == "round_robin":
            return self._round_robin(service_name, healthy)
        elif self.strategy == "weighted_round_robin":
            return self._weighted_round_robin(service_name, healthy)
        elif self.strategy == "least_connections":
            return self._least_connections(healthy)
        elif self.strategy == "consistent_hash" and key:
            return self._consistent_hash(healthy, key)
        return random.choice(healthy)

    def _round_robin(self, service_name, instances):
        idx = self._rr_index[service_name] % len(instances)
        self._rr_index[service_name] += 1
        return instances[idx]

    def _weighted_round_robin(self, service_name, instances):
        weighted = []
        for inst in instances:
            weighted.extend([inst] * inst.weight)
        if not weighted:
            return instances[0]
        idx = self._rr_index[service_name] % len(weighted)
        self._rr_index[service_name] += 1
        return weighted[idx]

    def _least_connections(self, instances):
        return min(instances, key=lambda i: i.active_connections)

    def _consistent_hash(self, instances, key):
        import hashlib
        hash_val = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
        idx = hash_val % len(instances)
        return instances[idx]

    def report_success(self, instance: ServiceInstance):
        instance.active_connections = max(0, instance.active_connections - 1)
        instance.total_requests += 1

    def report_failure(self, instance: ServiceInstance):
        instance.active_connections = max(0, instance.active_connections - 1)
        instance.total_requests += 1
        instance.total_errors += 1
        # 错误率过高自动摘除
        if instance.total_requests > 10:
            error_rate = instance.total_errors / instance.total_requests
            if error_rate > 0.5:
                instance.healthy = False

    def health_check(self, service_name: str):
        """健康检查（简化版）"""
        for instance in self.instances.get(service_name, []):
            instance.last_health_check = time.time()
            # 实际应 ping 实例
