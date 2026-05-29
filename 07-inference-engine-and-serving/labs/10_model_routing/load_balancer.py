"""负载均衡器"""
from dataclasses import dataclass, field
from typing import List, Optional
import random
import time


@dataclass
class Backend:
    name: str
    url: str
    weight: int = 1
    current_connections: int = 0
    pending_requests: int = 0
    total_requests: int = 0
    total_latency: float = 0.0
    is_healthy: bool = True
    last_health_check: float = 0.0


class LoadBalancer:
    """LLM 推理负载均衡器"""

    def __init__(self, strategy: str = "least_pending"):
        self.backends: List[Backend] = []
        self.strategy = strategy

    def add_backend(self, backend: Backend):
        self.backends.append(backend)

    def select_backend(self) -> Optional[Backend]:
        healthy = [b for b in self.backends if b.is_healthy]
        if not healthy:
            return None

        if self.strategy == "round_robin":
            return healthy[random.randint(0, len(healthy) - 1)]
        elif self.strategy == "least_pending":
            return min(healthy, key=lambda b: b.pending_requests)
        elif self.strategy == "least_latency":
            return min(healthy, key=lambda b: b.total_latency / max(b.total_requests, 1))
        elif self.strategy == "weighted":
            total_weight = sum(b.weight for b in healthy)
            r = random.uniform(0, total_weight)
            for b in healthy:
                r -= b.weight
                if r <= 0:
                    return b
            return healthy[-1]
        return healthy[0]


def demo_load_balancer():
    print("=" * 70)
    print("  Load Balancer Demo")
    print("=" * 70)

    lb = LoadBalancer(strategy="least_pending")
    lb.add_backend(Backend("gpu-0", "http://gpu0:8000", pending_requests=5))
    lb.add_backend(Backend("gpu-1", "http://gpu1:8000", pending_requests=2))
    lb.add_backend(Backend("gpu-2", "http://gpu2:8000", pending_requests=8))

    print(f"\n  Strategy: least_pending")
    for i in range(10):
        b = lb.select_backend()
        print(f"  Request {i}: → {b.name} (pending={b.pending_requests})")
        b.pending_requests += 1


if __name__ == "__main__":
    demo_load_balancer()
