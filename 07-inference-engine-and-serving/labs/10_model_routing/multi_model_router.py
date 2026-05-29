"""
多模型路由器

根据请求特征将请求路由到不同的模型实例。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import random
import time


@dataclass
class ModelEndpoint:
    """模型端点"""
    name: str
    url: str
    model_size: str  # "7b", "13b", "70b"
    capabilities: List[str]  # ["chat", "code", "reasoning"]
    max_context: int
    current_load: int = 0
    max_load: int = 100
    is_healthy: bool = True


class ModelRouter:
    """多模型路由器"""

    def __init__(self):
        self.endpoints: Dict[str, ModelEndpoint] = {}
        self.routing_rules = []

    def register_endpoint(self, endpoint: ModelEndpoint):
        self.endpoints[endpoint.name] = endpoint

    def add_rule(self, condition: callable, target: str, priority: int = 0):
        """添加路由规则"""
        self.routing_rules.append((priority, condition, target))
        self.routing_rules.sort(key=lambda x: -x[0])  # 高优先级在前

    def route(self, request: dict) -> Optional[ModelEndpoint]:
        """路由请求到合适的模型"""
        # 规则匹配
        for priority, condition, target in self.routing_rules:
            if condition(request):
                ep = self.endpoints.get(target)
                if ep and ep.is_healthy and ep.current_load < ep.max_load:
                    return ep

        # 默认: 选负载最低的
        available = [ep for ep in self.endpoints.values()
                    if ep.is_healthy and ep.current_load < ep.max_load]
        if available:
            return min(available, key=lambda x: x.current_load / x.max_load)
        return None


def demo_routing():
    """演示多模型路由"""
    print("=" * 70)
    print("  Multi-Model Router Demo")
    print("=" * 70)

    router = ModelRouter()

    # 注册模型
    router.register_endpoint(ModelEndpoint(
        name="llama-7b", url="http://gpu0:8000",
        model_size="7b", capabilities=["chat"],
        max_context=4096, max_load=200
    ))
    router.register_endpoint(ModelEndpoint(
        name="llama-70b", url="http://gpu1:8000",
        model_size="70b", capabilities=["chat", "reasoning"],
        max_context=8192, max_load=50
    ))
    router.register_endpoint(ModelEndpoint(
        name="codellama-34b", url="http://gpu2:8000",
        model_size="34b", capabilities=["code"],
        max_context=16384, max_load=80
    ))

    # 路由规则
    router.add_rule(
        lambda r: "code" in r.get("content", "").lower(),
        "codellama-34b", priority=10
    )
    router.add_rule(
        lambda r: r.get("require_reasoning", False),
        "llama-70b", priority=5
    )
    router.add_rule(
        lambda r: len(r.get("content", "")) < 100,
        "llama-7b", priority=1  # 短请求用小模型
    )

    # 测试路由
    requests = [
        {"content": "Hello!", "require_reasoning": False},
        {"content": "Write a Python function to sort a list", "require_reasoning": False},
        {"content": "Explain quantum computing in detail", "require_reasoning": True},
        {"content": "Hi", "require_reasoning": False},
    ]

    print(f"\n  {'Request':<50} {'Routed To'}")
    print(f"  {'-'*70}")
    for req in requests:
        ep = router.route(req)
        content = req['content'][:45]
        print(f"  {content:<50} {ep.name if ep else 'NONE'}")


if __name__ == "__main__":
    demo_routing()
