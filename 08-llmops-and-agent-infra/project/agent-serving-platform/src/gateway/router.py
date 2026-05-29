"""请求路由 - 根据请求特征路由到合适的 Agent"""
import hashlib
from typing import Optional
from dataclasses import dataclass


@dataclass
class RouteDecision:
    agent_id: str
    version: str
    reason: str
    is_canary: bool = False


class RequestRouter:
    """
    请求路由器
    支持：规则路由、意图路由、灰度路由
    """

    def __init__(self, agent_registry, canary_config: dict = None):
        self.agent_registry = agent_registry
        self.canary_config = canary_config or {}

    def route(self, request: dict) -> RouteDecision:
        """路由请求到合适的 Agent"""
        # 1. 显式指定 Agent
        if "agent_id" in request:
            return RouteDecision(
                agent_id=request["agent_id"],
                version=self._get_version(request),
                reason="explicit",
            )

        # 2. 意图分类路由
        intent = self._classify_intent(request.get("query", ""))
        agent_id = self._intent_to_agent(intent)

        # 3. 灰度路由
        version = self._canary_route(
            agent_id, request.get("user_id", "anonymous")
        )

        return RouteDecision(
            agent_id=agent_id,
            version=version,
            reason=f"intent:{intent}",
            is_canary=version != "stable",
        )

    def _classify_intent(self, query: str) -> str:
        """意图分类（简化版 — 生产中应使用模型分类）"""
        query_lower = query.lower()
        if any(kw in query_lower for kw in ["代码", "编程", "实现", "code"]):
            return "coding"
        elif any(kw in query_lower for kw in ["分析", "报告", "数据"]):
            return "analysis"
        elif any(kw in query_lower for kw in ["搜索", "查找", "文档"]):
            return "knowledge_qa"
        return "general"

    def _intent_to_agent(self, intent: str) -> str:
        """意图映射到 Agent"""
        mapping = {
            "coding": "code_agent",
            "analysis": "analysis_agent",
            "knowledge_qa": "rag_agent",
            "general": "general_agent",
        }
        return mapping.get(intent, "general_agent")

    def _canary_route(self, agent_id: str, user_id: str) -> str:
        """灰度路由"""
        canary = self.canary_config.get(agent_id)
        if not canary:
            return "stable"

        # 确定性哈希分流
        hash_val = int(hashlib.md5(
            f"{agent_id}:{user_id}".encode()
        ).hexdigest()[:8], 16) / 0xFFFFFFFF

        if hash_val < canary.get("traffic_ratio", 0):
            return canary.get("canary_version", "canary")
        return "stable"

    def _get_version(self, request: dict) -> str:
        return request.get("version", "stable")
