"""Agent 注册中心 - 管理所有 Agent 的注册和发现"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


class AgentStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    CANARY = "canary"
    DEPRECATED = "deprecated"


@dataclass
class AgentConfig:
    agent_id: str
    name: str
    description: str
    version: str
    model: str = "qwen2.5-72b"
    system_prompt: str = ""
    tools: list[str] = field(default_factory=list)
    max_steps: int = 10
    temperature: float = 0.3
    status: AgentStatus = AgentStatus.ACTIVE
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)


class AgentRegistry:
    """Agent 注册中心"""

    def __init__(self):
        self.agents: dict[str, AgentConfig] = {}
        self._version_history: dict[str, list[str]] = {}

    def register(self, config: AgentConfig) -> str:
        """注册 Agent"""
        key = f"{config.agent_id}:{config.version}"
        self.agents[key] = config

        if config.agent_id not in self._version_history:
            self._version_history[config.agent_id] = []
        self._version_history[config.agent_id].append(config.version)

        return key

    def get(self, agent_id: str, version: str = "latest") -> Optional[AgentConfig]:
        """获取 Agent 配置"""
        if version == "latest":
            versions = self._version_history.get(agent_id, [])
            if not versions:
                return None
            version = versions[-1]

        key = f"{agent_id}:{version}"
        return self.agents.get(key)

    def get_active(self, agent_id: str) -> Optional[AgentConfig]:
        """获取活跃版本"""
        for ver in reversed(self._version_history.get(agent_id, [])):
            config = self.agents.get(f"{agent_id}:{ver}")
            if config and config.status == AgentStatus.ACTIVE:
                return config
        return None

    def list_agents(self, status: AgentStatus = None) -> list[AgentConfig]:
        """列出所有 Agent"""
        agents = list(self.agents.values())
        if status:
            agents = [a for a in agents if a.status == status]
        return agents

    def deactivate(self, agent_id: str, version: str):
        """停用 Agent"""
        key = f"{agent_id}:{version}"
        if key in self.agents:
            self.agents[key].status = AgentStatus.INACTIVE
