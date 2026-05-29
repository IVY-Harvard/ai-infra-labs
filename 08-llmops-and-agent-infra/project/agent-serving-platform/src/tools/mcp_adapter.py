"""MCP 适配器 - 将 MCP Server 的工具集成到平台"""
from dataclasses import dataclass
from typing import Optional
from .tool_registry import ToolRegistry, ToolDefinition


@dataclass
class MCPServerConfig:
    name: str
    transport: str  # "stdio" or "sse"
    command: str = ""
    args: list = None
    url: str = ""


class MCPAdapter:
    """
    MCP 适配器
    连接外部 MCP Server，将其工具注册到平台的 ToolRegistry
    """

    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self.servers: dict[str, MCPServerConfig] = {}
        self.sessions = {}

    async def connect_server(self, config: MCPServerConfig):
        """连接 MCP Server 并注册其工具"""
        self.servers[config.name] = config

        # 实际实现中使用 MCP SDK 连接
        # 这里模拟注册 MCP 工具
        mock_tools = self._get_mock_tools(config.name)
        for tool_def in mock_tools:
            self.tool_registry.register(tool_def)

        print(f"  MCP Server '{config.name}' 连接成功，注册了 {len(mock_tools)} 个工具")

    async def disconnect_server(self, name: str):
        """断开 MCP Server"""
        if name in self.servers:
            # 移除该 Server 的工具
            to_remove = [
                t.name for t in self.tool_registry.list_tools()
                if t.source == f"mcp:{name}"
            ]
            for tool_name in to_remove:
                if tool_name in self.tool_registry.tools:
                    self.tool_registry.tools[tool_name].enabled = False
            del self.servers[name]

    def _get_mock_tools(self, server_name: str) -> list[ToolDefinition]:
        """模拟 MCP Server 的工具（实际会通过 MCP 协议发现）"""
        return [
            ToolDefinition(
                name=f"{server_name}_search",
                description=f"通过 {server_name} 搜索信息",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                source=f"mcp:{server_name}",
                tags=["mcp", server_name],
            ),
        ]
