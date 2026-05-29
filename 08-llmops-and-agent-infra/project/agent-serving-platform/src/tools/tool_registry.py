"""工具注册中心 - 管理所有可用工具"""
from dataclasses import dataclass, field
from typing import Callable, Optional
from datetime import datetime


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict
    handler: Optional[Callable] = None
    source: str = "local"  # local / mcp
    version: str = "1.0.0"
    enabled: bool = True
    tags: list[str] = field(default_factory=list)
    registered_at: str = field(default_factory=lambda: datetime.now().isoformat())


class ToolRegistry:
    """工具注册中心"""

    def __init__(self):
        self.tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition):
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolDefinition]:
        tool = self.tools.get(name)
        if tool and tool.enabled:
            return tool
        return None

    def list_tools(self, tags: list[str] = None) -> list[ToolDefinition]:
        tools = [t for t in self.tools.values() if t.enabled]
        if tags:
            tools = [t for t in tools if any(tag in t.tags for tag in tags)]
        return tools

    def to_openai_functions(self, tool_names: list[str] = None) -> list[dict]:
        """导出为 OpenAI Function Calling 格式"""
        tools = self.tools.values() if not tool_names else [
            self.tools[n] for n in tool_names if n in self.tools
        ]
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
            }
            for t in tools if t.enabled
        ]

    async def execute(self, tool_name: str, arguments: dict) -> str:
        """执行工具"""
        tool = self.get(tool_name)
        if not tool:
            return f"Tool not found: {tool_name}"
        if not tool.handler:
            return f"Tool has no handler: {tool_name}"

        try:
            result = tool.handler(**arguments)
            if hasattr(result, '__await__'):
                result = await result
            return str(result)
        except Exception as e:
            return f"Tool execution error: {e}"
