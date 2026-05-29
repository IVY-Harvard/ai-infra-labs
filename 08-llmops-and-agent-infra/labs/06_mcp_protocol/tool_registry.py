"""
Lab 06: MCP Tool Registry - 工具注册中心
管理多个 MCP Server 的工具，提供统一的发现和调用接口
"""
import json
import asyncio
from typing import Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class ToolDefinition:
    """工具定义"""
    name: str
    description: str
    input_schema: dict
    server_name: str
    server_uri: str
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)
    registered_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ServerRegistration:
    """Server 注册信息"""
    name: str
    uri: str
    transport: str  # "stdio" or "sse"
    command: Optional[str] = None
    args: Optional[list[str]] = None
    tools: list[ToolDefinition] = field(default_factory=list)
    status: str = "registered"  # registered / connected / disconnected / error
    last_heartbeat: Optional[str] = None


class ToolRegistry:
    """
    MCP 工具注册中心
    管理多个 MCP Server 的工具，提供统一发现和路由
    类比微服务架构中的服务注册中心
    """

    def __init__(self):
        self.servers: dict[str, ServerRegistration] = {}
        self.tool_index: dict[str, ToolDefinition] = {}
        self.tag_index: dict[str, list[str]] = {}  # tag → [tool_name]

    def register_server(self, name: str, uri: str,
                        transport: str = "stdio",
                        command: str = None,
                        args: list[str] = None) -> ServerRegistration:
        """注册一个 MCP Server"""
        server = ServerRegistration(
            name=name, uri=uri, transport=transport,
            command=command, args=args,
        )
        self.servers[name] = server
        print(f"✓ 注册 Server: {name} ({transport}://{uri})")
        return server

    def register_tool(self, server_name: str, tool: ToolDefinition):
        """注册工具到索引"""
        self.tool_index[tool.name] = tool
        for tag in tool.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = []
            self.tag_index[tag].append(tool.name)

        if server_name in self.servers:
            self.servers[server_name].tools.append(tool)

        print(f"  ✓ 注册工具: {tool.name} (from {server_name})")

    def discover_tools(self, query: str = None,
                       tags: list[str] = None) -> list[ToolDefinition]:
        """发现工具 — 支持关键词搜索和标签过滤"""
        tools = list(self.tool_index.values())

        if query:
            query_lower = query.lower()
            tools = [
                t for t in tools
                if query_lower in t.name.lower() or query_lower in t.description.lower()
            ]

        if tags:
            tag_tools = set()
            for tag in tags:
                tag_tools.update(self.tag_index.get(tag, []))
            tools = [t for t in tools if t.name in tag_tools]

        return tools

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """获取工具定义"""
        return self.tool_index.get(name)

    def route_tool_call(self, tool_name: str) -> Optional[ServerRegistration]:
        """路由工具调用到对应的 Server"""
        tool = self.tool_index.get(tool_name)
        if not tool:
            return None
        return self.servers.get(tool.server_name)

    def to_openai_functions(self, tools: list[ToolDefinition] = None) -> list[dict]:
        """将工具导出为 OpenAI Function Calling 格式"""
        if tools is None:
            tools = list(self.tool_index.values())

        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                }
            }
            for t in tools
        ]

    def status_report(self) -> dict:
        """状态报告"""
        return {
            "total_servers": len(self.servers),
            "total_tools": len(self.tool_index),
            "servers": {
                name: {
                    "status": s.status,
                    "tool_count": len(s.tools),
                    "transport": s.transport,
                }
                for name, s in self.servers.items()
            },
            "tools_by_tag": {
                tag: len(tools) for tag, tools in self.tag_index.items()
            },
        }


# =============================================================================
# 演示
# =============================================================================

def demo_registry():
    """工具注册中心演示"""
    print("=" * 60)
    print("MCP Tool Registry 演示")
    print("=" * 60)

    registry = ToolRegistry()

    # 注册 Server
    registry.register_server(
        "knowledge-server", "localhost:8001",
        transport="stdio", command="python", args=["mcp_server.py", "--serve"],
    )
    registry.register_server(
        "code-server", "localhost:8002",
        transport="sse",
    )
    registry.register_server(
        "monitoring-server", "localhost:8003",
        transport="sse",
    )

    # 注册工具
    tools = [
        ToolDefinition(
            name="search_knowledge", description="搜索内部知识库",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            server_name="knowledge-server", server_uri="localhost:8001",
            tags=["knowledge", "search"],
        ),
        ToolDefinition(
            name="get_metrics", description="获取系统监控指标",
            input_schema={"type": "object", "properties": {"metric_type": {"type": "string"}}},
            server_name="monitoring-server", server_uri="localhost:8003",
            tags=["monitoring", "metrics"],
        ),
        ToolDefinition(
            name="run_code", description="在沙箱中执行代码",
            input_schema={"type": "object", "properties": {"code": {"type": "string"}}},
            server_name="code-server", server_uri="localhost:8002",
            tags=["code", "execution"],
        ),
        ToolDefinition(
            name="create_ticket", description="创建工单",
            input_schema={"type": "object", "properties": {"title": {"type": "string"}}},
            server_name="knowledge-server", server_uri="localhost:8001",
            tags=["knowledge", "workflow"],
        ),
    ]

    for tool in tools:
        registry.register_tool(tool.server_name, tool)

    # 工具发现
    print(f"\n--- 工具发现 ---")

    all_tools = registry.discover_tools()
    print(f"\n所有工具 ({len(all_tools)} 个):")
    for t in all_tools:
        print(f"  {t.name}: {t.description} [tags: {t.tags}]")

    search_tools = registry.discover_tools(query="搜索")
    print(f"\n搜索 '搜索' ({len(search_tools)} 个):")
    for t in search_tools:
        print(f"  {t.name}: {t.description}")

    knowledge_tools = registry.discover_tools(tags=["knowledge"])
    print(f"\n标签 'knowledge' ({len(knowledge_tools)} 个):")
    for t in knowledge_tools:
        print(f"  {t.name}: {t.description}")

    # 路由
    print(f"\n--- 工具路由 ---")
    for tool_name in ["search_knowledge", "run_code"]:
        server = registry.route_tool_call(tool_name)
        if server:
            print(f"  {tool_name} → Server: {server.name} ({server.transport}://{server.uri})")

    # OpenAI Functions 导出
    print(f"\n--- OpenAI Functions 格式 ---")
    functions = registry.to_openai_functions()
    print(json.dumps(functions[:2], indent=2, ensure_ascii=False)[:500])

    # 状态报告
    print(f"\n--- 状态报告 ---")
    report = registry.status_report()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    demo_registry()
