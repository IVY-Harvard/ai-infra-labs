"""
Lab 06: MCP Client 实现
连接 MCP Server，列出工具并调用
"""
import asyncio
import json
from typing import Optional

try:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("mcp 未安装，请运行: pip install mcp")


# =============================================================================
# MCP Client
# =============================================================================

class MCPClient:
    """MCP 客户端 — 连接并调用 MCP Server"""

    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.tools: list = []

    async def connect(self, server_command: str, server_args: list[str] = None):
        """连接到 MCP Server"""
        server_params = StdioServerParameters(
            command=server_command,
            args=server_args or [],
        )

        transport = await stdio_client(server_params).__aenter__()
        read, write = transport
        self.session = ClientSession(read, write)
        await self.session.__aenter__()
        await self.session.initialize()

        print(f"✓ 已连接 MCP Server: {server_command}")

    async def list_tools(self) -> list:
        """列出所有可用工具"""
        if not self.session:
            raise RuntimeError("未连接到 Server")

        response = await self.session.list_tools()
        self.tools = response.tools

        print(f"\n可用工具 ({len(self.tools)} 个):")
        for tool in self.tools:
            print(f"  - {tool.name}: {tool.description}")
            if tool.inputSchema:
                props = tool.inputSchema.get("properties", {})
                for prop_name, prop_info in props.items():
                    print(f"    参数 {prop_name}: {prop_info.get('description', '')}")

        return self.tools

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """调用工具"""
        if not self.session:
            raise RuntimeError("未连接到 Server")

        print(f"\n调用工具: {tool_name}")
        print(f"  参数: {json.dumps(arguments, ensure_ascii=False)}")

        result = await self.session.call_tool(tool_name, arguments)

        output = ""
        for content in result.content:
            if hasattr(content, "text"):
                output += content.text
        print(f"  结果: {output[:200]}...")

        return output

    async def list_resources(self) -> list:
        """列出可用资源"""
        if not self.session:
            raise RuntimeError("未连接到 Server")

        response = await self.session.list_resources()
        print(f"\n可用资源 ({len(response.resources)} 个):")
        for resource in response.resources:
            print(f"  - {resource.uri}: {resource.name}")

        return response.resources

    async def read_resource(self, uri: str) -> str:
        """读取资源"""
        if not self.session:
            raise RuntimeError("未连接到 Server")

        response = await self.session.read_resource(uri)
        content = response.contents[0].text if response.contents else ""
        print(f"\n读取资源 {uri}:")
        print(f"  {content[:200]}...")
        return content

    async def close(self):
        """关闭连接"""
        if self.session:
            await self.session.__aexit__(None, None, None)
            print("✓ 连接已关闭")


# =============================================================================
# 与 LLM 集成使用 MCP
# =============================================================================

class MCPLLMIntegration:
    """
    将 MCP 工具集成到 LLM 的 Function Calling 中
    MCP Tools → OpenAI Function Calling Schema → LLM 调用
    """

    @staticmethod
    def mcp_tools_to_openai_functions(mcp_tools: list) -> list[dict]:
        """将 MCP 工具定义转换为 OpenAI Function Calling 格式"""
        functions = []
        for tool in mcp_tools:
            functions.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                }
            })
        return functions

    @staticmethod
    def demo_conversion():
        """演示 MCP → OpenAI Functions 转换"""
        print("\n" + "=" * 60)
        print("MCP Tools → OpenAI Functions 转换示例")
        print("=" * 60)

        # 模拟 MCP 工具定义
        mock_tools = [
            {
                "name": "search_knowledge",
                "description": "搜索内部知识库",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词"}
                    },
                    "required": ["query"],
                },
            }
        ]

        # 转换为 OpenAI 格式
        openai_format = [{
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["inputSchema"],
            }
        } for t in mock_tools]

        print(f"\nMCP 格式:\n{json.dumps(mock_tools, indent=2, ensure_ascii=False)}")
        print(f"\nOpenAI 格式:\n{json.dumps(openai_format, indent=2, ensure_ascii=False)}")


# =============================================================================
# 演示（不需要实际 MCP Server）
# =============================================================================

def demo_client_concept():
    """MCP Client 概念演示"""
    print("=" * 60)
    print("MCP Client 概念演示")
    print("=" * 60)

    print("""
MCP 交互流程:

1. Client 连接 Server (stdio 或 SSE)
   client.connect("python mcp_server.py --serve")

2. Client 发现 Server 能力
   tools = client.list_tools()
   resources = client.list_resources()

3. Client（通过 LLM）调用工具
   LLM 分析用户问题 → 选择工具 → 生成参数
   result = client.call_tool("search_knowledge", {"query": "GPU"})

4. Client 读取资源
   content = client.read_resource("knowledge://gpu_policy")

实际使用示例（需要 Server 运行）:

  # 终端 1: 启动 Server
  python mcp_server.py --serve

  # 终端 2: 运行 Client
  python mcp_client.py --connect
""")

    MCPLLMIntegration.demo_conversion()


# =============================================================================
# 主程序
# =============================================================================

async def main_async():
    """异步主程序 — 连接真实 MCP Server"""
    client = MCPClient()
    try:
        await client.connect("python", ["mcp_server.py", "--serve"])
        await client.list_tools()
        await client.list_resources()
        await client.call_tool("search_knowledge", {"query": "GPU"})
        await client.call_tool("get_metrics", {"metric_type": "rag_quality"})
        await client.read_resource("knowledge://gpu_policy")
    finally:
        await client.close()


if __name__ == "__main__":
    import sys
    if "--connect" in sys.argv and MCP_AVAILABLE:
        asyncio.run(main_async())
    else:
        demo_client_concept()
