"""
Lab 06: MCP Server 实现
构建一个提供知识库查询和数据分析工具的 MCP Server
"""
import json
import asyncio
from datetime import datetime
from typing import Any

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool, TextContent, Resource, ResourceTemplate,
        GetPromptResult, PromptMessage,
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("mcp 未安装，请运行: pip install mcp")


# =============================================================================
# 模拟的知识库和数据
# =============================================================================

KNOWLEDGE_BASE = {
    "gpu_policy": {
        "title": "GPU 资源使用政策",
        "content": "公司有 8 张 H20 GPU。申请使用需填写 GPU 申请单。"
                   "训练任务优先级低于推理服务。每个团队最多占用 4 张卡。",
        "updated": "2024-11-01",
    },
    "rag_guide": {
        "title": "RAG 系统开发指南",
        "content": "推荐使用 LangChain + Milvus 构建 RAG。Embedding 模型选用 bge-m3。"
                   "分块大小建议 500-1000 字符，overlap 50-200。",
        "updated": "2024-10-15",
    },
    "deployment": {
        "title": "LLM 服务部署手册",
        "content": "使用 vLLM 部署 LLM 推理服务。72B 模型需要 4 卡 TP。"
                   "建议配置 PagedAttention，设置 max_model_len=4096。",
        "updated": "2024-11-10",
    },
}

METRICS_DATA = {
    "rag_quality": {"faithfulness": 0.92, "relevancy": 0.88, "latency_p95_ms": 1500},
    "gpu_usage": {"gpu_0": 0.85, "gpu_1": 0.72, "gpu_2": 0.91, "gpu_3": 0.65},
    "api_stats": {"qps": 150, "error_rate": 0.02, "avg_tokens": 850},
}


# =============================================================================
# MCP Server 定义
# =============================================================================

if MCP_AVAILABLE:

    server = Server("knowledge-server")

    # ----- Tools -----

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """列出所有可用工具"""
        return [
            Tool(
                name="search_knowledge",
                description="搜索内部知识库。输入关键词，返回匹配的文档。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索关键词",
                        }
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="get_metrics",
                description="获取系统监控指标。可选类型：rag_quality, gpu_usage, api_stats",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "metric_type": {
                            "type": "string",
                            "enum": ["rag_quality", "gpu_usage", "api_stats"],
                            "description": "指标类型",
                        }
                    },
                    "required": ["metric_type"],
                },
            ),
            Tool(
                name="create_ticket",
                description="创建工单。用于提交 GPU 申请、问题报告等。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "工单标题"},
                        "description": {"type": "string", "description": "工单描述"},
                        "priority": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "critical"],
                            "description": "优先级",
                        },
                    },
                    "required": ["title", "description"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """执行工具调用"""
        if name == "search_knowledge":
            return await _search_knowledge(arguments["query"])
        elif name == "get_metrics":
            return await _get_metrics(arguments["metric_type"])
        elif name == "create_ticket":
            return await _create_ticket(
                arguments["title"],
                arguments["description"],
                arguments.get("priority", "medium"),
            )
        else:
            return [TextContent(type="text", text=f"未知工具: {name}")]

    async def _search_knowledge(query: str) -> list[TextContent]:
        """搜索知识库"""
        results = []
        for key, doc in KNOWLEDGE_BASE.items():
            if query.lower() in doc["title"].lower() or query.lower() in doc["content"].lower():
                results.append(f"[{doc['title']}] (更新: {doc['updated']})\n{doc['content']}")

        if results:
            return [TextContent(type="text", text="\n\n---\n\n".join(results))]
        return [TextContent(type="text", text=f"未找到与 '{query}' 相关的文档")]

    async def _get_metrics(metric_type: str) -> list[TextContent]:
        """获取监控指标"""
        metrics = METRICS_DATA.get(metric_type)
        if metrics:
            formatted = json.dumps(metrics, indent=2, ensure_ascii=False)
            return [TextContent(type="text", text=f"[{metric_type}] 指标:\n{formatted}")]
        return [TextContent(type="text", text=f"未知指标类型: {metric_type}")]

    async def _create_ticket(title, description, priority) -> list[TextContent]:
        """创建工单"""
        ticket_id = f"TICKET-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        result = {
            "id": ticket_id,
            "title": title,
            "description": description,
            "priority": priority,
            "status": "created",
            "created_at": datetime.now().isoformat(),
        }
        return [TextContent(
            type="text",
            text=f"工单创建成功:\n{json.dumps(result, indent=2, ensure_ascii=False)}"
        )]

    # ----- Resources -----

    @server.list_resources()
    async def list_resources() -> list[Resource]:
        """列出可用资源"""
        resources = []
        for key, doc in KNOWLEDGE_BASE.items():
            resources.append(Resource(
                uri=f"knowledge://{key}",
                name=doc["title"],
                description=f"知识库文档: {doc['title']}",
                mimeType="text/plain",
            ))
        return resources

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        """读取资源内容"""
        # 解析 URI: knowledge://gpu_policy
        key = uri.replace("knowledge://", "")
        doc = KNOWLEDGE_BASE.get(key)
        if doc:
            return doc["content"]
        return f"资源不存在: {uri}"

    # ----- 启动 -----

    async def run_server():
        """启动 MCP Server（stdio 模式）"""
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream)


# =============================================================================
# 独立运行演示（不需要 MCP 传输层）
# =============================================================================

def demo_standalone():
    """独立演示 MCP Server 的功能"""
    print("=" * 60)
    print("MCP Server 功能演示（独立模式）")
    print("=" * 60)

    # 模拟工具调用
    print("\n1. 搜索知识库:")
    for key, doc in KNOWLEDGE_BASE.items():
        if "gpu" in key.lower():
            print(f"  [{doc['title']}]: {doc['content'][:80]}...")

    print("\n2. 获取监控指标:")
    for metric_type, metrics in METRICS_DATA.items():
        print(f"  [{metric_type}]: {json.dumps(metrics)}")

    print("\n3. 创建工单:")
    ticket_id = f"TICKET-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    print(f"  工单 {ticket_id} 创建成功")

    print("\n4. 可用资源:")
    for key, doc in KNOWLEDGE_BASE.items():
        print(f"  knowledge://{key} - {doc['title']}")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    if MCP_AVAILABLE and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "--serve":
        # 作为 MCP Server 运行
        asyncio.run(run_server())
    else:
        # 独立演示
        demo_standalone()
