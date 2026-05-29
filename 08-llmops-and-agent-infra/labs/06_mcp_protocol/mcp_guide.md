# MCP (Model Context Protocol) 详细指南

## 1. MCP 是什么

MCP 是 Anthropic 于 2024 年提出的开放标准协议，定义了 LLM 应用与外部数据源和工具之间的标准化交互方式。

**核心理念**：类似 USB-C 统一了设备接口，MCP 统一了 LLM 与外部世界的交互接口。

## 2. 核心架构

```
┌─────────────────────────────────────────────┐
│                 MCP 架构                     │
│                                              │
│  Host (宿主应用)                             │
│  ├── Claude Desktop / IDE / 自定义应用       │
│  │                                           │
│  │   ┌─────────────────┐                    │
│  │   │   MCP Client    │ ← 每个连接一个      │
│  │   └────────┬────────┘                    │
│  │            │ MCP Protocol                │
│  │   ┌────────▼────────┐                    │
│  │   │   MCP Server    │ ← 提供能力         │
│  │   │                 │                    │
│  │   │ • Tools         │ ← 可执行的操作      │
│  │   │ • Resources     │ ← 上下文数据        │
│  │   │ • Prompts       │ ← 预定义模板        │
│  │   └─────────────────┘                    │
│  │                                           │
│  │   可以同时连接多个 Server                  │
└─────────────────────────────────────────────┘
```

## 3. 三大能力

### 3.1 Tools（工具）
- LLM 可以调用的操作
- 需要用户/LLM 决定何时调用
- 例：搜索数据库、发送邮件、执行代码

### 3.2 Resources（资源）
- 提供上下文数据
- 由应用控制何时读取
- 例：文件内容、API 响应、配置数据

### 3.3 Prompts（提示模板）
- 预定义的交互模板
- 用户可选择的工作流
- 例：代码审查模板、写作模板

## 4. 传输方式

### stdio（标准输入输出）
```
适用：本地工具、开发调试
通信：JSON-RPC over stdin/stdout
示例：Claude Desktop 本地插件
```

### SSE / HTTP Streamable
```
适用：远程服务、生产部署
通信：HTTP + Server-Sent Events
示例：云端 MCP Server
```

## 5. 实际使用

### 5.1 Claude Desktop 配置
```json
// ~/Library/Application Support/Claude/claude_desktop_config.json
{
  "mcpServers": {
    "knowledge": {
      "command": "python",
      "args": ["path/to/mcp_server.py", "--serve"]
    }
  }
}
```

### 5.2 代码中使用
```python
# Server 端
from mcp.server import Server
server = Server("my-server")

@server.list_tools()
async def list_tools():
    return [Tool(name="...", description="...", inputSchema={...})]

@server.call_tool()
async def call_tool(name, arguments):
    # 执行工具逻辑
    return [TextContent(type="text", text="结果")]

# Client 端
from mcp import ClientSession
session = ClientSession(read, write)
tools = await session.list_tools()
result = await session.call_tool("tool_name", {"arg": "value"})
```

## 6. 生产建议

1. **安全**：MCP Server 应实现认证和授权
2. **超时**：设置合理的工具执行超时
3. **日志**：记录所有工具调用用于审计
4. **限流**：防止 LLM 过度调用工具
5. **版本**：工具 Schema 变更需要版本管理

## 7. 与 Function Calling 的关系

```
Function Calling = LLM 调用工具的"接口格式"
MCP = 工具"提供和发现"的标准协议

两者互补：
  MCP Server 暴露工具 → 转换为 Function Calling Schema → LLM 调用
```
