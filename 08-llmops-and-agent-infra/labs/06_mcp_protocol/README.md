# Lab 06: MCP (Model Context Protocol) 实战

## 目标
理解并实现 MCP 协议：构建 MCP Server 和 Client，实现工具注册与调用。

## 前置准备
```bash
pip install mcp httpx
```

## 实验内容
1. `mcp_server.py` — MCP 服务端实现（提供 Tools 和 Resources）
2. `mcp_client.py` — MCP 客户端实现（连接并调用 Server）
3. `tool_registry.py` — 工具注册中心（管理多个 MCP Server）
4. `mcp_guide.md` — MCP 协议详细指南

## 核心概念
- MCP Server 暴露 Tools（可执行操作）和 Resources（上下文数据）
- MCP Client（通常是 LLM 应用）连接 Server 并调用能力
- 传输方式：stdio（本地）/ SSE（远程）
