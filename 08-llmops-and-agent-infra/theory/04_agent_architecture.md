# Agent 架构

## 1. Agent 概述

### 1.1 什么是 LLM Agent

Agent = LLM + Planning + Memory + Tools

```
┌─────────────────────────────────────────────────┐
│                    Agent                          │
│                                                   │
│   ┌──────────────────────────────────────┐       │
│   │            LLM (大脑)                 │       │
│   │  理解意图 → 制定计划 → 决定行动       │       │
│   └──────────┬───────────────────────────┘       │
│              │                                    │
│   ┌──────────▼──────────┐                        │
│   │     Planning        │                        │
│   │  任务分解/策略选择   │                        │
│   └──────────┬──────────┘                        │
│              │                                    │
│   ┌──────────┼──────────┐                        │
│   ▼          ▼          ▼                        │
│ ┌─────┐  ┌─────┐  ┌─────┐                      │
│ │Memory│  │Tools│  │Action│                      │
│ │记忆  │  │工具 │  │执行  │                      │
│ └─────┘  └─────┘  └─────┘                      │
└─────────────────────────────────────────────────┘
```

### 1.2 Agent vs Chain

| 维度 | Chain（链） | Agent（代理） |
|------|------------|--------------|
| 控制流 | 预定义的固定流程 | LLM 动态决定下一步 |
| 工具调用 | 编排时确定 | 运行时决定 |
| 适应性 | 低（硬编码路径） | 高（根据反馈调整） |
| 可预测性 | 高 | 低 |
| 调试难度 | 低 | 高 |
| 适用场景 | 标准化流程 | 开放式问题 |

**工程决策**：优先用 Chain，只在需要动态决策时用 Agent。

## 2. Agent 范式

### 2.1 ReAct（Reasoning + Acting）

最经典的 Agent 范式，交替进行推理和行动：

```
┌─────────────────────────────────────────────────┐
│                  ReAct 循环                       │
│                                                   │
│   用户问题: "北京今天天气怎么样？适合跑步吗？"     │
│                                                   │
│   Thought 1: 需要先查询北京的天气信息              │
│   Action 1:  search_weather(city="北京")          │
│   Observation 1: 晴，25°C，AQI 45                 │
│                                                   │
│   Thought 2: 天气数据已获取，需要判断是否适合跑步   │
│   Action 2:  (无需工具，直接推理)                  │
│   Observation 2: -                                │
│                                                   │
│   Thought 3: 综合分析：晴天+适温+空气好=适合跑步   │
│   Final Answer: 北京今天晴朗，25°C，空气质量优...  │
└─────────────────────────────────────────────────┘
```

**核心 Prompt 结构**：
```python
REACT_PROMPT = """Answer the question using the following format:

Thought: reasoning about what to do
Action: tool_name[tool_input]
Observation: tool output
... (repeat Thought/Action/Observation)
Thought: I now know the final answer
Final Answer: the answer

Available tools: {tools}
Question: {question}
"""
```

**优劣势**：
- 优势：直觉性强，易于理解和调试
- 劣势：可能陷入循环，长链推理容易偏离

### 2.2 Plan-and-Execute

先制定计划再逐步执行：

```
┌─────────────────────────────────────────────────┐
│              Plan-and-Execute                      │
│                                                   │
│  Planning Phase:                                  │
│  ┌───────────────────────────────────┐           │
│  │  任务: 比较 Milvus 和 Qdrant      │           │
│  │                                    │           │
│  │  计划:                             │           │
│  │  1. 搜索 Milvus 最新特性和性能数据 │           │
│  │  2. 搜索 Qdrant 最新特性和性能数据 │           │
│  │  3. 对比两者的优劣势               │           │
│  │  4. 给出选型建议                   │           │
│  └───────────────────────────────────┘           │
│                                                   │
│  Execution Phase:                                 │
│  Step 1 → Execute → Result → Replan (if needed)  │
│  Step 2 → Execute → Result → Continue            │
│  Step 3 → Execute → Result → Continue            │
│  Step 4 → Execute → Final Answer                 │
│                                                   │
│  Re-planning: 执行过程中根据结果调整后续计划       │
└─────────────────────────────────────────────────┘
```

**优劣势**：
- 优势：适合复杂多步任务，计划可审查
- 劣势：计划制定消耗 Token，小任务开销大

### 2.3 Reflection（反思）

Agent 对自己的输出进行自我批评和改进：

```
┌─────────────────────────────────────────────────┐
│                  Reflection                       │
│                                                   │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐     │
│  │ Generate │ →  │ Reflect │ →  │ Improve │     │
│  │ 生成初稿 │    │ 自我批评 │    │ 改进输出 │     │
│  └─────────┘    └─────────┘    └─────────┘     │
│       ↑                              │           │
│       └──────────────────────────────┘           │
│              (迭代直到满意)                        │
│                                                   │
│  示例：                                           │
│  Round 1: 生成代码 → 发现缺少错误处理             │
│  Round 2: 添加错误处理 → 发现缺少日志              │
│  Round 3: 添加日志 → 质量达标，输出最终结果        │
└─────────────────────────────────────────────────┘
```

### 2.4 范式选型

```
任务类型                    推荐范式
  ├── 简单问答 + 工具调用   → ReAct
  ├── 复杂多步任务         → Plan-and-Execute
  ├── 内容生成 + 质量要求   → Reflection
  ├── 复杂推理 + 质量要求   → Plan-and-Execute + Reflection
  └── 多领域协作           → Multi-Agent
```

## 3. Multi-Agent 协作

### 3.1 为什么需要 Multi-Agent

单一 Agent 的局限：
- 上下文窗口有限，无法同时处理多领域知识
- 单一 Prompt 难以兼顾多种能力
- 长链推理容易失焦

Multi-Agent 的优势（类比微服务）：
```
单体 Agent                     Multi-Agent
┌──────────────┐              ┌────────┐ ┌────────┐
│ 所有能力     │      →       │ 搜索   │ │ 分析   │
│ 混在一起     │              │ Agent  │ │ Agent  │
│              │              └────────┘ └────────┘
│ 难以维护     │              ┌────────┐ ┌────────┐
│ 难以扩展     │              │ 编码   │ │ 审核   │
└──────────────┘              │ Agent  │ │ Agent  │
                              └────────┘ └────────┘
```

### 3.2 协作模式

**Supervisor（监督者模式）**：
```
                    ┌──────────────┐
                    │  Supervisor  │
                    │  (调度中心)   │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Worker A │ │ Worker B │ │ Worker C │
        │ (搜索)   │ │ (分析)   │ │ (写作)   │
        └──────────┘ └──────────┘ └──────────┘
```

**Hierarchical（层级模式）**：
```
                    ┌──────────────┐
                    │  Top Manager │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Team Lead│ │ Team Lead│ │ Team Lead│
        │ (研究组) │ │ (开发组) │ │ (测试组) │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
          ┌──┼──┐      ┌──┼──┐      ┌──┼──┐
          ▼  ▼  ▼      ▼  ▼  ▼      ▼  ▼  ▼
         W1 W2 W3     W4 W5 W6     W7 W8 W9
```

**Debate（辩论模式）**：
```
        ┌──────────┐           ┌──────────┐
        │ Agent A  │ ←──辩论──→ │ Agent B  │
        │ (正方)   │           │ (反方)   │
        └──────────┘           └──────────┘
              │                      │
              └──────────┬───────────┘
                         ▼
                  ┌──────────────┐
                  │    Judge     │
                  │  (裁判总结)  │
                  └──────────────┘
```

### 3.3 框架对比

**LangGraph**：
```python
# 基于有向图的 Agent 编排
# 优势：精确控制流程，状态管理强，生产就绪
# 适用：需要精确控制 Agent 行为的场景

from langgraph.graph import StateGraph

graph = StateGraph(State)
graph.add_node("researcher", researcher_agent)
graph.add_node("writer", writer_agent)
graph.add_node("reviewer", reviewer_agent)
graph.add_edge("researcher", "writer")
graph.add_conditional_edges("reviewer", should_revise,
    {"revise": "writer", "accept": END})
```

**AutoGen**：
```python
# 基于对话的 Multi-Agent 框架
# 优势：对话式交互自然，Group Chat 模式灵活
# 适用：需要 Agent 之间自由讨论的场景

from autogen import AssistantAgent, UserProxyAgent, GroupChat

researcher = AssistantAgent("researcher", system_message="...")
coder = AssistantAgent("coder", system_message="...")
critic = AssistantAgent("critic", system_message="...")

group_chat = GroupChat(agents=[researcher, coder, critic])
```

**CrewAI**：
```python
# 基于角色的 Agent 编排
# 优势：角色定义直观，任务分配清晰
# 适用：任务导向的团队协作场景

from crewai import Agent, Task, Crew

researcher = Agent(role="Senior Researcher", goal="...", tools=[...])
writer = Agent(role="Technical Writer", goal="...", tools=[...])

task = Task(description="...", expected_output="...", agent=researcher)
crew = Crew(agents=[researcher, writer], tasks=[...])
```

| 维度 | LangGraph | AutoGen | CrewAI |
|------|-----------|---------|--------|
| 控制粒度 | 最高（图定义） | 中（对话驱动） | 低（角色+任务） |
| 学习曲线 | 陡峭 | 中等 | 平缓 |
| 生产就绪 | ★★★★★ | ★★★ | ★★★ |
| 灵活性 | 最高 | 高 | 中 |
| 状态管理 | 内置 | 需自行管理 | 基础 |
| 人工介入 | 原生支持 | 支持 | 有限 |

## 4. Function Calling 机制

### 4.1 原理

Function Calling 让 LLM 以结构化方式调用外部工具：

```
┌─────────────────────────────────────────────────┐
│               Function Calling 流程              │
│                                                   │
│  1. 定义工具（JSON Schema）                       │
│     tools = [{                                   │
│       "name": "get_weather",                     │
│       "description": "获取天气信息",               │
│       "parameters": {                            │
│         "type": "object",                        │
│         "properties": {                          │
│           "city": {"type": "string"}             │
│         }                                        │
│       }                                          │
│     }]                                           │
│                                                   │
│  2. LLM 决定调用                                  │
│     User: "北京天气怎么样？"                       │
│     LLM → {"name": "get_weather",                │
│            "arguments": {"city": "北京"}}         │
│                                                   │
│  3. 应用层执行工具                                 │
│     result = get_weather(city="北京")             │
│                                                   │
│  4. 将结果返回 LLM                                │
│     LLM → "北京今天晴朗，气温25°C..."             │
└─────────────────────────────────────────────────┘
```

### 4.2 Parallel Function Calling

现代 LLM 支持并行调用多个工具：

```python
# LLM 可以一次返回多个工具调用
# "北京和上海的天气分别是什么？"
tool_calls = [
    {"name": "get_weather", "arguments": {"city": "北京"}},
    {"name": "get_weather", "arguments": {"city": "上海"}},
]
# 应用层并行执行，提升效率
```

### 4.3 工具设计原则

```
好的工具设计（类比好的 API 设计）：
├── 单一职责：每个工具做一件事
├── 明确描述：让 LLM 理解何时使用
├── 参数校验：Schema 严格定义
├── 错误处理：返回有意义的错误信息
├── 幂等性：相同输入产生相同结果
└── 超时控制：避免工具调用阻塞
```

## 5. MCP（Model Context Protocol）

### 5.1 MCP 概述

MCP 是 Anthropic 提出的开放标准，定义了 LLM 应用与外部数据/工具的标准化交互协议。

```
┌─────────────────────────────────────────────────┐
│                MCP 架构                           │
│                                                   │
│   传统方式：每个 LLM 应用自己对接每个工具        │
│   ┌─────┐     ┌─────┐                           │
│   │App 1│──×──│Tool1│  N 个应用 × M 个工具      │
│   │App 2│──×──│Tool2│  = N×M 个适配器           │
│   │App 3│──×──│Tool3│                            │
│   └─────┘     └─────┘                           │
│                                                   │
│   MCP 方式：统一协议                              │
│   ┌─────┐            ┌─────────┐    ┌─────┐     │
│   │App 1│──┐         │  MCP    │──→─│Tool1│     │
│   │App 2│──┼── MCP ──│ Server  │──→─│Tool2│     │
│   │App 3│──┘         │         │──→─│Tool3│     │
│   └─────┘            └─────────┘    └─────┘     │
│   (MCP Client)       (标准接口)    (MCP Server)  │
└─────────────────────────────────────────────────┘
```

### 5.2 MCP 核心概念

```
MCP 协议三大能力：

1. Resources（资源）
   • 类比 REST API 的 GET 端点
   • 提供上下文数据给 LLM
   • 例：文件内容、数据库记录、API 响应

2. Tools（工具）
   • 类比 REST API 的 POST/PUT/DELETE 端点
   • LLM 可以调用的操作
   • 例：执行查询、发送消息、创建资源

3. Prompts（提示模板）
   • 预定义的交互模板
   • 用户可选择的标准化工作流
   • 例：代码审查模板、文档生成模板
```

### 5.3 MCP 传输方式

```
Stdio（标准输入输出）：
  • 本地进程间通信
  • 适合 CLI 工具和本地开发
  • Claude Desktop / Claude Code 使用

SSE (Server-Sent Events) / HTTP Streamable：
  • 基于 HTTP 的远程通信
  • 适合网络服务和生产部署
  • 支持认证和授权
```

### 5.4 MCP 生态

```
官方 MCP Server：
  ├── filesystem  — 文件系统操作
  ├── github      — GitHub API
  ├── postgres    — PostgreSQL 查询
  ├── slack       — Slack 消息
  ├── puppeteer   — 浏览器自动化
  └── ...

社区 MCP Server：
  ├── mcp-server-kubernetes — K8s 管理
  ├── mcp-server-docker     — Docker 操作
  ├── mcp-server-redis      — Redis 操作
  └── 数百个社区贡献的 Server
```

## 6. Agent 安全与可靠性

### 6.1 Agent 风险

```
风险类别          具体风险                  缓解措施
├── Prompt 注入   恶意用户操控 Agent        输入过滤 + 权限隔离
├── 工具滥用      Agent 执行危险操作        权限白名单 + 人工审批
├── 无限循环      Agent 陷入推理循环        最大步数限制 + 超时
├── 幻觉行动      基于错误推理执行操作      关键操作双重确认
├── 成本失控      过多 Token 消耗           预算上限 + 监控告警
└── 数据泄露      Agent 访问敏感数据        数据分级 + 访问控制
```

### 6.2 生产安全机制

```python
class SafeAgentExecutor:
    """生产级 Agent 执行器 - 带安全护栏"""
    
    def __init__(self, agent, max_steps=10, max_tokens=50000):
        self.agent = agent
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.token_usage = 0
    
    async def execute(self, task: str) -> str:
        for step in range(self.max_steps):
            # 1. 获取 Agent 决策
            action = await self.agent.decide(task)
            
            # 2. 安全检查
            if not self._is_action_safe(action):
                return "操作被安全策略阻止"
            
            # 3. 预算检查
            if self.token_usage > self.max_tokens:
                return "Token 预算耗尽"
            
            # 4. 执行并记录
            result = await self._execute_with_timeout(action)
            self._log_action(step, action, result)
            
            if action.is_final:
                return result
        
        return "达到最大步数限制"
```

## 7. Agent 可观测性

### 7.1 Trace 结构

```
Agent Trace（类比分布式追踪的 Span）：

Trace: user_query_123
  ├── Span: planning (200ms, 500 tokens)
  │     └── LLM call: plan generation
  ├── Span: step_1 (1500ms, 300 tokens)
  │     ├── LLM call: decide action
  │     ├── Tool call: search_database (800ms)
  │     └── LLM call: process result
  ├── Span: step_2 (2000ms, 400 tokens)
  │     ├── LLM call: decide action
  │     ├── Tool call: call_api (1200ms)
  │     └── LLM call: process result
  └── Span: final_answer (300ms, 200 tokens)
        └── LLM call: generate response

Total: 4000ms, 1400 tokens, 2 tool calls
```

## 8. 小结

Agent 架构选择取决于任务复杂度：

```
简单任务 → Function Calling（无需 Agent）
中等任务 → ReAct Agent（单 Agent + 工具）
复杂任务 → Plan-and-Execute（单 Agent + 计划）
协作任务 → Multi-Agent（LangGraph/AutoGen）
标准化  → MCP（工具协议标准化）
```

关键原则：
1. **能用 Chain 不用 Agent**：可预测性优先
2. **能用单 Agent 不用多 Agent**：简单性优先
3. **始终设置安全护栏**：最大步数、Token 预算、权限控制
4. **全链路追踪**：每个决策步骤都要可观测
