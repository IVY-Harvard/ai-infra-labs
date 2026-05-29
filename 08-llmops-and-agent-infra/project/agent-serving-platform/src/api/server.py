"""FastAPI 服务入口"""
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

from ..gateway.router import RequestRouter
from ..gateway.rate_limiter import RateLimiter
from ..agent.agent_registry import AgentRegistry, AgentConfig, AgentStatus
from ..agent.executor import AgentExecutor
from ..agent.orchestrator import AgentOrchestrator
from ..memory.memory_manager import MemoryManager
from ..evaluation.online_evaluator import OnlineEvaluator
from ..evaluation.quality_gate import QualityGate


# =============================================================================
# 全局组件
# =============================================================================

registry = AgentRegistry()
executor = AgentExecutor(registry)
orchestrator = AgentOrchestrator(registry, executor)
memory_manager = MemoryManager()
rate_limiter = RateLimiter()
evaluator = OnlineEvaluator(sample_rate=0.1)
quality_gate = QualityGate()
router = RequestRouter(registry)


# =============================================================================
# 启动时注册默认 Agent
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时注册默认 Agent
    default_agents = [
        AgentConfig(
            agent_id="general_agent", name="通用助手",
            description="通用问答 Agent", version="1.0.0",
            system_prompt="你是一个有帮助的 AI 助手。",
        ),
        AgentConfig(
            agent_id="rag_agent", name="知识问答",
            description="基于 RAG 的知识问答 Agent", version="1.0.0",
            system_prompt="你是一个知识问答专家，请基于提供的文档回答问题。",
        ),
        AgentConfig(
            agent_id="code_agent", name="编程助手",
            description="代码生成和调试 Agent", version="1.0.0",
            system_prompt="你是一个资深 Python 开发工程师。",
        ),
    ]
    for agent in default_agents:
        registry.register(agent)

    yield  # 应用运行中

    # 关闭时清理


app = FastAPI(
    title="Agent Serving Platform",
    version="1.0.0",
    description="企业级 Agent 服务平台",
    lifespan=lifespan,
)


# =============================================================================
# 请求/响应模型
# =============================================================================

class ChatRequest(BaseModel):
    query: str
    user_id: str = "anonymous"
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    version: Optional[str] = None
    context: Optional[dict] = None


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    agent_id: str
    version: str
    latency_ms: float
    metadata: dict = Field(default_factory=dict)


class AgentRegisterRequest(BaseModel):
    agent_id: str
    name: str
    description: str
    version: str
    model: str = "qwen2.5-72b"
    system_prompt: str = ""
    tools: list[str] = []
    temperature: float = 0.3


# =============================================================================
# API 端点
# =============================================================================

@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """单轮对话"""
    request_id = str(uuid.uuid4())[:8]

    # 限流
    limit_result = rate_limiter.check(request.user_id)
    if not limit_result.allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limited. Retry after {limit_result.retry_after:.1f}s",
        )

    # 路由
    route = router.route({
        "query": request.query,
        "user_id": request.user_id,
        "agent_id": request.agent_id,
    })

    # 获取上下文
    session_id = request.session_id or request_id
    context = memory_manager.get_context(
        session_id, request.user_id, request.query
    )

    # 执行
    result = await executor.execute(
        agent_id=route.agent_id,
        query=request.query,
        context={**(request.context or {}), **context},
        version=route.version,
    )

    # 记忆
    memory_manager.add_interaction(
        session_id, request.user_id,
        request.query, result.get("answer", ""),
    )

    # 在线评估（采样）
    if evaluator.should_evaluate():
        await evaluator.evaluate_async(
            request_id, request.query,
            result.get("answer", ""),
        )

    return ChatResponse(
        request_id=request_id,
        answer=result.get("answer", ""),
        agent_id=route.agent_id,
        version=route.version,
        latency_ms=result.get("latency_ms", 0),
        metadata={"is_canary": route.is_canary},
    )


@app.post("/v1/agents")
async def register_agent(request: AgentRegisterRequest):
    """注册 Agent"""
    config = AgentConfig(
        agent_id=request.agent_id,
        name=request.name,
        description=request.description,
        version=request.version,
        model=request.model,
        system_prompt=request.system_prompt,
        tools=request.tools,
        temperature=request.temperature,
    )
    key = registry.register(config)
    return {"status": "registered", "key": key}


@app.get("/v1/agents")
async def list_agents():
    """列出所有 Agent"""
    agents = registry.list_agents(status=AgentStatus.ACTIVE)
    return {
        "agents": [
            {
                "agent_id": a.agent_id,
                "name": a.name,
                "version": a.version,
                "status": a.status.value,
            }
            for a in agents
        ]
    }


@app.get("/v1/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "agents": len(registry.list_agents()),
        "sessions": memory_manager.short_term.get_session_count(),
    }


@app.get("/v1/metrics")
async def metrics():
    """评估指标"""
    return {
        "online_eval": evaluator.get_recent_metrics(),
        "quality_gate_thresholds": quality_gate.thresholds,
    }
