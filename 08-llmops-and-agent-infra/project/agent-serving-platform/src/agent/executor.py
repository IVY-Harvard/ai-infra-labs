"""Agent 执行器 - 执行单个 Agent 的推理"""
import os
import time
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from .agent_registry import AgentRegistry, AgentConfig


LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")


class AgentExecutor:
    """Agent 执行器 - 带安全护栏"""

    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self.max_tokens_per_request = 50000
        self.max_execution_time = 30  # seconds

    async def execute(self, agent_id: str, query: str,
                      context: dict = None, version: str = "latest") -> dict:
        """执行 Agent"""
        config = self.registry.get(agent_id, version)
        if not config:
            return {"error": f"Agent not found: {agent_id}", "answer": ""}

        start_time = time.time()

        try:
            llm = ChatOpenAI(
                base_url=LLM_BASE_URL,
                model=config.model,
                api_key="not-needed",
                temperature=config.temperature,
                max_tokens=2000,
            )

            prompt = ChatPromptTemplate.from_template(
                """{system_prompt}

{context_section}

用户问题：{query}

请给出回答："""
            )

            context_section = ""
            if context and context.get("last_result"):
                context_section = f"参考信息：\n{context['last_result']}"

            chain = prompt | llm | StrOutputParser()
            answer = chain.invoke({
                "system_prompt": config.system_prompt or "你是一个有帮助的助手。",
                "context_section": context_section,
                "query": query,
            })

            latency = (time.time() - start_time) * 1000

            return {
                "answer": answer,
                "agent_id": agent_id,
                "version": config.version,
                "latency_ms": latency,
                "tokens": len(answer),  # 简化估算
                "metadata": {"model": config.model},
            }

        except Exception as e:
            return {
                "error": str(e),
                "answer": f"Agent 执行失败: {e}",
                "agent_id": agent_id,
                "latency_ms": (time.time() - start_time) * 1000,
            }
