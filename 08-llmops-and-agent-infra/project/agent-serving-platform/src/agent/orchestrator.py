"""多 Agent 编排器 - 协调多个 Agent 完成复杂任务"""
import asyncio
from typing import Optional
from dataclasses import dataclass, field

from .agent_registry import AgentRegistry, AgentConfig
from .executor import AgentExecutor


@dataclass
class OrchestrationPlan:
    steps: list[dict]  # [{"agent_id": ..., "task": ..., "depends_on": [...]}]
    strategy: str = "sequential"  # sequential / parallel / conditional


@dataclass
class OrchestrationResult:
    task: str
    steps_executed: int
    results: list[dict] = field(default_factory=list)
    final_answer: str = ""
    total_tokens: int = 0
    total_latency_ms: float = 0


class AgentOrchestrator:
    """
    多 Agent 编排器
    支持：顺序执行、并行执行、条件分支、循环
    """

    def __init__(self, registry: AgentRegistry, executor: AgentExecutor):
        self.registry = registry
        self.executor = executor

    async def execute_plan(self, plan: OrchestrationPlan,
                           context: dict = None) -> OrchestrationResult:
        """执行编排计划"""
        if plan.strategy == "sequential":
            return await self._execute_sequential(plan, context)
        elif plan.strategy == "parallel":
            return await self._execute_parallel(plan, context)
        else:
            return await self._execute_sequential(plan, context)

    async def _execute_sequential(self, plan, context) -> OrchestrationResult:
        """顺序执行"""
        result = OrchestrationResult(task=str(plan.steps))
        accumulated_context = dict(context or {})

        for step in plan.steps:
            agent_id = step["agent_id"]
            task = step["task"]

            # 注入上下文
            if accumulated_context:
                task = f"{task}\n\n上下文: {accumulated_context.get('last_result', '')}"

            # 执行
            step_result = await self.executor.execute(
                agent_id=agent_id, query=task, context=accumulated_context,
            )

            result.results.append({
                "agent_id": agent_id,
                "task": step["task"],
                "output": step_result.get("answer", ""),
                "tokens": step_result.get("tokens", 0),
            })
            result.steps_executed += 1
            accumulated_context["last_result"] = step_result.get("answer", "")

        result.final_answer = accumulated_context.get("last_result", "")
        return result

    async def _execute_parallel(self, plan, context) -> OrchestrationResult:
        """并行执行"""
        result = OrchestrationResult(task=str(plan.steps))

        tasks = []
        for step in plan.steps:
            task = self.executor.execute(
                agent_id=step["agent_id"],
                query=step["task"],
                context=context,
            )
            tasks.append(task)

        step_results = await asyncio.gather(*tasks, return_exceptions=True)

        for step, step_result in zip(plan.steps, step_results):
            if isinstance(step_result, Exception):
                result.results.append({
                    "agent_id": step["agent_id"],
                    "error": str(step_result),
                })
            else:
                result.results.append({
                    "agent_id": step["agent_id"],
                    "output": step_result.get("answer", ""),
                })
            result.steps_executed += 1

        # 合并结果
        outputs = [r.get("output", "") for r in result.results if "output" in r]
        result.final_answer = "\n\n".join(outputs)
        return result

    def create_plan(self, task: str, available_agents: list[str]) -> OrchestrationPlan:
        """根据任务自动生成编排计划（简化版）"""
        # 生产环境中应使用 LLM 来生成计划
        if len(available_agents) == 1:
            return OrchestrationPlan(
                steps=[{"agent_id": available_agents[0], "task": task}],
                strategy="sequential",
            )

        # 默认：研究 → 执行 → 审核
        steps = []
        if "rag_agent" in available_agents:
            steps.append({"agent_id": "rag_agent", "task": f"搜索相关信息：{task}"})
        steps.append({"agent_id": available_agents[0], "task": task})

        return OrchestrationPlan(steps=steps, strategy="sequential")
