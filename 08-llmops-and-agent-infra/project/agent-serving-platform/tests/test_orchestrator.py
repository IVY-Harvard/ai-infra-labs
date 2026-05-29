"""测试：Agent 编排器"""
import pytest
import asyncio
from src.agent.agent_registry import AgentRegistry, AgentConfig, AgentStatus
from src.agent.executor import AgentExecutor
from src.agent.orchestrator import AgentOrchestrator, OrchestrationPlan


@pytest.fixture
def registry():
    reg = AgentRegistry()
    reg.register(AgentConfig(
        agent_id="test_agent", name="Test Agent",
        description="A test agent", version="1.0.0",
        system_prompt="你是测试助手",
    ))
    reg.register(AgentConfig(
        agent_id="rag_agent", name="RAG Agent",
        description="RAG agent", version="1.0.0",
        system_prompt="你是知识问答助手",
    ))
    return reg


@pytest.fixture
def executor(registry):
    return AgentExecutor(registry)


@pytest.fixture
def orchestrator(registry, executor):
    return AgentOrchestrator(registry, executor)


class TestAgentRegistry:
    def test_register_and_get(self, registry):
        config = registry.get("test_agent", "1.0.0")
        assert config is not None
        assert config.name == "Test Agent"

    def test_get_latest(self, registry):
        registry.register(AgentConfig(
            agent_id="test_agent", name="Test Agent v2",
            description="Updated", version="2.0.0",
            system_prompt="v2",
        ))
        config = registry.get("test_agent", "latest")
        assert config.version == "2.0.0"

    def test_list_active_agents(self, registry):
        agents = registry.list_agents(status=AgentStatus.ACTIVE)
        assert len(agents) == 2

    def test_deactivate(self, registry):
        registry.deactivate("test_agent", "1.0.0")
        config = registry.get("test_agent", "1.0.0")
        assert config.status == AgentStatus.INACTIVE


class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_sequential_plan(self, orchestrator):
        plan = OrchestrationPlan(
            steps=[
                {"agent_id": "rag_agent", "task": "搜索 GPU 信息"},
                {"agent_id": "test_agent", "task": "总结结果"},
            ],
            strategy="sequential",
        )
        # 注意：实际测试需要 LLM 服务运行
        # result = await orchestrator.execute_plan(plan)
        # assert result.steps_executed == 2

    @pytest.mark.asyncio
    async def test_parallel_plan(self, orchestrator):
        plan = OrchestrationPlan(
            steps=[
                {"agent_id": "test_agent", "task": "任务1"},
                {"agent_id": "rag_agent", "task": "任务2"},
            ],
            strategy="parallel",
        )
        # result = await orchestrator.execute_plan(plan)
        # assert result.steps_executed == 2

    def test_create_plan(self, orchestrator):
        plan = orchestrator.create_plan(
            "分析数据", ["rag_agent", "test_agent"]
        )
        assert len(plan.steps) >= 1


class TestQualityGate:
    def test_pass(self):
        from src.evaluation.quality_gate import QualityGate
        gate = QualityGate()
        result = gate.check({
            "faithfulness": 0.92,
            "relevancy": 0.88,
            "correctness": 0.85,
            "safety": 0.99,
            "latency_p95_ms": 1500,
            "error_rate": 0.01,
        })
        assert result.passed is True

    def test_fail(self):
        from src.evaluation.quality_gate import QualityGate
        gate = QualityGate()
        result = gate.check({
            "faithfulness": 0.50,  # 低于阈值
            "relevancy": 0.88,
            "safety": 0.99,
        })
        assert result.passed is False
        assert len(result.failed_checks) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
