"""
Lab 05: 多 Agent 协作系统
实现一个完整的研究-编码-审核多 Agent 系统
"""
import os
import asyncio
from typing import TypedDict, Optional
from dataclasses import dataclass, field
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser


# =============================================================================
# 配置
# =============================================================================

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-72b-instruct")


def get_llm(temperature=0.3):
    return ChatOpenAI(
        base_url=LLM_BASE_URL, model=LLM_MODEL,
        api_key="not-needed", temperature=temperature,
    )


# =============================================================================
# Agent 基类
# =============================================================================

class AgentRole(Enum):
    RESEARCHER = "researcher"
    CODER = "coder"
    REVIEWER = "reviewer"
    COORDINATOR = "coordinator"


@dataclass
class Message:
    sender: str
    receiver: str
    content: str
    msg_type: str = "text"  # text / task / result / feedback


@dataclass
class AgentContext:
    task: str = ""
    messages: list = field(default_factory=list)
    results: dict = field(default_factory=dict)
    iteration: int = 0


class BaseAgent:
    """Agent 基类"""

    def __init__(self, name: str, role: AgentRole, system_prompt: str):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.llm = get_llm()

    def process(self, context: AgentContext, message: Message) -> Message:
        """处理消息并返回响应"""
        # 构建对话历史
        history = "\n".join([
            f"[{m.sender} → {m.receiver}]: {m.content[:200]}"
            for m in context.messages[-5:]  # 最近 5 条
        ])

        prompt = ChatPromptTemplate.from_template(
            """{system_prompt}

当前任务：{task}
最近消息：
{history}

收到来自 {sender} 的消息：
{message}

请给出你的回复："""
        )

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "system_prompt": self.system_prompt,
            "task": context.task,
            "history": history or "（无历史消息）",
            "sender": message.sender,
            "message": message.content,
        })

        return Message(
            sender=self.name,
            receiver=message.sender,
            content=response,
            msg_type="result",
        )


# =============================================================================
# 专业 Agent 实现
# =============================================================================

class ResearcherAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Researcher",
            role=AgentRole.RESEARCHER,
            system_prompt="""你是一个技术研究员。你的职责是：
1. 搜索和分析技术方案
2. 总结最佳实践
3. 提供技术选型建议
回复要包含具体的技术细节和数据支持。"""
        )


class CoderAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Coder",
            role=AgentRole.CODER,
            system_prompt="""你是一个资深 Python 开发工程师。你的职责是：
1. 根据需求编写代码
2. 遵循最佳实践（类型注解、文档、错误处理）
3. 考虑性能和可维护性
回复必须包含可运行的代码。"""
        )


class ReviewerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Reviewer",
            role=AgentRole.REVIEWER,
            system_prompt="""你是一个严格的代码审核员。你的职责是：
1. 审查代码质量和正确性
2. 检查潜在的安全问题
3. 给出改进建议
审核结果必须包含：APPROVED（通过）或 NEEDS_REVISION（需修改）"""
        )


# =============================================================================
# 多 Agent 编排器
# =============================================================================

class MultiAgentOrchestrator:
    """多 Agent 编排器 — 协调多个 Agent 完成复杂任务"""

    def __init__(self):
        self.agents = {
            "Researcher": ResearcherAgent(),
            "Coder": CoderAgent(),
            "Reviewer": ReviewerAgent(),
        }
        self.coordinator_llm = get_llm(temperature=0)

    def run(self, task: str, max_rounds: int = 5) -> dict:
        """运行多 Agent 协作流程"""
        print(f"\n{'='*60}")
        print(f"多 Agent 协作系统")
        print(f"任务: {task}")
        print(f"{'='*60}")

        context = AgentContext(task=task)

        # 工作流：Research → Code → Review → (Revise if needed)
        workflow = [
            ("Coordinator", "Researcher", "请研究以下任务的技术方案：" + task),
            ("Researcher", "Coder", None),  # Researcher 的输出作为 Coder 的输入
            ("Coder", "Reviewer", None),    # Coder 的输出作为 Reviewer 的输入
        ]

        for round_num in range(max_rounds):
            print(f"\n--- Round {round_num + 1} ---")

            for step_idx, (sender, receiver, initial_msg) in enumerate(workflow):
                # 确定消息内容
                if initial_msg:
                    msg_content = initial_msg
                elif context.messages:
                    msg_content = context.messages[-1].content
                else:
                    continue

                msg = Message(
                    sender=sender, receiver=receiver,
                    content=msg_content, msg_type="task",
                )

                # Agent 处理
                if receiver in self.agents:
                    print(f"\n  [{sender} → {receiver}]")
                    response = self.agents[receiver].process(context, msg)
                    context.messages.append(msg)
                    context.messages.append(response)
                    context.results[receiver] = response.content
                    print(f"  {receiver} 回复: {response.content[:150]}...")

            # 检查 Reviewer 结果
            if "Reviewer" in context.results:
                review = context.results["Reviewer"]
                if "APPROVED" in review:
                    print(f"\n✓ 审核通过！")
                    break
                else:
                    print(f"\n⟳ 需要修改，进入下一轮...")
                    # 重置 workflow 为修改流程
                    workflow = [
                        ("Reviewer", "Coder", f"请根据审核意见修改：{review}"),
                        ("Coder", "Reviewer", None),
                    ]

            context.iteration = round_num + 1

        # 汇总结果
        print(f"\n{'='*60}")
        print(f"协作完成，共 {context.iteration + 1} 轮")
        print(f"{'='*60}")

        return {
            "task": task,
            "rounds": context.iteration + 1,
            "research": context.results.get("Researcher", ""),
            "code": context.results.get("Coder", ""),
            "review": context.results.get("Reviewer", ""),
        }


# =============================================================================
# 主程序
# =============================================================================

def main():
    orchestrator = MultiAgentOrchestrator()

    tasks = [
        "实现一个 Python 类 SemanticCache，使用向量相似度缓存 LLM 查询结果。要求：支持 TTL 过期、相似度阈值配置、线程安全。",
    ]

    for task in tasks:
        result = orchestrator.run(task)
        print(f"\n最终代码:\n{result['code'][:500]}...")


if __name__ == "__main__":
    main()
