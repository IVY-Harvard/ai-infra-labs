"""
Lab 05: Supervisor Pattern（监督者模式）
一个 Supervisor Agent 调度多个 Worker Agent
"""
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser


LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-72b-instruct")


def get_llm(temperature=0.0):
    return ChatOpenAI(
        base_url=LLM_BASE_URL, model=LLM_MODEL,
        api_key="not-needed", temperature=temperature,
    )


class WorkerAgent:
    """Worker Agent - 专注某个领域的执行者"""

    def __init__(self, name: str, expertise: str, system_prompt: str):
        self.name = name
        self.expertise = expertise
        self.system_prompt = system_prompt
        self.llm = get_llm(temperature=0.3)

    def execute(self, task: str, context: str = "") -> str:
        prompt = ChatPromptTemplate.from_template(
            """{system_prompt}

上下文信息：{context}

请完成以下任务：
{task}

请给出详细的回答："""
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "system_prompt": self.system_prompt,
            "context": context or "无",
            "task": task,
        })


class SupervisorAgent:
    """
    Supervisor Agent - 调度中心
    职责：
    1. 分析用户请求
    2. 决定派发给哪个 Worker
    3. 汇总 Worker 结果
    4. 决定是否需要更多工作
    """

    def __init__(self, workers: list[WorkerAgent]):
        self.workers = {w.name: w for w in workers}
        self.llm = get_llm(temperature=0)

        self.routing_prompt = ChatPromptTemplate.from_template(
            """你是一个任务调度器。根据用户请求，决定将任务分配给哪个专家。

可用专家：
{worker_descriptions}

用户请求：{request}
已完成的工作：{completed_work}

请决定下一步：
1. 如果还需要分配任务，回复格式：ASSIGN: <expert_name> | TASK: <specific_task>
2. 如果所有工作已完成，回复格式：DONE: <summary>

你的决定："""
        )

    def run(self, request: str, max_steps: int = 5) -> str:
        """运行 Supervisor 流程"""
        print(f"\n{'='*60}")
        print(f"[Supervisor] 收到请求: {request}")
        print(f"{'='*60}")

        worker_desc = "\n".join([
            f"- {name}: {w.expertise}" for name, w in self.workers.items()
        ])

        completed_work = []

        for step in range(max_steps):
            # Supervisor 决策
            completed_text = "\n".join(completed_work) or "无"
            decision = (self.routing_prompt | self.llm | StrOutputParser()).invoke({
                "worker_descriptions": worker_desc,
                "request": request,
                "completed_work": completed_text,
            })

            print(f"\n  [Step {step+1}] Supervisor 决策: {decision[:100]}...")

            if "DONE:" in decision:
                summary = decision.split("DONE:")[-1].strip()
                print(f"\n[Supervisor] 任务完成: {summary[:200]}...")
                return summary

            if "ASSIGN:" in decision:
                # 解析分配
                parts = decision.split("ASSIGN:")[-1]
                if "|" in parts and "TASK:" in parts:
                    worker_name = parts.split("|")[0].strip()
                    task = parts.split("TASK:")[-1].strip()

                    if worker_name in self.workers:
                        print(f"  → 分配给 [{worker_name}]: {task[:80]}...")
                        result = self.workers[worker_name].execute(
                            task, context="\n".join(completed_work[-3:])
                        )
                        completed_work.append(
                            f"[{worker_name}] 任务: {task}\n结果: {result[:300]}"
                        )
                        print(f"  ← [{worker_name}] 完成: {result[:100]}...")
                    else:
                        print(f"  ✗ 未知专家: {worker_name}")

        return "\n".join(completed_work)


def main():
    # 创建 Worker 团队
    workers = [
        WorkerAgent(
            name="DataAnalyst",
            expertise="数据分析、性能评估、指标设计",
            system_prompt="你是数据分析专家，擅长分析技术指标和性能数据。",
        ),
        WorkerAgent(
            name="Architect",
            expertise="系统架构设计、技术选型、扩展性规划",
            system_prompt="你是系统架构师，擅长设计高可用、可扩展的系统。",
        ),
        WorkerAgent(
            name="SecurityExpert",
            expertise="安全审计、风险评估、合规检查",
            system_prompt="你是安全专家，擅长识别安全风险并提出缓解方案。",
        ),
    ]

    # 创建 Supervisor
    supervisor = SupervisorAgent(workers)

    # 执行任务
    request = "请帮我评估在生产环境部署一个 RAG 系统的方案，需要考虑性能、架构和安全三个方面。"
    result = supervisor.run(request)
    print(f"\n最终结果:\n{result[:500]}")


if __name__ == "__main__":
    main()
