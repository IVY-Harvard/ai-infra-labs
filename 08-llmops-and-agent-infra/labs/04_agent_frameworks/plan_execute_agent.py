"""
Lab 04: Plan-and-Execute Agent
先制定计划，再逐步执行，支持动态调整
"""
import os
from typing import TypedDict, Annotated, Sequence
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.tools import tool


# =============================================================================
# 配置
# =============================================================================

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-72b-instruct")


def get_llm(temperature=0.0):
    return ChatOpenAI(
        base_url=LLM_BASE_URL, model=LLM_MODEL,
        api_key="not-needed", temperature=temperature,
    )


# =============================================================================
# 工具定义
# =============================================================================

@tool
def search_web(query: str) -> str:
    """搜索互联网获取信息"""
    # 模拟搜索结果
    mock_results = {
        "Milvus": "Milvus 是 Zilliz 开源的云原生向量数据库，支持万亿级向量检索。最新版本 2.4 引入了 GPU 索引。",
        "Qdrant": "Qdrant 是 Rust 实现的高性能向量数据库，强调过滤性能和易用性。",
        "性能": "在 ANN-Benchmarks 上，Milvus HNSW 和 Qdrant 的 recall@10 都超过 95%。",
    }
    for key, result in mock_results.items():
        if key.lower() in query.lower():
            return result
    return f"搜索 '{query}' 未找到相关结果"


@tool
def analyze_data(description: str) -> str:
    """分析数据并生成报告"""
    return f"数据分析完成。关键发现：根据 '{description}' 的分析，两个系统各有优劣。"


@tool
def write_report(content: str) -> str:
    """编写结构化报告"""
    return f"报告已生成：\n{content}\n[报告格式化完成]"


# =============================================================================
# Plan-and-Execute 实现
# =============================================================================

class PlanAndExecuteAgent:
    """
    Plan-and-Execute Agent
    1. Planner: 将任务分解为步骤列表
    2. Executor: 逐步执行每个步骤
    3. Replanner: 根据执行结果调整后续计划
    """

    def __init__(self):
        self.llm = get_llm()
        self.tools = {
            "search_web": search_web,
            "analyze_data": analyze_data,
            "write_report": write_report,
        }

        self.planner_prompt = ChatPromptTemplate.from_template(
            """你是一个任务规划专家。请将以下任务分解为清晰的执行步骤。

任务：{task}

请以编号列表的形式输出步骤（每步一行）：
1. ...
2. ...
...

注意：
- 每个步骤应该是可独立执行的
- 步骤之间有明确的顺序依赖
- 最后一步应该是总结/输出结果
"""
        )

        self.executor_prompt = ChatPromptTemplate.from_template(
            """你是一个执行专家。请执行以下步骤。

可用工具：
- search_web: 搜索互联网获取信息
- analyze_data: 分析数据并生成报告
- write_report: 编写结构化报告

当前任务：{task}
当前步骤：{step}
之前步骤的结果：
{previous_results}

请决定：
1. 是否需要调用工具？如果需要，输出：TOOL: tool_name | INPUT: tool_input
2. 还是可以直接给出结果？如果是，输出：RESULT: your_result
"""
        )

        self.replanner_prompt = ChatPromptTemplate.from_template(
            """你是一个计划调整专家。根据已完成的步骤和结果，判断是否需要调整剩余计划。

原始任务：{task}
原始计划：{original_plan}
已完成步骤及结果：{completed_steps}
剩余步骤：{remaining_steps}

请判断：
1. 剩余步骤是否仍然合适？
2. 是否需要添加新步骤？
3. 是否可以跳过某些步骤？

输出调整后的剩余步骤列表（如果不需要调整，原样输出）：
"""
        )

    def plan(self, task: str) -> list[str]:
        """制定计划"""
        chain = self.planner_prompt | self.llm | StrOutputParser()
        plan_text = chain.invoke({"task": task})

        # 解析步骤
        steps = []
        for line in plan_text.strip().split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                # 移除编号
                step = line.split(".", 1)[-1].strip() if "." in line else line
                steps.append(step)

        return steps

    def execute_step(self, task: str, step: str,
                     previous_results: list[dict]) -> str:
        """执行单个步骤"""
        prev_text = "\n".join([
            f"  Step: {r['step']}\n  Result: {r['result']}"
            for r in previous_results
        ]) or "（无）"

        chain = self.executor_prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "task": task,
            "step": step,
            "previous_results": prev_text,
        })

        # 解析是否需要调用工具
        if "TOOL:" in response:
            tool_line = response.split("TOOL:")[-1].strip()
            parts = tool_line.split("|")
            tool_name = parts[0].strip()
            tool_input = parts[1].replace("INPUT:", "").strip() if len(parts) > 1 else ""

            if tool_name in self.tools:
                result = self.tools[tool_name].invoke(tool_input)
                return f"[工具 {tool_name}] {result}"
            else:
                return f"未知工具: {tool_name}"
        elif "RESULT:" in response:
            return response.split("RESULT:")[-1].strip()
        else:
            return response

    def replan(self, task, original_plan, completed, remaining):
        """动态调整计划"""
        chain = self.replanner_prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "task": task,
            "original_plan": "\n".join(f"{i+1}. {s}" for i, s in enumerate(original_plan)),
            "completed_steps": "\n".join(
                f"  {r['step']} → {r['result'][:100]}" for r in completed
            ),
            "remaining_steps": "\n".join(f"- {s}" for s in remaining),
        })
        # 简化处理：返回原始剩余步骤
        return remaining

    def run(self, task: str, max_steps: int = 10) -> str:
        """运行完整的 Plan-and-Execute 流程"""
        print(f"\n{'='*60}")
        print(f"任务: {task}")
        print(f"{'='*60}")

        # Phase 1: 计划
        print("\n[Phase 1: Planning]")
        steps = self.plan(task)
        for i, step in enumerate(steps):
            print(f"  {i+1}. {step}")

        # Phase 2: 执行
        print("\n[Phase 2: Execution]")
        completed = []
        remaining = list(steps)

        for i in range(min(len(steps), max_steps)):
            if not remaining:
                break

            current_step = remaining.pop(0)
            print(f"\n  --- 执行步骤 {i+1}: {current_step} ---")

            result = self.execute_step(task, current_step, completed)
            print(f"  结果: {result[:200]}...")

            completed.append({"step": current_step, "result": result})

            # Phase 3: Replan (每执行一步后检查)
            if remaining and i < len(steps) - 1:
                remaining = self.replan(task, steps, completed, remaining)

        # 最终结果
        final_result = completed[-1]["result"] if completed else "无结果"
        print(f"\n[最终结果]")
        print(final_result)

        return final_result


# =============================================================================
# 主程序
# =============================================================================

def main():
    agent = PlanAndExecuteAgent()

    tasks = [
        "比较 Milvus 和 Qdrant 向量数据库，写一份技术选型报告",
        "评估在 8 张 H20 GPU 上部署 72B 模型的可行性",
    ]

    for task in tasks:
        agent.run(task)


if __name__ == "__main__":
    main()
