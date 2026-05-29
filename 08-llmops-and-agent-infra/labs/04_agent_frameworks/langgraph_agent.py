"""
Lab 04: LangGraph Agent - 有向图编排
使用 LangGraph 构建可控的 Agent 工作流
"""
import os
import operator
from typing import TypedDict, Annotated, Literal
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool

try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("langgraph 未安装，请运行: pip install langgraph")


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
# 工具
# =============================================================================

@tool
def search_docs(query: str) -> str:
    """搜索技术文档库"""
    docs = {
        "向量": "向量数据库选型：Milvus(大规模)、Qdrant(中规模)、pgvector(小规模)",
        "部署": "LLM 部署推荐 vLLM，支持 Tensor Parallel 和 PagedAttention",
        "评估": "RAG 评估推荐 Ragas 框架，核心指标包括 Faithfulness 和 Relevancy",
    }
    for key, value in docs.items():
        if key in query:
            return value
    return "未找到相关文档"


@tool
def run_benchmark(config: str) -> str:
    """运行性能测试"""
    return f"Benchmark 结果：配置 '{config}' 下，QPS=500, P95延迟=15ms, 准确率=92%"


@tool
def generate_report(data: str) -> str:
    """生成分析报告"""
    return f"## 分析报告\n\n基于数据：{data}\n\n结论：系统性能达标，建议进入灰度发布阶段。"


# =============================================================================
# LangGraph Agent
# =============================================================================

if LANGGRAPH_AVAILABLE:

    # 定义状态
    class AgentState(TypedDict):
        messages: Annotated[list, operator.add]
        next_action: str
        iteration: int

    def create_research_agent():
        """
        创建研究型 Agent：
        搜索 → 分析 → 报告 → 审核 → (通过 | 重来)
        """
        llm = get_llm()
        tools = [search_docs, run_benchmark, generate_report]
        llm_with_tools = llm.bind_tools(tools)

        # 节点函数
        def researcher(state: AgentState) -> dict:
            """研究员：搜索和收集信息"""
            messages = state["messages"]
            system = SystemMessage(content="你是一个技术研究员，负责搜索和收集相关信息。使用 search_docs 工具查找信息。")
            response = llm_with_tools.invoke([system] + messages)
            return {"messages": [response], "iteration": state.get("iteration", 0)}

        def analyzer(state: AgentState) -> dict:
            """分析师：运行测试和分析数据"""
            messages = state["messages"]
            system = SystemMessage(content="你是一个技术分析师，负责分析数据和运行测试。使用 run_benchmark 工具进行测试。")
            response = llm_with_tools.invoke([system] + messages)
            return {"messages": [response], "iteration": state.get("iteration", 0)}

        def reporter(state: AgentState) -> dict:
            """报告员：生成分析报告"""
            messages = state["messages"]
            system = SystemMessage(content="你是一个技术写作专家，负责将分析结果整理为报告。使用 generate_report 工具。")
            response = llm_with_tools.invoke([system] + messages)
            return {"messages": [response], "iteration": state.get("iteration", 0)}

        def reviewer(state: AgentState) -> dict:
            """审核员：审核报告质量"""
            messages = state["messages"]
            system = SystemMessage(content="""你是质量审核员。审核报告并决定：
- 如果报告质量合格，回复包含 "APPROVED"
- 如果需要改进，回复包含 "REVISION_NEEDED" 并说明原因""")
            response = llm.invoke([system] + messages)
            return {
                "messages": [response],
                "iteration": state.get("iteration", 0) + 1,
            }

        # 路由函数
        def should_continue(state: AgentState) -> Literal["tool_node", "analyzer"]:
            """判断是否需要调用工具"""
            last_msg = state["messages"][-1]
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                return "tool_node"
            return "analyzer"

        def review_decision(state: AgentState) -> Literal["reporter", END]:
            """审核决策"""
            last_msg = state["messages"][-1]
            if state.get("iteration", 0) >= 3:
                return END  # 最多迭代 3 次
            if "APPROVED" in last_msg.content:
                return END
            return "reporter"  # 需要修改

        # 构建图
        tool_node = ToolNode(tools)

        graph = StateGraph(AgentState)

        # 添加节点
        graph.add_node("researcher", researcher)
        graph.add_node("tool_node", tool_node)
        graph.add_node("analyzer", analyzer)
        graph.add_node("reporter", reporter)
        graph.add_node("reviewer", reviewer)

        # 添加边
        graph.set_entry_point("researcher")
        graph.add_conditional_edges("researcher", should_continue)
        graph.add_edge("tool_node", "researcher")
        graph.add_edge("analyzer", "reporter")
        graph.add_edge("reporter", "reviewer")
        graph.add_conditional_edges("reviewer", review_decision)

        return graph.compile()

    def demo_langgraph_agent():
        """LangGraph Agent 演示"""
        print("\n" + "=" * 60)
        print("LangGraph Agent: 研究 → 分析 → 报告 → 审核")
        print("=" * 60)

        agent = create_research_agent()

        initial_state = {
            "messages": [
                HumanMessage(content="请调研向量数据库选型方案，并给出部署建议")
            ],
            "next_action": "",
            "iteration": 0,
        }

        # 运行 Agent
        print("\n开始执行...")
        result = agent.invoke(initial_state)

        print("\n--- 执行轨迹 ---")
        for msg in result["messages"]:
            role = type(msg).__name__
            content = msg.content[:150] if msg.content else "(tool call)"
            print(f"  [{role}]: {content}...")

        print(f"\n迭代次数: {result.get('iteration', 0)}")


# =============================================================================
# 简化版 LangGraph 演示（不依赖 langgraph）
# =============================================================================

def demo_graph_concept():
    """演示图编排的概念（不依赖 langgraph 库）"""
    print("\n" + "=" * 60)
    print("图编排概念演示（纯 Python 实现）")
    print("=" * 60)

    llm = get_llm()

    # 定义节点
    nodes = {
        "classify": lambda q: llm.predict(
            f"将以下问题分类为 simple/complex: {q}"
        ),
        "simple_answer": lambda q: llm.predict(
            f"简洁回答: {q}"
        ),
        "research": lambda q: search_docs.invoke(q),
        "complex_answer": lambda q, ctx: llm.predict(
            f"基于以下上下文详细回答。\n上下文: {ctx}\n问题: {q}"
        ),
    }

    # 定义边（路由逻辑）
    def route(question: str):
        classification = nodes["classify"](question)
        print(f"  分类: {classification}")

        if "simple" in classification.lower():
            answer = nodes["simple_answer"](question)
            print(f"  路径: classify → simple_answer")
        else:
            context = nodes["research"](question)
            answer = nodes["complex_answer"](question, context)
            print(f"  路径: classify → research → complex_answer")

        return answer

    # 测试
    questions = [
        "Python 是什么语言？",
        "如何选择适合的向量数据库？",
    ]

    for q in questions:
        print(f"\n问题: {q}")
        answer = route(q)
        print(f"回答: {answer[:200]}...")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    if LANGGRAPH_AVAILABLE:
        demo_langgraph_agent()
    demo_graph_concept()
