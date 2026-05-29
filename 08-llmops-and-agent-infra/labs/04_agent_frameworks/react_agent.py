"""
Lab 04: ReAct Agent 实现
ReAct = Reasoning + Acting：交替推理和行动
"""
import os
import json
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool, tool
from langchain.prompts import PromptTemplate
from langchain_community.utilities import SerpAPIWrapper


# =============================================================================
# 配置
# =============================================================================

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-72b-instruct")


def get_llm(temperature: float = 0.0):
    return ChatOpenAI(
        base_url=LLM_BASE_URL, model=LLM_MODEL,
        api_key="not-needed", temperature=temperature,
    )


# =============================================================================
# 自定义工具
# =============================================================================

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。输入应该是一个合法的 Python 数学表达式。"""
    try:
        # 安全限制：只允许数学运算
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含不允许的字符"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


@tool
def get_company_info(company_name: str) -> str:
    """查询公司信息。输入公司名称，返回公司基本信息。"""
    # 模拟数据库查询
    companies = {
        "Anthropic": {
            "founded": 2021,
            "ceo": "Dario Amodei",
            "products": ["Claude"],
            "employees": "~1000",
        },
        "OpenAI": {
            "founded": 2015,
            "ceo": "Sam Altman",
            "products": ["GPT-4", "DALL-E", "Whisper"],
            "employees": "~2000",
        },
    }
    info = companies.get(company_name)
    if info:
        return json.dumps(info, ensure_ascii=False)
    return f"未找到 {company_name} 的信息"


@tool
def search_knowledge_base(query: str) -> str:
    """搜索内部知识库。输入查询关键词，返回相关文档。"""
    # 模拟 RAG 检索
    knowledge = {
        "年假": "员工入职满1年享有5天年假，满5年10天，满10年15天。",
        "报销": "差旅报销需在7个工作日内提交，需附发票原件。",
        "GPU": "公司有8张H20 GPU，由AI平台组统一管理。申请使用需填写GPU申请单。",
    }
    for key, value in knowledge.items():
        if key in query:
            return value
    return "未找到相关信息，请换个关键词重试。"


# =============================================================================
# 方式 1: 手动实现 ReAct 循环
# =============================================================================

def manual_react_agent():
    """手动实现 ReAct Agent —— 理解底层原理"""
    print("\n" + "=" * 60)
    print("手动实现 ReAct Agent")
    print("=" * 60)

    llm = get_llm()
    tools = {
        "calculator": calculator,
        "get_company_info": get_company_info,
        "search_knowledge_base": search_knowledge_base,
    }

    tools_description = "\n".join([
        f"- {name}: {t.description}" for name, t in tools.items()
    ])

    react_prompt = f"""Answer the following question by reasoning step by step.

Available tools:
{tools_description}

Use the following format:

Question: the question to answer
Thought: reasoning about what to do next
Action: tool_name
Action Input: the input to the tool
Observation: the result of the tool
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the answer to the question

Question: {{question}}
"""

    question = "公司有多少张GPU？如果每张GPU每天成本是100元，一个月（30天）的总成本是多少？"

    # ReAct 循环
    prompt = react_prompt.format(question=question)
    max_steps = 5

    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")
        response = llm.predict(prompt)
        print(response)

        # 解析 Action
        if "Final Answer:" in response:
            final = response.split("Final Answer:")[-1].strip()
            print(f"\n最终答案: {final}")
            break

        if "Action:" in response and "Action Input:" in response:
            action = response.split("Action:")[-1].split("\n")[0].strip()
            action_input = response.split("Action Input:")[-1].split("\n")[0].strip()

            # 执行工具
            if action in tools:
                observation = tools[action].invoke(action_input)
                print(f"Observation: {observation}")
                prompt += response + f"\nObservation: {observation}\n"
            else:
                prompt += response + f"\nObservation: Unknown tool: {action}\n"
        else:
            break


# =============================================================================
# 方式 2: LangChain ReAct Agent
# =============================================================================

def langchain_react_agent():
    """使用 LangChain 的 ReAct Agent"""
    print("\n" + "=" * 60)
    print("LangChain ReAct Agent")
    print("=" * 60)

    llm = get_llm()
    tools = [calculator, get_company_info, search_knowledge_base]

    # LangChain 的 ReAct Prompt
    react_prompt = PromptTemplate.from_template(
        """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
    )

    agent = create_react_agent(llm, tools, react_prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
    )

    # 测试
    questions = [
        "Anthropic 是什么公司？他们的产品是什么？",
        "如果一个项目需要4张GPU，每张每天100元，运行2周需要多少钱？",
    ]

    for q in questions:
        print(f"\n问题: {q}")
        result = agent_executor.invoke({"input": q})
        print(f"答案: {result['output']}")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    manual_react_agent()
    langchain_react_agent()
