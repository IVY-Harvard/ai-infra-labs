"""
Lab 04: AutoGen Multi-Agent 对话演示
使用 AutoGen 框架实现多 Agent 协作
"""
import os

try:
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    print("autogen 未安装，请运行: pip install autogen-agentchat")


# =============================================================================
# 配置
# =============================================================================

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-72b-instruct")

llm_config = {
    "config_list": [{
        "model": LLM_MODEL,
        "base_url": LLM_BASE_URL,
        "api_key": "not-needed",
    }],
    "temperature": 0.3,
}


# =============================================================================
# 1. 双 Agent 对话
# =============================================================================

def demo_two_agent():
    """两个 Agent 之间的对话"""
    print("\n" + "=" * 60)
    print("AutoGen: 双 Agent 对话")
    print("=" * 60)

    if not AUTOGEN_AVAILABLE:
        print("  autogen 未安装，跳过")
        return

    # 助手 Agent
    assistant = AssistantAgent(
        name="TechAdvisor",
        system_message="""你是一个技术架构顾问，擅长 LLM 应用架构设计。
请给出专业、有深度的建议。每次回复控制在 200 字以内。""",
        llm_config=llm_config,
    )

    # 用户代理 Agent（代表用户，也可以执行代码）
    user_proxy = UserProxyAgent(
        name="Engineer",
        human_input_mode="NEVER",  # 不需要人工输入
        max_consecutive_auto_reply=3,
        code_execution_config=False,
        system_message="你是一个后端工程师，正在学习 LLM 应用开发。请提出具体问题。",
    )

    # 发起对话
    user_proxy.initiate_chat(
        assistant,
        message="我们有 8 张 H20 GPU，想搭建一个企业级 RAG 系统。应该怎么规划架构？",
    )


# =============================================================================
# 2. 多 Agent Group Chat
# =============================================================================

def demo_group_chat():
    """多 Agent 群聊"""
    print("\n" + "=" * 60)
    print("AutoGen: 多 Agent 群聊")
    print("=" * 60)

    if not AUTOGEN_AVAILABLE:
        print("  autogen 未安装，跳过")
        return

    # 定义多个角色 Agent
    architect = AssistantAgent(
        name="Architect",
        system_message="""你是系统架构师。负责：
- 整体架构设计
- 技术选型决策
- 性能和可扩展性考虑
每次回复 100 字以内，聚焦架构视角。""",
        llm_config=llm_config,
    )

    ml_engineer = AssistantAgent(
        name="MLEngineer",
        system_message="""你是机器学习工程师。负责：
- 模型选择和优化
- RAG 管道设计
- 评估方案设计
每次回复 100 字以内，聚焦 ML 视角。""",
        llm_config=llm_config,
    )

    devops = AssistantAgent(
        name="DevOps",
        system_message="""你是 DevOps 工程师。负责：
- 部署和运维方案
- 监控和告警设计
- GPU 资源管理
每次回复 100 字以内，聚焦运维视角。""",
        llm_config=llm_config,
    )

    # 用户代理
    user_proxy = UserProxyAgent(
        name="ProjectLead",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        code_execution_config=False,
        system_message="你是项目负责人，协调团队讨论。",
    )

    # 创建群聊
    group_chat = GroupChat(
        agents=[user_proxy, architect, ml_engineer, devops],
        messages=[],
        max_round=8,
        speaker_selection_method="round_robin",
    )

    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=llm_config,
    )

    # 发起群聊
    user_proxy.initiate_chat(
        manager,
        message="""团队讨论：我们需要设计一个 Agent 服务平台，包含以下需求：
1. 支持多个 Agent 同时服务
2. RAG 知识问答
3. 灰度发布和质量监控
请各位从自己的专业角度讨论方案。""",
    )


# =============================================================================
# 3. 概念演示（不依赖 AutoGen）
# =============================================================================

def demo_concept():
    """AutoGen 概念演示（纯 Python 模拟）"""
    print("\n" + "=" * 60)
    print("Multi-Agent 对话概念演示")
    print("=" * 60)

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        base_url=LLM_BASE_URL, model=LLM_MODEL,
        api_key="not-needed", temperature=0.3,
    )

    agents = {
        "Architect": "你是系统架构师，关注可扩展性和整体设计。",
        "MLEngineer": "你是 ML 工程师，关注模型性能和 RAG 质量。",
        "DevOps": "你是 DevOps 工程师，关注部署、监控和资源管理。",
    }

    topic = "设计一个基于 8 张 H20 GPU 的 Agent 服务平台"
    conversation_history = [f"讨论主题：{topic}"]

    for round_num in range(3):
        print(f"\n--- Round {round_num + 1} ---")
        for agent_name, system_prompt in agents.items():
            prompt = f"""{system_prompt}

之前的讨论：
{chr(10).join(conversation_history[-6:])}

请从你的专业角度发表看法（100字以内）："""

            response = llm.predict(prompt)
            print(f"\n[{agent_name}]: {response[:200]}...")
            conversation_history.append(f"[{agent_name}]: {response}")

    print("\n--- 讨论结束 ---")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    if AUTOGEN_AVAILABLE:
        demo_two_agent()
        demo_group_chat()
    else:
        demo_concept()
