"""
Lab 01: Chain 编排模式演示
展示 LangChain 中的 Chain 编排：顺序链、路由链、转换链
"""
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser


# =============================================================================
# 配置
# =============================================================================

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-72b-instruct")


def get_llm(temperature: float = 0.7):
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        api_key="not-needed",
        temperature=temperature,
    )


# =============================================================================
# 1. 顺序链 (Sequential Chain) - LCEL 方式
# =============================================================================

def demo_sequential_chain():
    """
    顺序链：多步处理管道
    示例：用户问题 → 分类 → 根据分类生成回答 → 格式化输出
    """
    print("\n" + "=" * 60)
    print("演示 1: 顺序链 (Sequential Chain)")
    print("=" * 60)

    llm = get_llm()

    # Step 1: 分类
    classify_prompt = ChatPromptTemplate.from_template(
        "将以下用户问题分类为：技术问题/业务问题/闲聊。只输出分类结果。\n问题：{question}"
    )

    # Step 2: 根据分类生成回答
    answer_prompt = ChatPromptTemplate.from_template(
        """你是一个智能助手。
问题分类：{category}
用户问题：{question}
请根据问题分类，用适当的风格回答用户问题。
技术问题请详细严谨，业务问题请专业简洁，闲聊请轻松友好。"""
    )

    # Step 3: 格式化
    format_prompt = ChatPromptTemplate.from_template(
        "将以下回答格式化为 Markdown 格式，添加适当的标题和列表：\n{answer}"
    )

    # 使用 LCEL (LangChain Expression Language) 组合
    chain = (
        # Step 1: 分类
        RunnablePassthrough.assign(
            category=classify_prompt | llm | StrOutputParser()
        )
        # Step 2: 生成回答
        | RunnablePassthrough.assign(
            answer=answer_prompt | llm | StrOutputParser()
        )
        # Step 3: 格式化
        | format_prompt | llm | StrOutputParser()
    )

    # 测试
    questions = [
        "Python 的 GIL 是什么？如何绕过它？",
        "下个季度的 OKR 怎么定比较好？",
        "周末有什么好玩的推荐吗？",
    ]

    for q in questions:
        print(f"\n问题: {q}")
        result = chain.invoke({"question": q})
        print(f"回答: {result[:200]}...")


# =============================================================================
# 2. 路由链 (Router Chain)
# =============================================================================

def demo_router_chain():
    """
    路由链：根据输入特征选择不同的处理路径
    类似微服务中的请求路由
    """
    print("\n" + "=" * 60)
    print("演示 2: 路由链 (Router Chain)")
    print("=" * 60)

    llm = get_llm()

    # 不同场景的 Prompt
    prompts = {
        "python": ChatPromptTemplate.from_template(
            "你是 Python 专家。请详细回答以下 Python 问题，包含代码示例：\n{question}"
        ),
        "database": ChatPromptTemplate.from_template(
            "你是数据库专家。请回答以下数据库问题，包含 SQL 示例：\n{question}"
        ),
        "devops": ChatPromptTemplate.from_template(
            "你是 DevOps 专家。请回答以下运维问题，包含命令行示例：\n{question}"
        ),
        "general": ChatPromptTemplate.from_template(
            "请回答以下问题：\n{question}"
        ),
    }

    # 路由分类器
    router_prompt = ChatPromptTemplate.from_template(
        """将以下问题路由到最适合的专家。只输出一个词：python/database/devops/general

问题：{question}
路由到："""
    )

    def route(info: dict) -> str:
        """路由函数：根据分类结果选择对应的 Prompt"""
        category = info["category"].strip().lower()
        if category in prompts:
            chain = prompts[category] | llm | StrOutputParser()
        else:
            chain = prompts["general"] | llm | StrOutputParser()
        return chain.invoke({"question": info["question"]})

    # 完整路由链
    router_chain = (
        RunnablePassthrough.assign(
            category=router_prompt | llm | StrOutputParser()
        )
        | RunnableLambda(route)
    )

    # 测试
    questions = [
        "如何用 Python 实现单例模式？",
        "MySQL 的索引什么时候会失效？",
        "Kubernetes Pod 一直 CrashLoopBackOff 怎么排查？",
    ]

    for q in questions:
        print(f"\n问题: {q}")
        result = router_chain.invoke({"question": q})
        print(f"回答: {result[:200]}...")


# =============================================================================
# 3. 转换链 (Transform Chain)
# =============================================================================

def demo_transform_chain():
    """
    转换链：在 LLM 调用之间插入自定义数据转换逻辑
    """
    print("\n" + "=" * 60)
    print("演示 3: 转换链 (Transform Chain)")
    print("=" * 60)

    llm = get_llm()

    # 自定义转换函数
    def extract_key_info(text: str) -> str:
        """提取关键信息（去除噪声）"""
        lines = text.strip().split("\n")
        # 过滤空行和过短的行
        meaningful_lines = [l for l in lines if len(l.strip()) > 10]
        return "\n".join(meaningful_lines[:10])  # 最多保留 10 行

    def add_metadata(result: str) -> str:
        """添加元数据"""
        word_count = len(result)
        return f"[长度: {word_count} 字]\n{result}"

    # 构建转换链
    prompt = ChatPromptTemplate.from_template(
        "请分析以下文本的核心观点：\n{text}"
    )

    chain = (
        {"text": RunnableLambda(lambda x: extract_key_info(x["raw_text"]))}
        | prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(add_metadata)
    )

    # 测试
    sample_text = """
    这是一段很长的文本。

    核心观点是：大语言模型正在改变软件开发的方式。
    传统的软件工程方法论需要适应 AI 时代的新需求。

    具体来说：
    1. 提示工程正在成为新的编程范式
    2. 评估方法需要从确定性转向概率性
    3. 工程师需要理解模型的能力边界

    这些是一些不太重要的补充信息...
    还有一些噪声数据...
    """

    result = chain.invoke({"raw_text": sample_text})
    print(f"转换链结果:\n{result}")


# =============================================================================
# 4. 并行链 (Parallel Chain)
# =============================================================================

def demo_parallel_chain():
    """
    并行链：同时执行多个独立的处理逻辑，合并结果
    """
    print("\n" + "=" * 60)
    print("演示 4: 并行链 (Parallel Chain)")
    print("=" * 60)

    llm = get_llm()

    # 三个并行的分析任务
    summary_chain = (
        ChatPromptTemplate.from_template("用一句话总结：{text}")
        | llm | StrOutputParser()
    )

    sentiment_chain = (
        ChatPromptTemplate.from_template(
            "分析以下文本的情感倾向（正面/负面/中性），只输出结果：{text}"
        )
        | llm | StrOutputParser()
    )

    keywords_chain = (
        ChatPromptTemplate.from_template(
            "提取以下文本的3个关键词，用逗号分隔：{text}"
        )
        | llm | StrOutputParser()
    )

    # 并行执行
    from langchain.schema.runnable import RunnableParallel

    parallel_chain = RunnableParallel(
        summary=summary_chain,
        sentiment=sentiment_chain,
        keywords=keywords_chain,
    )

    # 测试
    text = "LangChain 是一个强大的 LLM 应用开发框架，它简化了 RAG 和 Agent 的实现过程。"
    result = parallel_chain.invoke({"text": text})

    print(f"输入: {text}")
    print(f"摘要: {result['summary']}")
    print(f"情感: {result['sentiment']}")
    print(f"关键词: {result['keywords']}")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    demo_sequential_chain()
    demo_router_chain()
    demo_transform_chain()
    demo_parallel_chain()
