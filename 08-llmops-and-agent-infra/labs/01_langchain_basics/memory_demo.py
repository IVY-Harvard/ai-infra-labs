"""
Lab 01: 对话记忆机制演示
展示不同的 Memory 策略：Buffer、Summary、Vector Memory
"""
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
    VectorStoreRetrieverMemory,
)
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma


# =============================================================================
# 配置
# =============================================================================

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-72b-instruct")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:8001/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")


def get_llm(temperature: float = 0.7):
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        api_key="not-needed",
        temperature=temperature,
    )


# =============================================================================
# 1. Buffer Memory（全量缓冲记忆）
# =============================================================================

def demo_buffer_memory():
    """
    Buffer Memory：保存所有对话历史
    优势：完整上下文
    劣势：Token 消耗随对话增长线性增加
    适用：短对话（< 10 轮）
    """
    print("\n" + "=" * 60)
    print("演示 1: Buffer Memory (全量记忆)")
    print("=" * 60)

    llm = get_llm()
    memory = ConversationBufferMemory(return_messages=True)

    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True,  # 显示完整 prompt
    )

    # 模拟多轮对话
    messages = [
        "你好，我是张三，我在研究向量数据库",
        "Milvus 和 Qdrant 哪个更适合我们的场景？我们有 5000 万条数据",
        "我叫什么名字？我在研究什么？",  # 测试是否记住上下文
    ]

    for msg in messages:
        print(f"\n用户: {msg}")
        response = conversation.predict(input=msg)
        print(f"AI: {response}")

    # 查看 memory 内容
    print(f"\n--- Memory 内容 ---")
    print(f"消息数: {len(memory.chat_memory.messages)}")
    for msg in memory.chat_memory.messages:
        print(f"  [{msg.type}]: {msg.content[:80]}...")


# =============================================================================
# 2. Window Memory（滑动窗口记忆）
# =============================================================================

def demo_window_memory():
    """
    Window Memory：只保留最近 K 轮对话
    优势：Token 消耗固定
    劣势：会丢失早期上下文
    适用：长对话 + 上下文主要在近期
    """
    print("\n" + "=" * 60)
    print("演示 2: Window Memory (窗口记忆, k=2)")
    print("=" * 60)

    llm = get_llm()
    memory = ConversationBufferWindowMemory(k=2, return_messages=True)

    conversation = ConversationChain(llm=llm, memory=memory)

    messages = [
        "我叫李四，是后端工程师",
        "我在用 Python 和 Go 开发微服务",
        "最近在学习 LLM 应用开发",
        "我叫什么？我的技术栈是什么？",  # k=2 时可能已忘记第一轮
    ]

    for msg in messages:
        print(f"\n用户: {msg}")
        response = conversation.predict(input=msg)
        print(f"AI: {response}")

    print(f"\n--- Window Memory 内容 (最近 {memory.k} 轮) ---")
    for msg in memory.chat_memory.messages:
        print(f"  [{msg.type}]: {msg.content[:80]}...")


# =============================================================================
# 3. Summary Memory（摘要记忆）
# =============================================================================

def demo_summary_memory():
    """
    Summary Memory：用 LLM 对历史对话生成摘要
    优势：保留关键信息，Token 消耗可控
    劣势：摘要过程消耗额外 Token，可能丢失细节
    适用：长对话 + 需要保持全局上下文
    """
    print("\n" + "=" * 60)
    print("演示 3: Summary Memory (摘要记忆)")
    print("=" * 60)

    llm = get_llm()
    memory = ConversationSummaryMemory(llm=llm, return_messages=True)

    conversation = ConversationChain(llm=llm, memory=memory)

    messages = [
        "我们团队有 8 张 H20 GPU，想搭建 LLM 推理服务",
        "我们的主要场景是企业知识问答，文档量大约 100 万篇",
        "目前在考虑用 vLLM 部署 Qwen2.5-72B，你觉得资源够吗？",
        "请总结一下我们之前讨论的需求和方案",
    ]

    for msg in messages:
        print(f"\n用户: {msg}")
        response = conversation.predict(input=msg)
        print(f"AI: {response}")

    # 查看摘要
    print(f"\n--- Summary Memory 摘要 ---")
    print(memory.buffer)


# =============================================================================
# 4. Vector Memory（向量记忆）
# =============================================================================

def demo_vector_memory():
    """
    Vector Memory：将对话历史存入向量数据库，按相关性检索
    优势：能在长对话中精确找到相关历史
    劣势：可能错过线性上下文
    适用：很长的对话 + 话题跳跃频繁
    """
    print("\n" + "=" * 60)
    print("演示 4: Vector Memory (向量记忆)")
    print("=" * 60)

    embeddings = OpenAIEmbeddings(
        base_url=EMBEDDING_BASE_URL,
        model=EMBEDDING_MODEL,
        api_key="not-needed",
    )

    # 使用 Chroma 作为向量存储
    vectorstore = Chroma(
        collection_name="conversation_memory",
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    memory = VectorStoreRetrieverMemory(retriever=retriever)

    # 预存一些对话历史
    memory.save_context(
        {"input": "我们的项目用 FastAPI 框架"},
        {"output": "FastAPI 是个很好的选择，性能高且支持异步。"}
    )
    memory.save_context(
        {"input": "数据库用的是 PostgreSQL + Redis"},
        {"output": "PG 做主存储，Redis 做缓存，经典组合。"}
    )
    memory.save_context(
        {"input": "部署在 Kubernetes 上，用 Helm 管理"},
        {"output": "K8s + Helm 是标准化的部署方案。"}
    )
    memory.save_context(
        {"input": "最近在集成 Milvus 做向量检索"},
        {"output": "Milvus 适合大规模向量检索，和 PG 可以互补。"}
    )

    # 测试：查询应该召回相关的历史
    llm = get_llm()

    # 自定义 prompt，包含向量记忆
    template = """以下是与当前问题相关的历史对话：
{history}

当前问题：{input}
AI:"""

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template,
    )

    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
    )

    # 测试检索
    queries = [
        "我们的 API 框架是什么？",       # 应该召回 FastAPI 相关
        "向量数据库的集成进展如何？",    # 应该召回 Milvus 相关
    ]

    for q in queries:
        print(f"\n用户: {q}")
        # 查看检索到的记忆
        relevant_docs = retriever.invoke(q)
        print(f"  召回的记忆:")
        for doc in relevant_docs:
            print(f"    - {doc.page_content[:60]}...")

        response = conversation.predict(input=q)
        print(f"AI: {response}")


# =============================================================================
# 5. 记忆策略对比
# =============================================================================

def compare_memory_strategies():
    """记忆策略选型指南"""
    print("\n" + "=" * 60)
    print("Memory 策略选型指南")
    print("=" * 60)

    comparison = """
    ┌────────────────┬────────────────┬───────────────┬─────────────────┐
    │  策略          │  Token 消耗    │  信息保留     │  适用场景       │
    ├────────────────┼────────────────┼───────────────┼─────────────────┤
    │ Buffer         │ O(n), 线性增长 │ 完整          │ 短对话(<10轮)  │
    │ Window (k)     │ O(k), 固定    │ 最近 k 轮     │ 实时对话       │
    │ Summary        │ O(1), 近似固定│ 摘要(可能丢失)│ 长对话         │
    │ Vector         │ O(k), 检索k条 │ 相关历史      │ 超长+跳跃话题  │
    │ Buffer+Summary │ O(k+1)        │ 近期完整+远期摘要│ 生产推荐    │
    └────────────────┴────────────────┴───────────────┴─────────────────┘

    生产最佳实践（混合策略）：
    ├── 最近 3 轮：完整保留（Buffer Window）
    ├── 3-20 轮：摘要（Summary）
    ├── 20 轮以上：向量检索（Vector）
    └── 关键信息：结构化存储（用户名/偏好/上下文）
    """
    print(comparison)


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    demo_buffer_memory()
    demo_window_memory()
    demo_summary_memory()
    demo_vector_memory()
    compare_memory_strategies()
