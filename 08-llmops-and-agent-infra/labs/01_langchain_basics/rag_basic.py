"""
Lab 01: 基础 RAG 实现
完整的 RAG 流水线：文档加载 → 分块 → Embedding → 存储 → 检索 → 生成
"""
import os
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# =============================================================================
# 配置
# =============================================================================

# 使用本地模型（通过 vLLM/Ollama 提供 OpenAI 兼容接口）
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-72b-instruct")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:8001/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")

# Chroma 持久化路径
CHROMA_PERSIST_DIR = "./chroma_db"


# =============================================================================
# Step 1: 文档加载
# =============================================================================

def load_documents(doc_path: str) -> list:
    """
    加载文档
    支持：txt, pdf, 目录批量加载
    """
    path = Path(doc_path)

    if path.is_dir():
        # 目录加载：自动识别文件类型
        loader = DirectoryLoader(
            str(path),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
    elif path.suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif path.suffix == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(f"不支持的文件类型: {path.suffix}")

    documents = loader.load()
    print(f"✓ 加载了 {len(documents)} 个文档")
    return documents


# =============================================================================
# Step 2: 文本分块
# =============================================================================

def split_documents(documents: list, chunk_size: int = 500,
                    chunk_overlap: int = 50) -> list:
    """
    文本分块
    使用 RecursiveCharacterTextSplitter：在自然边界处分割
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    print(f"✓ 分块完成: {len(documents)} 个文档 → {len(chunks)} 个块")
    print(f"  平均块长度: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} 字符")
    return chunks


# =============================================================================
# Step 3: Embedding + 向量存储
# =============================================================================

def create_vector_store(chunks: list, persist_directory: str = CHROMA_PERSIST_DIR):
    """
    创建向量存储
    使用 Chroma（轻量级向量数据库，适合开发测试）
    """
    embeddings = OpenAIEmbeddings(
        base_url=EMBEDDING_BASE_URL,
        model=EMBEDDING_MODEL,
        api_key="not-needed",  # 本地模型不需要 key
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    print(f"✓ 向量存储创建完成，共 {vectorstore._collection.count()} 条记录")
    return vectorstore


def load_vector_store(persist_directory: str = CHROMA_PERSIST_DIR):
    """加载已有的向量存储"""
    embeddings = OpenAIEmbeddings(
        base_url=EMBEDDING_BASE_URL,
        model=EMBEDDING_MODEL,
        api_key="not-needed",
    )

    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )


# =============================================================================
# Step 4: 构建 RAG Chain
# =============================================================================

# 自定义 QA Prompt
QA_PROMPT_TEMPLATE = """基于以下上下文回答问题。如果上下文中没有相关信息，请明确说明"根据提供的文档无法回答该问题"。

上下文：
{context}

问题：{question}

要求：
1. 仅基于上下文中的信息回答
2. 如果不确定，说明不确定的原因
3. 引用相关段落支持你的回答

回答："""

QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def build_rag_chain(vectorstore, top_k: int = 3):
    """
    构建 RAG 问答链
    检索 top_k 个最相关的文档块，拼接为上下文
    """
    llm = ChatOpenAI(
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        api_key="not-needed",
        temperature=0.3,
        max_tokens=1000,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 将所有检索结果拼接到 prompt
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )

    return qa_chain


# =============================================================================
# Step 5: 运行 RAG
# =============================================================================

def run_rag_query(qa_chain, question: str) -> dict:
    """执行 RAG 查询并返回结果"""
    result = qa_chain.invoke({"query": question})

    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"{'='*60}")
    print(f"回答: {result['result']}")
    print(f"\n引用来源 ({len(result['source_documents'])} 个文档块):")
    for i, doc in enumerate(result["source_documents"]):
        print(f"  [{i+1}] {doc.page_content[:100]}...")
        if doc.metadata:
            print(f"      来源: {doc.metadata.get('source', 'unknown')}")

    return result


# =============================================================================
# 主程序
# =============================================================================

def main():
    """完整 RAG 流水线演示"""

    # 创建示例文档（实际使用时替换为真实文档路径）
    sample_docs_dir = Path("./sample_docs")
    sample_docs_dir.mkdir(exist_ok=True)

    # 写入示例文档
    sample_content = """
    # 公司年假政策

    ## 基本规定
    员工入职满一年后，享有带薪年假。年假天数根据工龄递增：
    - 入职满 1 年不满 5 年：5 天年假
    - 入职满 5 年不满 10 年：10 天年假
    - 入职满 10 年及以上：15 天年假

    ## 使用规则
    1. 年假须在当年 12 月 31 日前使用完毕
    2. 未休年假可折算为工资补偿，标准为日薪的 200%
    3. 连续休假超过 3 天需提前 7 个工作日审批
    4. 法定节假日不计入年假

    ## 特殊情况
    - 试用期内不享有年假
    - 病假超过 30 天的，当年年假相应减少
    - 产假/陪产假期间年假正常累计
    """

    (sample_docs_dir / "hr_policy.txt").write_text(sample_content, encoding="utf-8")

    # 执行 RAG 流水线
    print("=" * 60)
    print("RAG 流水线演示")
    print("=" * 60)

    # 1. 加载文档
    documents = load_documents(str(sample_docs_dir))

    # 2. 分块
    chunks = split_documents(documents, chunk_size=300, chunk_overlap=50)

    # 3. 创建向量存储
    vectorstore = create_vector_store(chunks)

    # 4. 构建 RAG Chain
    qa_chain = build_rag_chain(vectorstore, top_k=3)

    # 5. 查询
    questions = [
        "公司的年假政策是什么？入职3年有几天假？",
        "未使用的年假怎么处理？",
        "试用期能请年假吗？",
        "公司的加班政策是什么？",  # 文档中没有的问题
    ]

    for q in questions:
        run_rag_query(qa_chain, q)


if __name__ == "__main__":
    main()
