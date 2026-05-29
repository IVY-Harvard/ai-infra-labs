"""长期记忆 - 基于向量数据库的持久化记忆"""
import os
from typing import Optional
from dataclasses import dataclass

try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    VECTORSTORE_AVAILABLE = True
except ImportError:
    VECTORSTORE_AVAILABLE = False


EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:8001/v1")


@dataclass
class MemoryEntry:
    content: str
    user_id: str
    session_id: str
    importance: float = 0.5
    metadata: dict = None


class LongTermMemory:
    """长期记忆 - 向量数据库存储"""

    def __init__(self, persist_dir: str = "./memory_store"):
        self.persist_dir = persist_dir
        if VECTORSTORE_AVAILABLE:
            self.embeddings = OpenAIEmbeddings(
                base_url=EMBEDDING_BASE_URL,
                model="bge-m3", api_key="not-needed",
            )
            self.vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings,
                collection_name="long_term_memory",
            )
        else:
            self.vectorstore = None

    def store(self, entry: MemoryEntry):
        """存储记忆"""
        if not self.vectorstore:
            return
        self.vectorstore.add_texts(
            texts=[entry.content],
            metadatas=[{
                "user_id": entry.user_id,
                "session_id": entry.session_id,
                "importance": entry.importance,
                **(entry.metadata or {}),
            }],
        )

    def recall(self, query: str, user_id: str = None,
               top_k: int = 5) -> list[dict]:
        """回忆相关记忆"""
        if not self.vectorstore:
            return []

        filter_dict = {}
        if user_id:
            filter_dict["user_id"] = user_id

        results = self.vectorstore.similarity_search_with_score(
            query, k=top_k, filter=filter_dict if filter_dict else None,
        )

        return [
            {"content": doc.page_content, "score": score, "metadata": doc.metadata}
            for doc, score in results
        ]

    def forget(self, user_id: str):
        """遗忘用户记忆（GDPR 合规）"""
        if self.vectorstore:
            # Chroma 支持按 metadata 删除
            self.vectorstore._collection.delete(where={"user_id": user_id})
