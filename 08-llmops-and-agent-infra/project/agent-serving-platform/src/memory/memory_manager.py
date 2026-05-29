"""记忆管理器 - 协调短期和长期记忆"""
from .short_term import ShortTermMemory
from .long_term import LongTermMemory, MemoryEntry


class MemoryManager:
    """记忆管理器 - 统一管理短期和长期记忆"""

    def __init__(self):
        self.short_term = ShortTermMemory(max_turns=20)
        self.long_term = LongTermMemory()

    def add_interaction(self, session_id: str, user_id: str,
                        query: str, response: str):
        """记录一次交互"""
        # 短期记忆
        self.short_term.add(session_id, "user", query)
        self.short_term.add(session_id, "assistant", response)

        # 长期记忆（有选择性地存储）
        if self._should_store_long_term(query, response):
            self.long_term.store(MemoryEntry(
                content=f"Q: {query}\nA: {response}",
                user_id=user_id,
                session_id=session_id,
                importance=self._estimate_importance(query),
            ))

    def get_context(self, session_id: str, user_id: str,
                    query: str) -> dict:
        """获取上下文（短期 + 长期）"""
        # 短期：最近对话
        recent_history = self.short_term.get_history(session_id, max_turns=5)

        # 长期：相关记忆
        relevant_memories = self.long_term.recall(query, user_id, top_k=3)

        return {
            "recent_history": recent_history,
            "relevant_memories": relevant_memories,
        }

    def _should_store_long_term(self, query: str, response: str) -> bool:
        """判断是否值得存入长期记忆"""
        # 简单启发式：包含关键信息的交互才存储
        if len(response) > 100:
            return True
        return False

    def _estimate_importance(self, query: str) -> float:
        """估算重要性（0-1）"""
        # 简化：越长的查询通常越重要
        return min(len(query) / 200, 1.0)
