"""短期记忆 - 会话级别的对话历史"""
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Message:
    role: str  # user / assistant / system
    content: str
    timestamp: str
    metadata: dict = None


class ShortTermMemory:
    """短期记忆 - 滑动窗口对话历史"""

    def __init__(self, max_turns: int = 20, max_tokens: int = 8000):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.sessions: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_turns * 2)
        )

    def add(self, session_id: str, role: str, content: str, metadata: dict = None):
        self.sessions[session_id].append(Message(
            role=role, content=content,
            timestamp=datetime.now().isoformat(),
            metadata=metadata,
        ))

    def get_history(self, session_id: str, max_turns: int = None) -> list[dict]:
        """获取对话历史"""
        messages = list(self.sessions.get(session_id, []))
        if max_turns:
            messages = messages[-(max_turns * 2):]

        # Token 限制（简化估算）
        result = []
        total_chars = 0
        for msg in reversed(messages):
            total_chars += len(msg.content)
            if total_chars > self.max_tokens:
                break
            result.insert(0, {"role": msg.role, "content": msg.content})

        return result

    def clear(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

    def get_session_count(self) -> int:
        return len(self.sessions)
