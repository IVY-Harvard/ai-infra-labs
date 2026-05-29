"""Output Guard - Detect toxic content, hallucination markers, and sensitive info in responses."""
import re
from typing import Any


TOXIC_PATTERNS = [
    r"(kill|murder|harm)\s+(yourself|others|people)",
    r"how\s+to\s+(make|build)\s+(bomb|weapon|drug)",
]

SENSITIVE_INFO_PATTERNS = [
    r"\b\d{17}[\dXx]\b",  # ID card
    r"\b1[3-9]\d{9}\b",   # Phone
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
]


class OutputGuard:
    def __init__(self):
        self.toxic_patterns = [re.compile(p, re.IGNORECASE) for p in TOXIC_PATTERNS]
        self.sensitive_patterns = [re.compile(p) for p in SENSITIVE_INFO_PATTERNS]

    def check(self, response_data: dict[str, Any]) -> dict:
        content = self._extract_content(response_data)

        for pattern in self.toxic_patterns:
            if pattern.search(content):
                return {"safe": False, "reason": "toxic_content"}

        for pattern in self.sensitive_patterns:
            if pattern.search(content):
                return {"safe": False, "reason": "sensitive_info_leak"}

        return {"safe": True, "reason": None}

    def _extract_content(self, response: dict) -> str:
        choices = response.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return message.get("content", "")
