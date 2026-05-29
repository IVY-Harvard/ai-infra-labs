"""Input Guard - Detect prompt injection, sensitive content, and PII."""
import re
from typing import Any


INJECTION_PATTERNS = [
    r"ignore\s+(previous|above|all)\s+(instructions|prompts)",
    r"you\s+are\s+now\s+(a|an|DAN)",
    r"system\s*:\s*",
    r"<\|im_start\|>system",
    r"\\n\\n###\\n\\n",
]

SENSITIVE_WORDS = ["暴力", "色情", "赌博", "毒品", "自杀", "恐怖"]


class InputGuard:
    def __init__(self):
        self.injection_patterns = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

    def check(self, request_body: dict[str, Any]) -> dict:
        messages = request_body.get("messages", [])
        text = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))

        for pattern in self.injection_patterns:
            if pattern.search(text):
                return {"safe": False, "reason": "prompt_injection_detected"}

        for word in SENSITIVE_WORDS:
            if word in text:
                return {"safe": False, "reason": f"sensitive_content: {word}"}

        if self._detect_pii(text):
            return {"safe": False, "reason": "pii_detected"}

        return {"safe": True, "reason": None}

    def _detect_pii(self, text: str) -> bool:
        id_pattern = r"\b\d{17}[\dXx]\b"
        phone_pattern = r"\b1[3-9]\d{9}\b"
        return bool(re.search(id_pattern, text) or re.search(phone_pattern, text))
