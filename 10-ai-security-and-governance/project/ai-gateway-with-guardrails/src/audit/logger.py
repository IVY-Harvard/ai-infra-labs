"""Audit Logger - Log all LLM interactions for compliance."""
import json
import time
from pathlib import Path
from typing import Any, Optional


class AuditLogger:
    def __init__(self, log_dir: str = "./audit_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, user: str, request: dict, response: Optional[dict],
            blocked: bool = False, reason: str = None):
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "user": user,
            "model": request.get("model", "unknown"),
            "blocked": blocked,
            "reason": reason,
            "request_tokens": self._estimate_tokens(request),
            "response_tokens": self._estimate_tokens(response) if response else 0,
        }
        log_file = self.log_dir / f"{time.strftime('%Y-%m-%d')}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def query(self, user: str = None, date: str = None) -> list[dict]:
        date = date or time.strftime("%Y-%m-%d")
        log_file = self.log_dir / f"{date}.jsonl"
        if not log_file.exists():
            return []
        entries = []
        with open(log_file) as f:
            for line in f:
                entry = json.loads(line)
                if user is None or entry["user"] == user:
                    entries.append(entry)
        return entries

    def _estimate_tokens(self, data: Any) -> int:
        if not data:
            return 0
        return len(json.dumps(data, ensure_ascii=False)) // 4
