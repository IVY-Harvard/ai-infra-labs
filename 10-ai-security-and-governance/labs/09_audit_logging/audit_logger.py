"""Async audit logger for LLM systems."""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


@dataclass
class AuditEntry:
    """Represents a single audit log entry."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    user_id: str = ""
    session_id: str = ""
    model: str = ""
    action: str = ""  # request, response, error
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    status: str = "success"
    metadata: dict = field(default_factory=dict)


class AuditLogger:
    """Async audit logger with buffered writes."""

    def __init__(self, log_dir: str = "./audit_logs", buffer_size: int = 10):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.buffer: list[AuditEntry] = []
        self.buffer_size = buffer_size
        self._lock = asyncio.Lock()

    async def log(self, entry: AuditEntry) -> None:
        """Add an entry to the buffer, flush if full."""
        async with self._lock:
            self.buffer.append(entry)
            if len(self.buffer) >= self.buffer_size:
                await self._flush()

    async def _flush(self) -> None:
        """Write buffered entries to disk."""
        if not self.buffer:
            return
        filename = self.log_dir / f"audit_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(filename, "a", encoding="utf-8") as f:
            for entry in self.buffer:
                f.write(json.dumps(asdict(entry)) + "\n")
        print(f"[AuditLogger] Flushed {len(self.buffer)} entries to {filename}")
        self.buffer.clear()

    async def close(self) -> None:
        """Flush remaining entries."""
        async with self._lock:
            await self._flush()


async def demo():
    """Demonstrate audit logging."""
    logger = AuditLogger(buffer_size=3)

    for i in range(5):
        entry = AuditEntry(
            user_id=f"user_{i % 2}",
            session_id="session_abc",
            model="gpt-4",
            action="request" if i % 2 == 0 else "response",
            prompt_tokens=100 + i * 10,
            completion_tokens=50 + i * 5,
            latency_ms=120.5 + i * 30,
            metadata={"endpoint": "/v1/chat/completions"},
        )
        await logger.log(entry)
        await asyncio.sleep(0.1)

    await logger.close()
    print("[Demo] Audit logging complete.")


if __name__ == "__main__":
    asyncio.run(demo())
