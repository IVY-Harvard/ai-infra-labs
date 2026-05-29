"""Audit report generator."""

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class ReportSummary:
    """Summary statistics for an audit report."""
    total_entries: int = 0
    total_violations: int = 0
    entries_by_action: dict = None
    entries_by_model: dict = None
    entries_by_user: dict = None
    avg_latency_ms: float = 0.0
    error_count: int = 0
    time_range: tuple = None

    def __post_init__(self):
        self.entries_by_action = self.entries_by_action or {}
        self.entries_by_model = self.entries_by_model or {}
        self.entries_by_user = self.entries_by_user or {}


class ReportGenerator:
    """Generates audit reports from log files."""

    def __init__(self, log_dir: str = "./audit_logs"):
        self.log_dir = Path(log_dir)

    def _load_entries(self) -> list[dict]:
        """Load all entries from log directory."""
        entries = []
        if not self.log_dir.exists():
            return entries
        for filepath in sorted(self.log_dir.glob("*.jsonl")):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line.strip()))
        return entries

    def generate_summary(self) -> ReportSummary:
        """Generate a summary report."""
        entries = self._load_entries()
        if not entries:
            return ReportSummary()

        actions = Counter(e.get("action", "unknown") for e in entries)
        models = Counter(e.get("model", "unknown") for e in entries)
        users = Counter(e.get("user_id", "unknown") for e in entries)
        latencies = [e.get("latency_ms", 0) for e in entries]
        errors = sum(1 for e in entries if e.get("status") == "error")
        timestamps = sorted(e.get("timestamp", "") for e in entries)

        return ReportSummary(
            total_entries=len(entries),
            entries_by_action=dict(actions),
            entries_by_model=dict(models),
            entries_by_user=dict(users),
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            error_count=errors,
            time_range=(timestamps[0], timestamps[-1]) if timestamps else None,
        )

    def generate_text_report(self) -> str:
        """Generate a human-readable text report."""
        summary = self.generate_summary()
        now = datetime.now(timezone.utc).isoformat()

        lines = [
            "=" * 60,
            "AI SYSTEM AUDIT REPORT",
            f"Generated: {now}",
            "=" * 60,
            f"\nTotal Log Entries: {summary.total_entries}",
            f"Error Count: {summary.error_count}",
            f"Average Latency: {summary.avg_latency_ms:.1f} ms",
            f"\nTime Range: {summary.time_range[0] if summary.time_range else 'N/A'}"
            f" to {summary.time_range[1] if summary.time_range else 'N/A'}",
            "\n--- Actions ---",
        ]
        for action, count in summary.entries_by_action.items():
            lines.append(f"  {action}: {count}")
        lines.append("\n--- Models ---")
        for model, count in summary.entries_by_model.items():
            lines.append(f"  {model}: {count}")
        lines.append("\n--- Users ---")
        for user, count in summary.entries_by_user.items():
            lines.append(f"  {user}: {count}")
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def export_json(self, output_path: Optional[str] = None) -> str:
        """Export report as JSON."""
        summary = self.generate_summary()
        data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_entries": summary.total_entries,
            "error_count": summary.error_count,
            "avg_latency_ms": round(summary.avg_latency_ms, 2),
            "actions": summary.entries_by_action,
            "models": summary.entries_by_model,
            "users": summary.entries_by_user,
        }
        output = json.dumps(data, indent=2)
        if output_path:
            Path(output_path).write_text(output, encoding="utf-8")
        return output


def demo():
    """Demonstrate report generation."""
    generator = ReportGenerator()
    print(generator.generate_text_report())
    print("\n--- JSON Report ---")
    print(generator.export_json())


if __name__ == "__main__":
    demo()
