"""Audit Reporter - Generate compliance reports from audit logs."""
import json
from collections import Counter
from pathlib import Path


class AuditReporter:
    def __init__(self, log_dir: str = "./audit_logs"):
        self.log_dir = Path(log_dir)

    def daily_report(self, date: str) -> dict:
        log_file = self.log_dir / f"{date}.jsonl"
        if not log_file.exists():
            return {"date": date, "total_requests": 0}

        entries = [json.loads(line) for line in open(log_file)]
        blocked = [e for e in entries if e["blocked"]]
        block_reasons = Counter(e["reason"] for e in blocked)
        user_usage = Counter(e["user"] for e in entries)

        return {
            "date": date,
            "total_requests": len(entries),
            "blocked_requests": len(blocked),
            "block_rate": len(blocked) / max(len(entries), 1),
            "block_reasons": dict(block_reasons),
            "top_users": dict(user_usage.most_common(10)),
            "total_tokens": sum(e.get("request_tokens", 0) + e.get("response_tokens", 0) for e in entries),
        }

    def compliance_summary(self, days: int = 7) -> dict:
        import time
        reports = []
        for i in range(days):
            t = time.time() - i * 86400
            date = time.strftime("%Y-%m-%d", time.gmtime(t))
            reports.append(self.daily_report(date))
        total = sum(r["total_requests"] for r in reports)
        blocked = sum(r["blocked_requests"] for r in reports)
        return {
            "period_days": days, "total_requests": total,
            "blocked_requests": blocked,
            "overall_block_rate": blocked / max(total, 1),
        }
