"""Quota Manager - Per-user token quota tracking."""
import time
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class UserQuota:
    daily_limit: int = 100000
    used_today: int = 0
    reset_date: str = ""


class QuotaManager:
    def __init__(self, default_daily: int = 100000):
        self.default_daily = default_daily
        self._quotas: dict[str, UserQuota] = defaultdict(
            lambda: UserQuota(daily_limit=default_daily)
        )

    def consume(self, user: str, estimated_tokens: int) -> bool:
        quota = self._quotas[user]
        today = time.strftime("%Y-%m-%d")
        if quota.reset_date != today:
            quota.used_today = 0
            quota.reset_date = today
        if quota.used_today + estimated_tokens > quota.daily_limit:
            return False
        quota.used_today += estimated_tokens
        return True

    def get_usage(self, user: str) -> dict:
        quota = self._quotas[user]
        return {"used": quota.used_today, "limit": quota.daily_limit,
                "remaining": quota.daily_limit - quota.used_today}

    def set_limit(self, user: str, daily_limit: int):
        self._quotas[user].daily_limit = daily_limit
