"""Token Bucket Rate Limiter."""
import time
from collections import defaultdict


class RateLimiter:
    def __init__(self, rate: float = 10.0, capacity: float = 100.0):
        self.rate = rate
        self.capacity = capacity
        self._buckets: dict[str, dict] = defaultdict(
            lambda: {"tokens": capacity, "last": time.monotonic()}
        )

    def allow(self, key: str, cost: float = 1.0) -> bool:
        bucket = self._buckets[key]
        now = time.monotonic()
        elapsed = now - bucket["last"]
        bucket["tokens"] = min(self.capacity, bucket["tokens"] + elapsed * self.rate)
        bucket["last"] = now

        if bucket["tokens"] >= cost:
            bucket["tokens"] -= cost
            return True
        return False

    def remaining(self, key: str) -> float:
        bucket = self._buckets[key]
        now = time.monotonic()
        elapsed = now - bucket["last"]
        return min(self.capacity, bucket["tokens"] + elapsed * self.rate)
