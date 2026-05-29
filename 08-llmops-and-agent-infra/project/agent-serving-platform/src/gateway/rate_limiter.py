"""限流器 - 多策略流量控制"""
import time
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class RateLimitResult:
    allowed: bool
    remaining: int
    reset_at: float
    retry_after: float = 0


class TokenBucketLimiter:
    """令牌桶限流"""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate          # 每秒填充速率
        self.capacity = capacity  # 桶容量
        self.tokens: dict[str, float] = defaultdict(lambda: float(capacity))
        self.last_refill: dict[str, float] = defaultdict(time.time)

    def allow(self, key: str, tokens: int = 1) -> RateLimitResult:
        now = time.time()
        # 补充令牌
        elapsed = now - self.last_refill[key]
        self.tokens[key] = min(
            self.capacity,
            self.tokens[key] + elapsed * self.rate
        )
        self.last_refill[key] = now

        if self.tokens[key] >= tokens:
            self.tokens[key] -= tokens
            return RateLimitResult(
                allowed=True,
                remaining=int(self.tokens[key]),
                reset_at=now + (self.capacity - self.tokens[key]) / self.rate,
            )
        else:
            wait_time = (tokens - self.tokens[key]) / self.rate
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=now + wait_time,
                retry_after=wait_time,
            )


class SlidingWindowLimiter:
    """滑动窗口限流"""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    def allow(self, key: str) -> RateLimitResult:
        now = time.time()
        window_start = now - self.window_seconds

        # 清理过期请求
        self.requests[key] = [t for t in self.requests[key] if t > window_start]

        if len(self.requests[key]) < self.max_requests:
            self.requests[key].append(now)
            return RateLimitResult(
                allowed=True,
                remaining=self.max_requests - len(self.requests[key]),
                reset_at=self.requests[key][0] + self.window_seconds if self.requests[key] else now,
            )
        else:
            earliest = self.requests[key][0]
            retry_after = earliest + self.window_seconds - now
            return RateLimitResult(
                allowed=False, remaining=0,
                reset_at=earliest + self.window_seconds,
                retry_after=max(0, retry_after),
            )


class RateLimiter:
    """
    多层限流器
    - 全局限流（保护系统总容量）
    - 用户级限流（公平使用）
    - Token 级限流（成本控制）
    """

    def __init__(self):
        self.global_limiter = TokenBucketLimiter(rate=100, capacity=500)
        self.user_limiter = SlidingWindowLimiter(max_requests=60, window_seconds=60)
        self.token_limiter = TokenBucketLimiter(rate=10000, capacity=50000)

    def check(self, user_id: str, estimated_tokens: int = 1000) -> RateLimitResult:
        """多层限流检查"""
        # Layer 1: 全局
        global_result = self.global_limiter.allow("global")
        if not global_result.allowed:
            return global_result

        # Layer 2: 用户
        user_result = self.user_limiter.allow(user_id)
        if not user_result.allowed:
            return user_result

        # Layer 3: Token 消耗
        token_result = self.token_limiter.allow(user_id, estimated_tokens)
        if not token_result.allowed:
            return token_result

        return RateLimitResult(
            allowed=True,
            remaining=min(global_result.remaining, user_result.remaining),
            reset_at=max(global_result.reset_at, user_result.reset_at),
        )
