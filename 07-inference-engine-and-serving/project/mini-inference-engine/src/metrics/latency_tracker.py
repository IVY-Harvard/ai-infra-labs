"""TTFT/TPOT 延迟追踪"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import statistics


@dataclass
class RequestLatency:
    """单个请求的延迟记录"""
    request_id: str
    arrival_time: float
    first_token_time: Optional[float] = None
    completion_time: Optional[float] = None
    token_times: List[float] = field(default_factory=list)

    @property
    def ttft(self) -> Optional[float]:
        if self.first_token_time:
            return self.first_token_time - self.arrival_time
        return None

    @property
    def tpot(self) -> Optional[float]:
        if len(self.token_times) >= 2:
            intervals = [self.token_times[i] - self.token_times[i-1]
                        for i in range(1, len(self.token_times))]
            return statistics.mean(intervals)
        return None

    @property
    def e2e_latency(self) -> Optional[float]:
        if self.completion_time:
            return self.completion_time - self.arrival_time
        return None


class LatencyTracker:
    """延迟追踪器"""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.active: Dict[str, RequestLatency] = {}
        self.completed: List[RequestLatency] = []

    def start_request(self, request_id: str):
        self.active[request_id] = RequestLatency(
            request_id=request_id,
            arrival_time=time.time(),
        )

    def record_first_token(self, request_id: str):
        if request_id in self.active:
            self.active[request_id].first_token_time = time.time()
            self.active[request_id].token_times.append(time.time())

    def record_token(self, request_id: str):
        if request_id in self.active:
            self.active[request_id].token_times.append(time.time())

    def complete_request(self, request_id: str):
        if request_id in self.active:
            record = self.active.pop(request_id)
            record.completion_time = time.time()
            self.completed.append(record)
            if len(self.completed) > self.window_size:
                self.completed = self.completed[-self.window_size:]

    def get_stats(self) -> Dict:
        if not self.completed:
            return {}

        ttfts = [r.ttft for r in self.completed if r.ttft is not None]
        tpots = [r.tpot for r in self.completed if r.tpot is not None]
        e2es = [r.e2e_latency for r in self.completed if r.e2e_latency is not None]

        stats = {"num_completed": len(self.completed)}
        if ttfts:
            ttfts.sort()
            stats["ttft_avg_ms"] = statistics.mean(ttfts) * 1000
            stats["ttft_p50_ms"] = ttfts[len(ttfts)//2] * 1000
            stats["ttft_p99_ms"] = ttfts[int(len(ttfts)*0.99)] * 1000
        if tpots:
            tpots.sort()
            stats["tpot_avg_ms"] = statistics.mean(tpots) * 1000
            stats["tpot_p50_ms"] = tpots[len(tpots)//2] * 1000
        if e2es:
            stats["e2e_avg_s"] = statistics.mean(e2es)

        return stats
