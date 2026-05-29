"""Prometheus 指标暴露"""
import time
from typing import Dict


class PrometheusMetrics:
    """
    Prometheus 指标收集器

    收集 LLM 推理的核心指标:
    - TTFT (Time To First Token)
    - TPOT (Time Per Output Token)
    - 吞吐量
    - KV Cache 利用率
    """

    def __init__(self):
        try:
            from prometheus_client import Histogram, Gauge, Counter, start_http_server
            self.ttft = Histogram("mini_engine_ttft_seconds", "Time to first token")
            self.tpot = Histogram("mini_engine_tpot_seconds", "Time per output token")
            self.e2e_latency = Histogram("mini_engine_e2e_latency_seconds", "End to end latency")
            self.tokens_generated = Counter("mini_engine_tokens_generated_total", "Total tokens generated")
            self.tokens_prompted = Counter("mini_engine_tokens_prompted_total", "Total prompt tokens")
            self.requests_completed = Counter("mini_engine_requests_completed_total", "Completed requests")
            self.kv_cache_usage = Gauge("mini_engine_kv_cache_usage_ratio", "KV cache utilization")
            self.running_requests = Gauge("mini_engine_running_requests", "Number of running requests")
            self.waiting_requests = Gauge("mini_engine_waiting_requests", "Number of waiting requests")
            self._enabled = True
        except ImportError:
            self._enabled = False
            print("[Metrics] prometheus_client not installed, metrics disabled")

    def start_server(self, port: int = 9090):
        if self._enabled:
            from prometheus_client import start_http_server
            start_http_server(port)
            print(f"[Metrics] Prometheus metrics at http://localhost:{port}/metrics")

    def record_ttft(self, seconds: float):
        if self._enabled:
            self.ttft.observe(seconds)

    def record_tpot(self, seconds: float):
        if self._enabled:
            self.tpot.observe(seconds)

    def record_request_complete(self, prompt_tokens: int, output_tokens: int, latency: float):
        if self._enabled:
            self.tokens_prompted.inc(prompt_tokens)
            self.tokens_generated.inc(output_tokens)
            self.requests_completed.inc()
            self.e2e_latency.observe(latency)

    def update_engine_stats(self, stats: Dict):
        if self._enabled:
            self.kv_cache_usage.set(stats.get("kv_cache_utilization", 0))
            self.running_requests.set(stats.get("num_running", 0))
            self.waiting_requests.set(stats.get("num_waiting", 0))
