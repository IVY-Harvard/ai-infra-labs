"""推理服务指标采集器 — 从 vLLM /metrics 端点采集"""

import time
import logging
from typing import Dict, Optional
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class InferenceMetrics:
    """推理服务指标快照"""
    instance: str
    throughput_tps: float = 0
    prompt_throughput_tps: float = 0
    ttft_p50_ms: float = 0
    ttft_p99_ms: float = 0
    tpot_p50_ms: float = 0
    tpot_p99_ms: float = 0
    kv_cache_usage: float = 0
    prefix_cache_hit_rate: float = 0
    requests_running: int = 0
    requests_waiting: int = 0
    requests_swapped: int = 0
    preemptions_total: int = 0
    request_success_total: int = 0
    request_failure_total: int = 0
    timestamp: float = 0


class InferenceCollector:
    """vLLM 推理指标采集器

    通过 Prometheus API 或直接从 vLLM /metrics 采集:
    - 吞吐量: generation_tokens_total, prompt_tokens_total
    - 延迟: time_to_first_token, time_per_output_token
    - 调度: running/waiting/swapped 请求数
    - KV Cache: gpu_cache_usage_perc, prefix_cache_hit_rate
    """

    def __init__(self, prometheus_url: str = "http://prometheus:9090"):
        self.prometheus_url = prometheus_url

    async def collect(self, instance: str = "") -> InferenceMetrics:
        """从 Prometheus 查询推理指标"""
        try:
            metrics = InferenceMetrics(instance=instance, timestamp=time.time())

            async with aiohttp.ClientSession() as session:
                # 吞吐
                metrics.throughput_tps = await self._query_scalar(
                    session, 'sum(rate(vllm:generation_tokens_total[5m]))'
                )
                # KV Cache
                metrics.kv_cache_usage = await self._query_scalar(
                    session, 'avg(vllm:gpu_cache_usage_perc)'
                )
                # 排队
                metrics.requests_waiting = int(await self._query_scalar(
                    session, 'sum(vllm:num_requests_waiting)'
                ))
                metrics.requests_running = int(await self._query_scalar(
                    session, 'sum(vllm:num_requests_running)'
                ))

            return metrics
        except Exception as e:
            logger.error(f"Failed to collect inference metrics: {e}")
            return InferenceMetrics(instance=instance, timestamp=time.time())

    async def _query_scalar(self, session: aiohttp.ClientSession, query: str) -> float:
        """查询 Prometheus 标量值"""
        url = f"{self.prometheus_url}/api/v1/query"
        try:
            async with session.get(url, params={"query": query}) as resp:
                data = await resp.json()
                if data["status"] == "success" and data["data"]["result"]:
                    return float(data["data"]["result"][0]["value"][1])
        except Exception:
            pass
        return 0.0

    def collect_mock(self) -> InferenceMetrics:
        """模拟数据"""
        import random
        return InferenceMetrics(
            instance="vllm-0",
            throughput_tps=1500 + random.uniform(-200, 200),
            kv_cache_usage=0.65 + random.uniform(-0.1, 0.2),
            requests_running=random.randint(20, 80),
            requests_waiting=random.randint(0, 10),
            ttft_p99_ms=random.uniform(500, 2000),
            tpot_p99_ms=random.uniform(20, 50),
            timestamp=time.time(),
        )
