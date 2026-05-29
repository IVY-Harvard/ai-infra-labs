"""在线评估器 - 采样评估生产流量"""
import random
import time
import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class EvalSample:
    request_id: str
    query: str
    response: str
    context: list[str]
    scores: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class OnlineEvaluator:
    """在线评估器 - 异步采样评估"""

    def __init__(self, sample_rate: float = 0.1, buffer_size: int = 1000):
        self.sample_rate = sample_rate
        self.buffer: deque[EvalSample] = deque(maxlen=buffer_size)
        self.metrics_history: deque[dict] = deque(maxlen=10000)
        self.alert_callbacks = []

    def should_evaluate(self) -> bool:
        return random.random() < self.sample_rate

    async def evaluate_async(self, request_id: str, query: str,
                              response: str, context: list[str] = None):
        """异步评估（不阻塞主请求）"""
        sample = EvalSample(
            request_id=request_id, query=query,
            response=response, context=context or [],
        )

        # 计算质量分数
        scores = self._compute_scores(sample)
        sample.scores = scores

        self.buffer.append(sample)
        self.metrics_history.append({
            "timestamp": time.time(),
            "scores": scores,
        })

        # 检查告警
        await self._check_alerts(scores)

    def _compute_scores(self, sample: EvalSample) -> dict:
        """计算评估分数（简化版 - 生产中使用 LLM-as-Judge）"""
        response = sample.response
        context_text = " ".join(sample.context)

        # 简化的评估指标
        scores = {
            "response_length": min(len(response) / 500, 1.0),
            "has_context_reference": 1.0 if any(
                kw in response for kw in context_text.split()[:10]
            ) else 0.5,
            "not_empty": 1.0 if len(response) > 10 else 0.0,
        }

        scores["overall"] = sum(scores.values()) / len(scores)
        return scores

    async def _check_alerts(self, scores: dict):
        """检查是否需要告警"""
        if scores.get("overall", 1.0) < 0.5:
            for callback in self.alert_callbacks:
                await callback("quality_low", scores)

    def get_recent_metrics(self, window: int = 100) -> dict:
        """获取最近的评估统计"""
        recent = list(self.metrics_history)[-window:]
        if not recent:
            return {}

        import numpy as np
        overall_scores = [m["scores"].get("overall", 0) for m in recent]

        return {
            "sample_count": len(recent),
            "avg_quality": float(np.mean(overall_scores)),
            "min_quality": float(np.min(overall_scores)),
            "std_quality": float(np.std(overall_scores)),
        }

    def on_alert(self, callback):
        self.alert_callbacks.append(callback)
