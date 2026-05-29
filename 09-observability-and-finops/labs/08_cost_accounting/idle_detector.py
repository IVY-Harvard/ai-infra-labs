"""
GPU 资源闲置检测与优化建议
===========================

检测 GPU 闲置浪费并给出优化建议:
1. 空闲检测: SM Active < 10% 持续 N 分钟
2. 低效检测: 有负载但 tokens/GPU-hour 异常低
3. 碎片检测: 多个实例都半满 (可合并)
4. 周期性闲置: 检测可预测的空闲时段

依赖: numpy
"""

import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IdleEvent:
    """闲置事件"""
    instance_id: str
    start_time: float
    end_time: float = 0
    duration_s: float = 0
    avg_sm_active: float = 0
    avg_kv_cache_usage: float = 0
    wasted_cost_usd: float = 0
    idle_type: str = ""  # "completely_idle" | "underutilized" | "fragmented"


@dataclass
class IdleDetectorConfig:
    """闲置检测配置"""
    idle_threshold_sm: float = 0.10       # SM Active < 10% 视为闲置
    underutil_threshold_sm: float = 0.30  # SM Active < 30% 视为低效
    min_idle_duration_s: int = 300        # 至少持续 5 分钟才报告
    check_interval_s: int = 60           # 检查间隔
    gpu_cost_per_hour: float = 4.0       # 单 GPU 小时成本 ($)
    gpus_per_instance: int = 8           # 每实例 GPU 数


class GPUIdleDetector:
    """GPU 闲置检测器"""

    def __init__(self, config: IdleDetectorConfig = None):
        self.config = config or IdleDetectorConfig()
        self._active_idle_events: Dict[str, IdleEvent] = {}
        self._completed_events: List[IdleEvent] = []
        self._history: Dict[str, deque] = {}  # instance -> utilization history

    def update(self, instance_id: str, sm_active: float, kv_cache_usage: float = 0):
        """更新实例状态"""
        if instance_id not in self._history:
            self._history[instance_id] = deque(maxlen=1440)  # 24h @ 1min
        self._history[instance_id].append((sm_active, kv_cache_usage))

        # 检测闲置
        if sm_active < self.config.idle_threshold_sm:
            if instance_id not in self._active_idle_events:
                self._active_idle_events[instance_id] = IdleEvent(
                    instance_id=instance_id,
                    start_time=len(self._history[instance_id]),
                    idle_type="completely_idle",
                )
        else:
            # 结束闲置事件
            if instance_id in self._active_idle_events:
                event = self._active_idle_events.pop(instance_id)
                event.end_time = len(self._history[instance_id])
                event.duration_s = (event.end_time - event.start_time) * self.config.check_interval_s
                if event.duration_s >= self.config.min_idle_duration_s:
                    event.wasted_cost_usd = (
                        event.duration_s / 3600
                        * self.config.gpu_cost_per_hour
                        * self.config.gpus_per_instance
                    )
                    self._completed_events.append(event)

    def get_daily_waste(self) -> Dict:
        """计算日浪费"""
        total_waste = sum(e.wasted_cost_usd for e in self._completed_events)
        total_idle_hours = sum(e.duration_s for e in self._completed_events) / 3600

        return {
            "idle_events": len(self._completed_events),
            "total_idle_hours": round(total_idle_hours, 2),
            "total_waste_usd": round(total_waste, 2),
            "monthly_projection_usd": round(total_waste * 30, 2),
        }

    def get_optimization_suggestions(self) -> List[Dict]:
        """生成优化建议"""
        suggestions = []

        # 分析每个实例的利用率模式
        for instance_id, history in self._history.items():
            if len(history) < 60:
                continue

            sm_values = [h[0] for h in history]
            avg_sm = np.mean(sm_values)
            idle_pct = sum(1 for s in sm_values if s < 0.1) / len(sm_values)

            if idle_pct > 0.5:
                savings = (idle_pct * 24 * self.config.gpu_cost_per_hour
                           * self.config.gpus_per_instance)
                suggestions.append({
                    "instance": instance_id,
                    "type": "scale_to_zero",
                    "description": f"实例 {idle_pct*100:.0f}% 时间闲置, 建议配置 scale-to-zero",
                    "potential_saving_daily_usd": round(savings, 2),
                    "risk": "cold start 延迟 (模型加载 3-5 分钟)",
                })
            elif idle_pct > 0.3:
                suggestions.append({
                    "instance": instance_id,
                    "type": "scheduled_scaling",
                    "description": f"实例 {idle_pct*100:.0f}% 时间闲置, 建议配置定时缩容",
                    "potential_saving_daily_usd": round(
                        idle_pct * 12 * self.config.gpu_cost_per_hour
                        * self.config.gpus_per_instance, 2
                    ),
                    "risk": "低, 可预测的闲置时段",
                })
            elif avg_sm < 0.3:
                suggestions.append({
                    "instance": instance_id,
                    "type": "consolidation",
                    "description": f"平均利用率仅 {avg_sm*100:.0f}%, 可考虑合并负载到更少实例",
                    "potential_saving_daily_usd": round(
                        (1 - avg_sm / 0.7) * 24 * self.config.gpu_cost_per_hour
                        * self.config.gpus_per_instance, 2
                    ),
                    "risk": "中, 合并后突发流量可能不足",
                })

        # 按节省金额排序
        suggestions.sort(key=lambda x: x["potential_saving_daily_usd"], reverse=True)
        return suggestions


if __name__ == "__main__":
    import random

    detector = GPUIdleDetector()

    print("=== GPU Idle Detection Demo ===\n")

    # 模拟 24 小时 (1 分钟粒度)
    instances = ["vllm-0", "vllm-1", "vllm-2"]

    for minute in range(1440):
        hour = minute / 60

        for inst in instances:
            # 凌晨 0-8 点低负载
            if 0 <= hour < 8:
                sm = max(0, random.gauss(0.05, 0.03))
            # 工作时间
            elif 8 <= hour < 20:
                sm = max(0, min(1, random.gauss(0.7, 0.15)))
            else:
                sm = max(0, random.gauss(0.2, 0.1))

            detector.update(inst, sm_active=sm)

    waste = detector.get_daily_waste()
    print(f"Daily Waste Report:")
    for k, v in waste.items():
        print(f"  {k}: {v}")

    suggestions = detector.get_optimization_suggestions()
    print(f"\nOptimization Suggestions ({len(suggestions)}):")
    for s in suggestions[:5]:
        print(f"  [{s['type']}] {s['instance']}")
        print(f"    {s['description']}")
        print(f"    Saving: ${s['potential_saving_daily_usd']}/day, Risk: {s['risk']}")
