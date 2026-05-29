"""
扩缩容模拟器 — 评估不同扩缩策略的效果
=========================================

在真正扩缩容之前,先用模拟器评估:
1. 不同 HPA 策略 (CPU/GPU/KV Cache/QPS 触发) 的响应速度
2. 扩容后的预期效果 (SLO 是否满足)
3. 成本影响 (扩容多少? 花多少钱?)
4. 预热时间考量 (GPU 推理服务启动慢)

依赖: numpy
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class ScalingTrigger(Enum):
    """扩缩容触发条件"""
    KV_CACHE_USAGE = "kv_cache_usage"
    GPU_UTILIZATION = "gpu_utilization"
    QUEUE_LENGTH = "queue_length"
    QPS = "qps"
    TTFT_P99 = "ttft_p99"
    MANUAL = "manual"


@dataclass
class ScalingPolicy:
    """扩缩容策略"""
    name: str
    trigger: ScalingTrigger
    scale_up_threshold: float       # 扩容阈值
    scale_down_threshold: float     # 缩容阈值
    scale_up_step: int = 1          # 每次扩容增加几个实例
    scale_down_step: int = 1        # 每次缩容减少几个实例
    cooldown_up_s: int = 300        # 扩容冷却期 (秒)
    cooldown_down_s: int = 600      # 缩容冷却期 (秒)
    stabilization_window_s: int = 60  # 稳定窗口 (连续 N 秒超阈值才触发)
    min_replicas: int = 1
    max_replicas: int = 16

    # GPU 推理特有: 预热时间
    warmup_time_s: int = 300        # 新实例从启动到可服务的时间 (模型加载)
    # Qwen2.5-72B from NVMe: ~180s, from NFS: ~300s, from S3: ~480s


@dataclass
class SimulationConfig:
    """模拟配置"""
    duration_hours: float = 24.0
    time_step_s: int = 15           # 模拟时间步 (秒)
    initial_replicas: int = 2
    gpus_per_replica: int = 8       # 每实例 GPU 数 (TP=8)
    gpu_cost_per_hour: float = 4.0  # 单 GPU 小时成本

    # 单实例容量
    single_instance_max_qps: float = 10.0
    single_instance_max_concurrent: int = 256
    single_instance_kv_cache_capacity: int = 1750000  # tokens


@dataclass
class TrafficPattern:
    """流量模式"""
    base_qps: float = 5.0
    peak_multiplier: float = 3.0      # 高峰是基线的几倍
    peak_start_hour: float = 9.0      # 高峰开始 (UTC)
    peak_end_hour: float = 18.0       # 高峰结束
    spike_probability: float = 0.02   # 每步发生突发的概率
    spike_multiplier: float = 5.0     # 突发倍率
    spike_duration_s: int = 300       # 突发持续时间 (秒)
    daily_growth_rate: float = 0.02   # 日增长率 (2%)


@dataclass
class SimulationStep:
    """模拟每一步的状态"""
    time_s: float
    actual_qps: float
    replicas: int
    warming_replicas: int             # 正在预热的实例
    effective_replicas: int            # 可用实例 (去除预热中的)
    kv_cache_usage: float
    queue_length: float
    ttft_p99_ms: float
    tpot_p99_ms: float
    dropped_requests: int
    cost_accumulated: float
    scaling_event: str = ""


class ScalingSimulator:
    """扩缩容模拟器

    模拟流程:
    1. 生成流量曲线 (日周期 + 随机突发)
    2. 每 time_step 计算当前负载
    3. 根据策略判断是否需要扩缩
    4. 考虑预热延迟
    5. 计算 SLO 指标
    6. 累计成本

    关键洞察:
    - GPU 推理扩容有 3-8 分钟延迟 (模型加载)
    - 如果用 KV Cache 触发, 可能已经太晚了
    - 基于流量预测的提前扩容比响应式更好
    """

    def __init__(
        self,
        policy: ScalingPolicy,
        config: SimulationConfig,
        traffic: TrafficPattern,
    ):
        self.policy = policy
        self.config = config
        self.traffic = traffic

    def generate_traffic(self) -> np.ndarray:
        """生成流量时序"""
        total_steps = int(self.config.duration_hours * 3600 / self.config.time_step_s)
        qps = np.zeros(total_steps)

        for i in range(total_steps):
            t_s = i * self.config.time_step_s
            hour_of_day = (t_s / 3600) % 24
            day = t_s / 86400

            # 基线 + 日增长
            base = self.traffic.base_qps * (1 + self.traffic.daily_growth_rate * day)

            # 日周期 (正弦近似)
            if self.traffic.peak_start_hour <= hour_of_day <= self.traffic.peak_end_hour:
                # 高峰时段
                peak_center = (self.traffic.peak_start_hour + self.traffic.peak_end_hour) / 2
                peak_width = (self.traffic.peak_end_hour - self.traffic.peak_start_hour) / 2
                peak_factor = 1 + (self.traffic.peak_multiplier - 1) * np.cos(
                    np.pi * (hour_of_day - peak_center) / peak_width
                )**2
                qps[i] = base * peak_factor
            else:
                qps[i] = base * 0.3  # 低谷

            # 噪声
            qps[i] *= (1 + np.random.normal(0, 0.1))

        # 突发流量
        spike_active = 0
        for i in range(total_steps):
            if spike_active > 0:
                qps[i] *= self.traffic.spike_multiplier
                spike_active -= 1
            elif np.random.random() < self.traffic.spike_probability:
                spike_active = self.traffic.spike_duration_s // self.config.time_step_s

        return np.maximum(0, qps)

    def simulate(self) -> List[SimulationStep]:
        """运行完整模拟"""
        traffic = self.generate_traffic()
        total_steps = len(traffic)

        # 状态初始化
        replicas = self.config.initial_replicas
        warming_queue: List[int] = []  # (ready_at_step, count)
        cost = 0.0
        last_scale_up_step = -9999
        last_scale_down_step = -9999
        above_threshold_count = 0
        below_threshold_count = 0

        history: List[SimulationStep] = []

        for step in range(total_steps):
            t_s = step * self.config.time_step_s
            qps = traffic[step]

            # 处理预热完成的实例
            effective_replicas = replicas
            warming = 0
            new_warming = []
            for ready_step, count in warming_queue:
                if step >= ready_step:
                    effective_replicas += 0  # 已在 replicas 中计数
                else:
                    effective_replicas -= count
                    warming += count
                    new_warming.append((ready_step, count))
            warming_queue = new_warming

            effective_replicas = max(1, effective_replicas)

            # 计算负载指标
            total_capacity_qps = effective_replicas * self.config.single_instance_max_qps
            load_ratio = qps / total_capacity_qps if total_capacity_qps > 0 else 10

            # KV Cache 使用率 (与负载比正相关)
            kv_usage = min(0.99, load_ratio * 0.7 + np.random.normal(0, 0.02))
            kv_usage = max(0.0, kv_usage)

            # 排队 (负载超过容量时出现)
            queue = max(0, (load_ratio - 0.8) * 100 + np.random.normal(0, 2))

            # TTFT (受排队和 KV Cache 影响)
            ttft_base = 250  # ms
            ttft_queue = queue * 100  # ms per queued request
            ttft_cache = max(0, (kv_usage - 0.8)) * 5000  # KV > 0.8 时 TTFT 飙升
            ttft_p99 = ttft_base + ttft_queue + ttft_cache + np.random.normal(0, 50)

            # TPOT (受 batch size 影响)
            batch_estimate = min(256, qps / effective_replicas * 2) if effective_replicas > 0 else 0
            tpot_p99 = 15 + 0.3 * batch_estimate + np.random.normal(0, 3)

            # 丢弃的请求
            dropped = max(0, int((load_ratio - 1.2) * qps)) if load_ratio > 1.2 else 0

            # 累积成本
            step_cost = (replicas * self.config.gpus_per_replica
                         * self.config.gpu_cost_per_hour
                         * self.config.time_step_s / 3600)
            cost += step_cost

            # === 扩缩容决策 ===
            trigger_value = self._get_trigger_value(
                self.policy.trigger, kv_usage, load_ratio, queue, qps / total_capacity_qps, ttft_p99
            )

            scaling_event = ""

            # 扩容检查
            if trigger_value > self.policy.scale_up_threshold:
                above_threshold_count += 1
            else:
                above_threshold_count = 0

            stabilization_steps = self.policy.stabilization_window_s // self.config.time_step_s
            cooldown_up_steps = self.policy.cooldown_up_s // self.config.time_step_s
            cooldown_down_steps = self.policy.cooldown_down_s // self.config.time_step_s

            if (above_threshold_count >= stabilization_steps
                and step - last_scale_up_step >= cooldown_up_steps
                and replicas < self.policy.max_replicas):
                # 扩容
                new_count = min(
                    self.policy.scale_up_step,
                    self.policy.max_replicas - replicas
                )
                replicas += new_count
                warmup_steps = self.policy.warmup_time_s // self.config.time_step_s
                warming_queue.append((step + warmup_steps, new_count))
                last_scale_up_step = step
                above_threshold_count = 0
                scaling_event = f"SCALE_UP +{new_count} (total={replicas}, warming={new_count})"
                logger.info(f"t={t_s/3600:.1f}h: {scaling_event}")

            # 缩容检查
            if trigger_value < self.policy.scale_down_threshold:
                below_threshold_count += 1
            else:
                below_threshold_count = 0

            if (below_threshold_count >= stabilization_steps * 2  # 缩容更保守
                and step - last_scale_down_step >= cooldown_down_steps
                and replicas > self.policy.min_replicas
                and not warming_queue):  # 有预热中的不缩容
                replicas -= self.policy.scale_down_step
                replicas = max(self.policy.min_replicas, replicas)
                last_scale_down_step = step
                below_threshold_count = 0
                scaling_event = f"SCALE_DOWN -1 (total={replicas})"
                logger.info(f"t={t_s/3600:.1f}h: {scaling_event}")

            history.append(SimulationStep(
                time_s=t_s,
                actual_qps=round(qps, 2),
                replicas=replicas,
                warming_replicas=warming,
                effective_replicas=effective_replicas,
                kv_cache_usage=round(kv_usage, 3),
                queue_length=round(max(0, queue), 1),
                ttft_p99_ms=round(max(0, ttft_p99), 1),
                tpot_p99_ms=round(max(0, tpot_p99), 1),
                dropped_requests=dropped,
                cost_accumulated=round(cost, 2),
                scaling_event=scaling_event,
            ))

        return history

    def _get_trigger_value(
        self, trigger: ScalingTrigger,
        kv_usage, load_ratio, queue, qps_ratio, ttft
    ) -> float:
        if trigger == ScalingTrigger.KV_CACHE_USAGE:
            return kv_usage
        elif trigger == ScalingTrigger.GPU_UTILIZATION:
            return min(1.0, load_ratio * 0.85)
        elif trigger == ScalingTrigger.QUEUE_LENGTH:
            return queue
        elif trigger == ScalingTrigger.QPS:
            return qps_ratio
        elif trigger == ScalingTrigger.TTFT_P99:
            return ttft / 1000.0  # convert to seconds
        return 0

    def summarize(self, history: List[SimulationStep]) -> Dict:
        """生成模拟报告摘要"""
        ttfts = [s.ttft_p99_ms for s in history]
        tpots = [s.tpot_p99_ms for s in history]
        replicas = [s.replicas for s in history]
        dropped = sum(s.dropped_requests for s in history)
        total_cost = history[-1].cost_accumulated if history else 0

        # SLO 达标率
        ttft_slo_met = sum(1 for t in ttfts if t < 5000) / len(ttfts) * 100
        tpot_slo_met = sum(1 for t in tpots if t < 80) / len(tpots) * 100

        scaling_events = [s for s in history if s.scaling_event]

        return {
            "policy": self.policy.name,
            "trigger": self.policy.trigger.value,
            "duration_hours": self.config.duration_hours,
            "slo_compliance": {
                "ttft_p99_slo_met_pct": round(ttft_slo_met, 2),
                "tpot_p99_slo_met_pct": round(tpot_slo_met, 2),
            },
            "latency": {
                "ttft_p99_avg_ms": round(np.mean(ttfts), 1),
                "ttft_p99_max_ms": round(max(ttfts), 1),
                "tpot_p99_avg_ms": round(np.mean(tpots), 1),
            },
            "scaling": {
                "min_replicas": min(replicas),
                "max_replicas": max(replicas),
                "avg_replicas": round(np.mean(replicas), 1),
                "total_scaling_events": len(scaling_events),
                "scale_up_events": sum(1 for s in scaling_events if "UP" in s.scaling_event),
                "scale_down_events": sum(1 for s in scaling_events if "DOWN" in s.scaling_event),
            },
            "cost": {
                "total_usd": round(total_cost, 2),
                "avg_hourly_usd": round(total_cost / self.config.duration_hours, 2),
                "total_gpu_hours": round(
                    np.sum(replicas) * self.config.gpus_per_replica
                    * self.config.time_step_s / 3600, 1
                ),
            },
            "quality": {
                "total_dropped_requests": dropped,
            },
        }


# ============================================================
# 策略对比
# ============================================================

def compare_policies():
    """对比不同扩缩策略"""
    config = SimulationConfig(
        duration_hours=24,
        initial_replicas=2,
        gpus_per_replica=8,
        gpu_cost_per_hour=4.0,
        single_instance_max_qps=10.0,
    )

    traffic = TrafficPattern(
        base_qps=8.0,
        peak_multiplier=3.0,
        spike_probability=0.01,
    )

    policies = [
        ScalingPolicy(
            name="KV Cache Based",
            trigger=ScalingTrigger.KV_CACHE_USAGE,
            scale_up_threshold=0.85,
            scale_down_threshold=0.5,
            warmup_time_s=300,
            min_replicas=1, max_replicas=8,
        ),
        ScalingPolicy(
            name="Queue Based",
            trigger=ScalingTrigger.QUEUE_LENGTH,
            scale_up_threshold=10,
            scale_down_threshold=2,
            warmup_time_s=300,
            min_replicas=1, max_replicas=8,
        ),
        ScalingPolicy(
            name="TTFT SLO Based",
            trigger=ScalingTrigger.TTFT_P99,
            scale_up_threshold=3.0,   # TTFT > 3s
            scale_down_threshold=1.0, # TTFT < 1s
            warmup_time_s=300,
            min_replicas=1, max_replicas=8,
        ),
        ScalingPolicy(
            name="Aggressive (Low Threshold)",
            trigger=ScalingTrigger.KV_CACHE_USAGE,
            scale_up_threshold=0.7,
            scale_down_threshold=0.4,
            scale_up_step=2,          # 每次扩 2 个
            warmup_time_s=300,
            min_replicas=2, max_replicas=8,
        ),
    ]

    results = []
    for policy in policies:
        sim = ScalingSimulator(policy, config, traffic)
        history = sim.simulate()
        summary = sim.summarize(history)
        results.append(summary)

    return results


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.WARNING)

    print("=== Scaling Strategy Comparison ===\n")
    results = compare_policies()

    for r in results:
        print(f"--- {r['policy']} ({r['trigger']}) ---")
        print(f"  SLO: TTFT={r['slo_compliance']['ttft_p99_slo_met_pct']:.1f}%, "
              f"TPOT={r['slo_compliance']['tpot_p99_slo_met_pct']:.1f}%")
        print(f"  Replicas: {r['scaling']['min_replicas']}-{r['scaling']['max_replicas']} "
              f"(avg {r['scaling']['avg_replicas']})")
        print(f"  Cost: ${r['cost']['total_usd']:.0f}/day "
              f"(${r['cost']['avg_hourly_usd']:.1f}/h)")
        print(f"  Scaling events: {r['scaling']['total_scaling_events']} "
              f"(up={r['scaling']['scale_up_events']}, down={r['scaling']['scale_down_events']})")
        print(f"  Dropped: {r['quality']['total_dropped_requests']} requests")
        print()
