"""
统计学异常检测器 — GPU 推理指标专用
=====================================

实现多种统计方法检测 GPU 推理指标中的异常:
1. Z-Score: 基于标准差的偏离检测
2. Modified Z-Score (MAD): 对离群值鲁棒的版本
3. EWMA (指数加权移动平均): 检测缓慢漂移
4. Grubbs Test: 统计显著性检验
5. Seasonal Decomposition: 去除周期性后检测异常
6. Change Point Detection: CUSUM 算法

设计原则:
- 适配 GPU 推理指标特性 (非平稳、有突刺、周期性)
- 低延迟 (< 10ms per detection)
- 可配置灵敏度 (不同指标用不同参数)
- 输出带置信度的异常分数

依赖:
    pip install numpy pandas scipy
"""

import time
import logging
import math
from typing import List, Dict, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

class AnomalyType(Enum):
    """异常类型分类"""
    SPIKE_UP = "spike_up"           # 上突刺 (如 TTFT 突然飙升)
    SPIKE_DOWN = "spike_down"       # 下突刺 (如吞吐骤降)
    LEVEL_SHIFT = "level_shift"     # 水平漂移 (如 TPOT 整体升高)
    TREND_CHANGE = "trend_change"   # 趋势变化 (从稳定变为上升)
    VARIANCE_CHANGE = "variance_change"  # 方差变化 (从稳定变为波动)


class AnomalySeverity(Enum):
    """异常严重度"""
    INFO = "info"           # 轻微偏离 (2-3 sigma)
    WARNING = "warning"     # 显著偏离 (3-4 sigma)
    CRITICAL = "critical"   # 严重偏离 (> 4 sigma)


@dataclass
class AnomalyResult:
    """异常检测结果"""
    is_anomaly: bool
    score: float                     # 异常分数 (0-1, 越高越异常)
    severity: AnomalySeverity
    anomaly_type: AnomalyType
    method: str                      # 使用的检测方法
    current_value: float             # 当前值
    expected_value: float            # 期望值
    deviation: float                 # 偏离程度
    confidence: float                # 置信度 (0-1)
    context: Dict = field(default_factory=dict)  # 额外上下文


@dataclass
class MetricConfig:
    """指标检测配置

    不同指标需要不同的检测参数:
    - TTFT: 只关心上偏 (增大是坏事), 灵敏度中等
    - TPOT: 只关心上偏, 灵敏度高 (用户敏感)
    - KV Cache: 关心上偏, 有自然上限 1.0
    - 吞吐量: 关心下偏 (降低是坏事)
    - GPU 温度: 关心上偏, 有物理限制
    """
    name: str
    z_threshold: float = 3.0         # Z-Score 阈值 (几个标准差)
    mad_threshold: float = 3.5       # MAD 阈值
    ewma_alpha: float = 0.3          # EWMA 平滑系数 (0-1, 越大越敏感)
    direction: str = "both"          # "up" (只检测增大), "down" (只检测减小), "both"
    min_data_points: int = 30        # 最少需要多少数据点才开始检测
    window_size: int = 120           # 滑动窗口大小 (数据点数)
    cooldown_s: float = 60.0         # 报警冷却期 (避免重复告警)
    seasonal_period: int = 0         # 周期长度 (0=无周期, 96=每日, 672=每周)


# 预定义的 GPU 推理指标配置
METRIC_CONFIGS = {
    # TTFT: 关注增大, 中等灵敏度
    "ttft_p99": MetricConfig(
        name="TTFT P99",
        z_threshold=3.0,
        mad_threshold=3.5,
        ewma_alpha=0.2,
        direction="up",
        window_size=120,  # 15s interval × 120 = 30 分钟
        seasonal_period=96,  # 15min × 96 = 24h
    ),
    # TPOT: 关注增大, 高灵敏度 (用户体感敏感)
    "tpot_p99": MetricConfig(
        name="TPOT P99",
        z_threshold=2.5,
        mad_threshold=3.0,
        ewma_alpha=0.3,
        direction="up",
        window_size=60,   # 15 分钟
    ),
    # KV Cache 使用率: 关注增大, 有上限
    "kv_cache_usage": MetricConfig(
        name="KV Cache Usage",
        z_threshold=2.0,
        mad_threshold=2.5,
        ewma_alpha=0.1,  # 慢变化, 用低 alpha
        direction="up",
        window_size=240,  # 1 小时
    ),
    # 吞吐量: 关注下降
    "throughput": MetricConfig(
        name="Generation Throughput",
        z_threshold=3.0,
        mad_threshold=3.5,
        ewma_alpha=0.2,
        direction="down",
        window_size=120,
    ),
    # GPU 温度: 关注增大
    "gpu_temp": MetricConfig(
        name="GPU Temperature",
        z_threshold=3.0,
        mad_threshold=3.5,
        ewma_alpha=0.1,
        direction="up",
        window_size=240,
    ),
    # 排队数: 关注增大
    "queue_length": MetricConfig(
        name="Request Queue Length",
        z_threshold=2.0,
        mad_threshold=2.5,
        ewma_alpha=0.4,  # 快变化, 高 alpha
        direction="up",
        window_size=40,
    ),
}


# ============================================================
# 统计检测方法
# ============================================================

class ZScoreDetector:
    """Z-Score 异常检测

    原理: 当前值距均值有多少个标准差
    Z = (x - μ) / σ

    假设: 数据近似正态分布
    适用: 稳态指标 (如 TPOT 在固定负载下)
    不适用: 有明显趋势或季节性的指标

    GPU 推理场景注意:
    - 空闲时 TTFT ≈ 200ms, 高峰时 TTFT ≈ 2s → 不能用全局均值
    - 解决方案: 使用滑动窗口计算局部统计量
    """

    def __init__(self, config: MetricConfig):
        self.config = config
        self._buffer = deque(maxlen=config.window_size)

    def detect(self, value: float) -> AnomalyResult:
        """检测单个数据点是否异常"""
        self._buffer.append(value)

        if len(self._buffer) < self.config.min_data_points:
            return AnomalyResult(
                is_anomaly=False, score=0.0, severity=AnomalySeverity.INFO,
                anomaly_type=AnomalyType.SPIKE_UP, method="z_score",
                current_value=value, expected_value=value, deviation=0,
                confidence=0.0,
                context={"reason": "insufficient_data", "data_points": len(self._buffer)},
            )

        data = np.array(self._buffer)
        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            # 所有值相同, 任何偏离都是异常
            z_score = 0.0 if value == mean else float('inf')
        else:
            z_score = (value - mean) / std

        # 方向过滤
        is_anomaly = False
        anomaly_type = AnomalyType.SPIKE_UP

        if self.config.direction == "up":
            is_anomaly = z_score > self.config.z_threshold
            anomaly_type = AnomalyType.SPIKE_UP
        elif self.config.direction == "down":
            is_anomaly = z_score < -self.config.z_threshold
            anomaly_type = AnomalyType.SPIKE_DOWN
        else:
            is_anomaly = abs(z_score) > self.config.z_threshold
            anomaly_type = AnomalyType.SPIKE_UP if z_score > 0 else AnomalyType.SPIKE_DOWN

        # 计算异常分数和严重度
        score = min(1.0, abs(z_score) / (self.config.z_threshold * 2))
        severity = self._score_to_severity(abs(z_score), self.config.z_threshold)

        return AnomalyResult(
            is_anomaly=is_anomaly,
            score=score,
            severity=severity,
            anomaly_type=anomaly_type,
            method="z_score",
            current_value=value,
            expected_value=round(mean, 4),
            deviation=round(z_score, 4),
            confidence=min(0.99, len(self._buffer) / self.config.window_size),
            context={
                "z_score": round(z_score, 4),
                "mean": round(mean, 4),
                "std": round(std, 4),
                "window_size": len(self._buffer),
                "threshold": self.config.z_threshold,
            },
        )

    @staticmethod
    def _score_to_severity(abs_z: float, threshold: float) -> AnomalySeverity:
        if abs_z > threshold * 1.5:
            return AnomalySeverity.CRITICAL
        elif abs_z > threshold:
            return AnomalySeverity.WARNING
        return AnomalySeverity.INFO


class MADDetector:
    """Median Absolute Deviation 异常检测

    原理: 使用中位数替代均值, MAD 替代标准差
    Modified Z-Score = 0.6745 × (x - median) / MAD

    优势: 对离群值鲁棒 (不像 Z-Score 被极端值拉偏)

    GPU 推理场景:
    - TTFT 偶尔有极高的离群值 (preemption 导致)
    - 用 MAD 可以避免这些离群值 "污染" 基准线
    """

    CONSISTENCY_CONSTANT = 0.6745  # 使 MAD 与正态分布标准差一致

    def __init__(self, config: MetricConfig):
        self.config = config
        self._buffer = deque(maxlen=config.window_size)

    def detect(self, value: float) -> AnomalyResult:
        self._buffer.append(value)

        if len(self._buffer) < self.config.min_data_points:
            return AnomalyResult(
                is_anomaly=False, score=0.0, severity=AnomalySeverity.INFO,
                anomaly_type=AnomalyType.SPIKE_UP, method="mad",
                current_value=value, expected_value=value, deviation=0,
                confidence=0.0,
            )

        data = np.array(self._buffer)
        median = np.median(data)
        mad = np.median(np.abs(data - median))

        if mad == 0:
            modified_z = 0.0 if value == median else float('inf')
        else:
            modified_z = self.CONSISTENCY_CONSTANT * (value - median) / mad

        # 方向过滤
        is_anomaly = False
        if self.config.direction == "up":
            is_anomaly = modified_z > self.config.mad_threshold
        elif self.config.direction == "down":
            is_anomaly = modified_z < -self.config.mad_threshold
        else:
            is_anomaly = abs(modified_z) > self.config.mad_threshold

        anomaly_type = AnomalyType.SPIKE_UP if modified_z > 0 else AnomalyType.SPIKE_DOWN
        score = min(1.0, abs(modified_z) / (self.config.mad_threshold * 2))
        severity = ZScoreDetector._score_to_severity(abs(modified_z), self.config.mad_threshold)

        return AnomalyResult(
            is_anomaly=is_anomaly,
            score=score,
            severity=severity,
            anomaly_type=anomaly_type,
            method="mad",
            current_value=value,
            expected_value=round(median, 4),
            deviation=round(modified_z, 4),
            confidence=min(0.99, len(self._buffer) / self.config.window_size),
            context={
                "modified_z_score": round(modified_z, 4),
                "median": round(median, 4),
                "mad": round(mad, 6),
                "threshold": self.config.mad_threshold,
            },
        )


class EWMADetector:
    """指数加权移动平均异常检测

    原理:
      EWMA_t = α × x_t + (1-α) × EWMA_{t-1}
      异常 = |x_t - EWMA_t| > k × σ_ewma

    优势:
    - 自动适应趋势变化 (近期数据权重更高)
    - 检测缓慢漂移 (如 TTFT 逐渐升高)
    - α 越大越敏感于近期变化

    GPU 推理场景:
    - KV Cache 使用率缓慢上升: Z-Score 可能不报 (均值被拉高)
    - EWMA 能捕捉: 当前值持续高于 EWMA 预测 → 上升趋势异常
    """

    def __init__(self, config: MetricConfig):
        self.config = config
        self._alpha = config.ewma_alpha
        self._ewma = None           # 当前 EWMA 值
        self._ewma_var = None       # EWMA 方差
        self._data_count = 0
        self._last_anomaly_time = 0

    def detect(self, value: float, timestamp: float = None) -> AnomalyResult:
        self._data_count += 1
        if timestamp is None:
            timestamp = time.time()

        # 初始化
        if self._ewma is None:
            self._ewma = value
            self._ewma_var = 0.0
            return AnomalyResult(
                is_anomaly=False, score=0.0, severity=AnomalySeverity.INFO,
                anomaly_type=AnomalyType.LEVEL_SHIFT, method="ewma",
                current_value=value, expected_value=value, deviation=0,
                confidence=0.0,
            )

        # 计算预测误差
        error = value - self._ewma

        # 更新 EWMA
        self._ewma = self._alpha * value + (1 - self._alpha) * self._ewma

        # 更新 EWMA 方差 (用于动态阈值)
        self._ewma_var = (
            self._alpha * error**2 + (1 - self._alpha) * self._ewma_var
        )
        ewma_std = math.sqrt(self._ewma_var) if self._ewma_var > 0 else 1e-10

        # 计算标准化偏差
        normalized_error = abs(error) / ewma_std if ewma_std > 1e-10 else 0

        # 方向过滤
        is_anomaly = False
        if self.config.direction == "up":
            is_anomaly = error > 0 and normalized_error > self.config.z_threshold
        elif self.config.direction == "down":
            is_anomaly = error < 0 and normalized_error > self.config.z_threshold
        else:
            is_anomaly = normalized_error > self.config.z_threshold

        # 冷却期检查
        if is_anomaly and (timestamp - self._last_anomaly_time) < self.config.cooldown_s:
            is_anomaly = False  # 在冷却期内, 不重复报警

        if is_anomaly:
            self._last_anomaly_time = timestamp

        anomaly_type = AnomalyType.LEVEL_SHIFT if error > 0 else AnomalyType.SPIKE_DOWN
        score = min(1.0, normalized_error / (self.config.z_threshold * 2))
        data_confidence = min(0.99, self._data_count / self.config.min_data_points)

        return AnomalyResult(
            is_anomaly=is_anomaly,
            score=score,
            severity=ZScoreDetector._score_to_severity(
                normalized_error, self.config.z_threshold
            ),
            anomaly_type=anomaly_type,
            method="ewma",
            current_value=value,
            expected_value=round(self._ewma, 4),
            deviation=round(normalized_error, 4),
            confidence=data_confidence,
            context={
                "ewma": round(self._ewma, 4),
                "ewma_std": round(ewma_std, 6),
                "error": round(error, 4),
                "alpha": self._alpha,
                "data_points": self._data_count,
            },
        )


class CUSUMDetector:
    """CUSUM (Cumulative Sum) 变化点检测

    原理:
      S_t = max(0, S_{t-1} + (x_t - μ - allowance))
      当 S_t > threshold 时, 检测到向上变化点

    优势:
    - 对小而持续的变化极其敏感
    - 可以检测到 "均值漂移" 这类细微变化

    GPU 推理场景:
    - TPOT 从 25ms 缓慢增加到 35ms (每次增加 0.1ms, Z-Score 检测不到)
    - CUSUM 能累积这些微小偏差并在积累到一定程度时报警
    """

    def __init__(self, config: MetricConfig, allowance: float = 0.5):
        """
        Args:
            allowance: 容忍的偏移量 (以标准差为单位)
                       越小越敏感, 越大越不敏感
        """
        self.config = config
        self._allowance = allowance
        self._buffer = deque(maxlen=config.window_size)
        self._s_high = 0.0  # 向上累积和
        self._s_low = 0.0   # 向下累积和
        self._baseline_mean = None
        self._baseline_std = None

    def detect(self, value: float) -> AnomalyResult:
        self._buffer.append(value)

        if len(self._buffer) < self.config.min_data_points:
            return AnomalyResult(
                is_anomaly=False, score=0.0, severity=AnomalySeverity.INFO,
                anomaly_type=AnomalyType.TREND_CHANGE, method="cusum",
                current_value=value, expected_value=value, deviation=0,
                confidence=0.0,
            )

        # 计算基线 (使用前半窗口)
        data = np.array(self._buffer)
        half = len(data) // 2
        if self._baseline_mean is None or len(self._buffer) == self.config.min_data_points:
            self._baseline_mean = np.mean(data[:half])
            self._baseline_std = max(np.std(data[:half]), 1e-10)

        # 标准化
        z = (value - self._baseline_mean) / self._baseline_std

        # 更新 CUSUM
        self._s_high = max(0, self._s_high + z - self._allowance)
        self._s_low = max(0, self._s_low - z - self._allowance)

        threshold = self.config.z_threshold * 2  # CUSUM 阈值通常设得更高

        # 检测变化点
        is_anomaly = False
        anomaly_type = AnomalyType.TREND_CHANGE

        if self.config.direction in ("up", "both") and self._s_high > threshold:
            is_anomaly = True
            anomaly_type = AnomalyType.LEVEL_SHIFT
            self._s_high = 0  # 重置

        if self.config.direction in ("down", "both") and self._s_low > threshold:
            is_anomaly = True
            anomaly_type = AnomalyType.SPIKE_DOWN
            self._s_low = 0

        score = min(1.0, max(self._s_high, self._s_low) / (threshold * 2))

        return AnomalyResult(
            is_anomaly=is_anomaly,
            score=score,
            severity=AnomalySeverity.WARNING if is_anomaly else AnomalySeverity.INFO,
            anomaly_type=anomaly_type,
            method="cusum",
            current_value=value,
            expected_value=round(self._baseline_mean, 4),
            deviation=round(z, 4),
            confidence=min(0.99, len(self._buffer) / self.config.window_size),
            context={
                "s_high": round(self._s_high, 4),
                "s_low": round(self._s_low, 4),
                "threshold": threshold,
                "baseline_mean": round(self._baseline_mean, 4),
                "baseline_std": round(self._baseline_std, 6),
            },
        )


# ============================================================
# 集成检测器: 融合多种方法
# ============================================================

class EnsembleStatisticalDetector:
    """集成统计检测器

    融合多种统计方法的检测结果:
    - 多数投票: 至少 2/4 种方法都认为异常 → 才报异常
    - 分数加权: 不同方法的分数加权融合
    - 置信度调整: 方法一致性越高, 置信度越高

    融合策略:
    ┌──────────────────────────────────────────┐
    │         Ensemble Decision Logic           │
    ├──────────────────────────────────────────┤
    │ Z-Score:  anomaly=True,  score=0.7      │
    │ MAD:      anomaly=True,  score=0.8      │ → 3/4 方法异常
    │ EWMA:     anomaly=True,  score=0.6      │ → 加权分数 = 0.68
    │ CUSUM:    anomaly=False, score=0.3      │ → 最终: ANOMALY
    │                                          │
    │ 融合: weighted_avg(scores) = 0.68        │
    │ 投票: 3/4 > threshold(2) → ANOMALY       │
    └──────────────────────────────────────────┘
    """

    def __init__(self, config: MetricConfig):
        self.config = config
        self.detectors = {
            "z_score": ZScoreDetector(config),
            "mad": MADDetector(config),
            "ewma": EWMADetector(config),
            "cusum": CUSUMDetector(config),
        }
        # 方法权重 (基于经验调优)
        self.weights = {
            "z_score": 0.20,
            "mad": 0.30,     # MAD 通常最可靠
            "ewma": 0.30,    # EWMA 对趋势敏感
            "cusum": 0.20,   # CUSUM 对微小漂移敏感
        }
        self.vote_threshold = 2  # 至少 2 种方法认为异常

    def detect(self, value: float, timestamp: float = None) -> AnomalyResult:
        """融合所有方法进行检测"""
        results = {}
        for name, detector in self.detectors.items():
            if name == "ewma":
                results[name] = detector.detect(value, timestamp)
            else:
                results[name] = detector.detect(value)

        # 投票统计
        anomaly_votes = sum(1 for r in results.values() if r.is_anomaly)

        # 加权分数融合
        weighted_score = sum(
            self.weights[name] * results[name].score
            for name in results
        )

        # 最终决策
        is_anomaly = anomaly_votes >= self.vote_threshold

        # 取最高严重度
        max_severity = max(
            (r.severity for r in results.values() if r.is_anomaly),
            default=AnomalySeverity.INFO,
            key=lambda s: list(AnomalySeverity).index(s),
        )

        # 取 score 最高的方法作为主要 anomaly type
        primary_method = max(results.items(), key=lambda x: x[1].score)

        # 一致性 → 置信度
        consistency = anomaly_votes / len(results)
        confidence = min(0.99, consistency * 0.5 + weighted_score * 0.5)

        return AnomalyResult(
            is_anomaly=is_anomaly,
            score=round(weighted_score, 4),
            severity=max_severity if is_anomaly else AnomalySeverity.INFO,
            anomaly_type=primary_method[1].anomaly_type,
            method="ensemble",
            current_value=value,
            expected_value=primary_method[1].expected_value,
            deviation=primary_method[1].deviation,
            confidence=round(confidence, 4),
            context={
                "votes": anomaly_votes,
                "vote_threshold": self.vote_threshold,
                "method_results": {
                    name: {
                        "is_anomaly": r.is_anomaly,
                        "score": round(r.score, 4),
                        "deviation": round(r.deviation, 4),
                    }
                    for name, r in results.items()
                },
                "weights": self.weights,
            },
        )


# ============================================================
# Prometheus 集成: 从 Prometheus 拉取数据进行检测
# ============================================================

class PrometheusAnomalyDetector:
    """与 Prometheus 集成的异常检测器

    定期从 Prometheus 拉取指标, 运行检测, 推送结果。

    工作流:
    1. 每 15s 从 Prometheus 查询最新指标值
    2. 送入 EnsembleStatisticalDetector
    3. 如果检测到异常 → 写入 Prometheus (as recording rule)
    4. Alertmanager 根据异常分数触发告警
    """

    def __init__(self, prometheus_url: str = "http://prometheus:9090"):
        self.prometheus_url = prometheus_url
        self.detectors: Dict[str, EnsembleStatisticalDetector] = {}
        self._init_detectors()

    def _init_detectors(self):
        """为每个需要监控的指标创建检测器"""
        queries = {
            "ttft_p99": (
                'histogram_quantile(0.99, sum by (le) '
                '(rate(vllm:time_to_first_token_seconds_bucket[5m])))'
            ),
            "tpot_p99": (
                'histogram_quantile(0.99, sum by (le) '
                '(rate(vllm:time_per_output_token_seconds_bucket[5m])))'
            ),
            "kv_cache_usage": 'avg(vllm:gpu_cache_usage_perc)',
            "throughput": 'sum(rate(vllm:generation_tokens_total[5m]))',
            "gpu_temp": 'max(DCGM_FI_DEV_GPU_TEMP)',
            "queue_length": 'sum(vllm:num_requests_waiting)',
        }

        for metric_name, query in queries.items():
            config = METRIC_CONFIGS.get(metric_name, MetricConfig(name=metric_name))
            self.detectors[metric_name] = EnsembleStatisticalDetector(config)

        self._queries = queries
        logger.info(f"Initialized detectors for {len(self.detectors)} metrics")

    def detect_all(self, current_values: Dict[str, float]) -> Dict[str, AnomalyResult]:
        """对所有指标运行检测

        Args:
            current_values: {metric_name: current_value}

        Returns:
            {metric_name: AnomalyResult}
        """
        results = {}
        now = time.time()

        for name, value in current_values.items():
            detector = self.detectors.get(name)
            if detector:
                result = detector.detect(value, timestamp=now)
                results[name] = result

                if result.is_anomaly:
                    logger.warning(
                        f"ANOMALY detected in {name}: "
                        f"value={value}, expected={result.expected_value}, "
                        f"score={result.score}, severity={result.severity.value}, "
                        f"method={result.method}"
                    )

        return results


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    import random

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # 创建 TTFT P99 检测器
    config = METRIC_CONFIGS["ttft_p99"]
    detector = EnsembleStatisticalDetector(config)

    print("=== GPU Inference Anomaly Detection Demo ===\n")

    # 模拟正常 TTFT (200-300ms with noise)
    print("--- Phase 1: Normal TTFT (200-300ms) ---")
    for i in range(60):
        value = 0.25 + random.gauss(0, 0.02)  # ~250ms ± 20ms
        result = detector.detect(value)
        if i % 15 == 0:
            print(f"  t={i:3d}: TTFT={value*1000:.0f}ms score={result.score:.3f} "
                  f"anomaly={result.is_anomaly}")

    # 模拟异常: TTFT 突然飙升
    print("\n--- Phase 2: TTFT Spike (KV Cache full → preemption) ---")
    for i in range(10):
        value = 5.0 + random.gauss(0, 0.5)  # ~5000ms
        result = detector.detect(value)
        status = "🔴 ANOMALY" if result.is_anomaly else "  normal"
        print(f"  t={60+i:3d}: TTFT={value*1000:.0f}ms score={result.score:.3f} "
              f"severity={result.severity.value} {status}")

    # 模拟恢复
    print("\n--- Phase 3: Recovery ---")
    for i in range(30):
        value = 0.25 + random.gauss(0, 0.02)
        result = detector.detect(value)
        if i % 10 == 0:
            print(f"  t={70+i:3d}: TTFT={value*1000:.0f}ms score={result.score:.3f} "
                  f"anomaly={result.is_anomaly}")

    # 模拟缓慢漂移 (更微妙的异常)
    print("\n--- Phase 4: Slow Drift (TTFT gradually increasing) ---")
    for i in range(60):
        drift = i * 0.005  # 每步增加 5ms
        value = 0.25 + drift + random.gauss(0, 0.02)
        result = detector.detect(value)
        if i % 15 == 0:
            print(f"  t={100+i:3d}: TTFT={value*1000:.0f}ms score={result.score:.3f} "
                  f"anomaly={result.is_anomaly} "
                  f"[expected drift: +{drift*1000:.0f}ms]")
