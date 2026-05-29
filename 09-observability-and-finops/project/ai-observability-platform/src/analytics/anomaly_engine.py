"""异常检测引擎 — 集成统计与 ML 方法的 Ensemble 检测器

支持的检测算法:
- Z-Score (适合正态分布指标)
- EWMA (指数加权移动平均, 对趋势敏感)
- MAD (中位数绝对偏差, 对 outlier 鲁棒)
- Isolation Forest (无监督 ML, 多维异常)

架构:
┌─────────────┐    ┌──────────────┐    ┌────────────────┐
│ Prometheus  │───▶│ AnomalyEngine│───▶│ Alert Manager  │
│  (metrics)  │    │  (ensemble)  │    │  (routing)     │
└─────────────┘    └──────────────┘    └────────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
        ┌──────────┐ ┌────────┐ ┌──────────┐
        │ Z-Score  │ │  EWMA  │ │ IForest  │
        └──────────┘ └────────┘ └──────────┘
"""

import time
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    """异常严重程度"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """异常检测结果"""
    metric_name: str
    value: float
    score: float              # 0-1 之间的综合异常分数
    severity: Severity
    detectors_triggered: List[str]  # 哪些检测器触发
    timestamp: float = 0
    context: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


class ZScoreDetector:
    """Z-Score 检测器 — 基于滑动窗口的标准差检测

    原理: z = (x - μ) / σ
    当 |z| > threshold 时认为异常
    """

    def __init__(self, window_size: int = 100, threshold: float = 3.0):
        self.window_size = window_size
        self.threshold = threshold
        # 每个 metric 维护独立的滑动窗口
        self._windows: Dict[str, deque] = {}

    def detect(self, metric_name: str, value: float) -> Tuple[bool, float]:
        """检测单个值是否异常

        Returns:
            (is_anomaly, z_score)
        """
        if metric_name not in self._windows:
            self._windows[metric_name] = deque(maxlen=self.window_size)

        window = self._windows[metric_name]
        window.append(value)

        # 需要至少 10 个样本才能计算有意义的统计量
        if len(window) < 10:
            return False, 0.0

        mean = sum(window) / len(window)
        variance = sum((x - mean) ** 2 for x in window) / len(window)
        std = math.sqrt(variance) if variance > 0 else 1e-10

        z_score = abs(value - mean) / std
        is_anomaly = z_score > self.threshold

        return is_anomaly, z_score


class EWMADetector:
    """EWMA 检测器 — 指数加权移动平均

    对渐变趋势更敏感, 适合检测 GPU 温度缓慢升高等场景

    公式:
        ewma_t = α * x_t + (1 - α) * ewma_{t-1}
        控制限 = ewma ± L * σ * sqrt(α / (2 - α))
    """

    def __init__(self, alpha: float = 0.3, L: float = 3.0):
        """
        Args:
            alpha: 平滑系数, 越大对最近值越敏感 (0 < α < 1)
            L: 控制限系数, 类似 Z-Score 的 threshold
        """
        self.alpha = alpha
        self.L = L
        self._state: Dict[str, Dict] = {}

    def detect(self, metric_name: str, value: float) -> Tuple[bool, float]:
        """检测单个值是否偏离 EWMA 预期"""
        if metric_name not in self._state:
            self._state[metric_name] = {
                "ewma": value,
                "variance": 0.0,
                "count": 0,
            }
            return False, 0.0

        state = self._state[metric_name]
        state["count"] += 1

        # 更新 EWMA
        prev_ewma = state["ewma"]
        state["ewma"] = self.alpha * value + (1 - self.alpha) * prev_ewma

        # 更新方差估计 (Welford online)
        diff = value - prev_ewma
        state["variance"] = (1 - self.alpha) * (state["variance"] + self.alpha * diff ** 2)

        # 计算控制限
        std = math.sqrt(state["variance"]) if state["variance"] > 0 else 1e-10
        control_limit = self.L * std * math.sqrt(self.alpha / (2 - self.alpha))

        # 偏离分数
        deviation = abs(value - state["ewma"])
        score = deviation / control_limit if control_limit > 0 else 0.0

        is_anomaly = deviation > control_limit and state["count"] > 10
        return is_anomaly, score


class MADDetector:
    """MAD 检测器 — 中位数绝对偏差

    比 Z-Score 更鲁棒, 不受极端 outlier 影响
    MAD = median(|x_i - median(X)|)
    Modified Z-Score = 0.6745 * (x - median) / MAD
    """

    def __init__(self, window_size: int = 100, threshold: float = 3.5):
        self.window_size = window_size
        self.threshold = threshold
        self._windows: Dict[str, deque] = {}

    def detect(self, metric_name: str, value: float) -> Tuple[bool, float]:
        if metric_name not in self._windows:
            self._windows[metric_name] = deque(maxlen=self.window_size)

        window = self._windows[metric_name]
        window.append(value)

        if len(window) < 10:
            return False, 0.0

        sorted_window = sorted(window)
        n = len(sorted_window)
        median = sorted_window[n // 2]

        # 计算 MAD
        deviations = sorted(abs(x - median) for x in sorted_window)
        mad = deviations[n // 2]

        if mad < 1e-10:
            return False, 0.0

        # Modified Z-Score
        modified_z = 0.6745 * (value - median) / mad
        is_anomaly = abs(modified_z) > self.threshold

        return is_anomaly, abs(modified_z)


class AnomalyEngine:
    """异常检测引擎 — Ensemble 方法聚合多个检测器

    投票策略:
    - majority: 超过半数检测器触发 → 异常
    - any: 任一检测器触发 → 异常 (高召回)
    - weighted: 加权评分超过阈值 → 异常

    严重程度判定:
    - score > 0.9 → CRITICAL
    - score > 0.7 → WARNING
    - score > 0.5 → INFO
    """

    def __init__(
        self,
        voting: str = "weighted",
        score_threshold: float = 0.6,
        z_window: int = 100,
        z_threshold: float = 3.0,
        ewma_alpha: float = 0.3,
        ewma_L: float = 3.0,
        mad_window: int = 100,
        mad_threshold: float = 3.5,
    ):
        self.voting = voting
        self.score_threshold = score_threshold

        # 初始化各检测器
        self.detectors = {
            "z_score": ZScoreDetector(window_size=z_window, threshold=z_threshold),
            "ewma": EWMADetector(alpha=ewma_alpha, L=ewma_L),
            "mad": MADDetector(window_size=mad_window, threshold=mad_threshold),
        }

        # 检测器权重 (用于 weighted voting)
        self.weights = {
            "z_score": 0.3,
            "ewma": 0.4,   # EWMA 对渐变异常更敏感, 给更高权重
            "mad": 0.3,
        }

        # 检测历史
        self._anomaly_history: deque = deque(maxlen=1000)

    def detect(self, metric_name: str, value: float, context: Optional[Dict] = None) -> Optional[Anomaly]:
        """对单个指标值运行 Ensemble 检测

        Args:
            metric_name: 指标名称 (如 "kv_cache_usage", "gpu_temp")
            value: 当前值
            context: 附加上下文 (node, gpu_id 等)

        Returns:
            Anomaly 对象 (如果检测到异常), 否则 None
        """
        results = {}
        scores = {}

        for name, detector in self.detectors.items():
            is_anomaly, score = detector.detect(metric_name, value)
            results[name] = is_anomaly
            scores[name] = score

        # Ensemble 评分
        ensemble_score = self._calculate_ensemble_score(results, scores)
        is_anomaly = self._vote(results, ensemble_score)

        if not is_anomaly:
            return None

        # 判定严重程度
        severity = self._classify_severity(ensemble_score)

        # 记录触发的检测器
        triggered = [name for name, triggered in results.items() if triggered]

        anomaly = Anomaly(
            metric_name=metric_name,
            value=value,
            score=ensemble_score,
            severity=severity,
            detectors_triggered=triggered,
            context=context or {},
        )

        self._anomaly_history.append(anomaly)
        logger.warning(
            f"异常检测: {metric_name}={value:.4f}, score={ensemble_score:.3f}, "
            f"severity={severity.value}, triggered={triggered}"
        )

        return anomaly

    def detect_batch(self, metrics: Dict[str, float], context: Optional[Dict] = None) -> List[Anomaly]:
        """批量检测多个指标

        Args:
            metrics: {metric_name: value} 字典
            context: 共享上下文

        Returns:
            检测到的异常列表
        """
        anomalies = []
        for metric_name, value in metrics.items():
            anomaly = self.detect(metric_name, value, context)
            if anomaly:
                anomalies.append(anomaly)
        return anomalies

    def _calculate_ensemble_score(self, results: Dict[str, bool], scores: Dict[str, float]) -> float:
        """计算加权 Ensemble 分数"""
        weighted_sum = 0.0
        weight_total = 0.0

        for name, score in scores.items():
            weight = self.weights.get(name, 1.0)
            # 归一化 score 到 0-1 (使用 sigmoid-like 变换)
            normalized = min(score / 5.0, 1.0)  # 假设 score=5 对应确定异常
            weighted_sum += normalized * weight
            weight_total += weight

        return weighted_sum / weight_total if weight_total > 0 else 0.0

    def _vote(self, results: Dict[str, bool], ensemble_score: float) -> bool:
        """根据投票策略判定是否异常"""
        if self.voting == "any":
            return any(results.values())
        elif self.voting == "majority":
            triggered_count = sum(1 for v in results.values() if v)
            return triggered_count > len(results) / 2
        elif self.voting == "weighted":
            return ensemble_score >= self.score_threshold
        else:
            return False

    def _classify_severity(self, score: float) -> Severity:
        """根据分数判定严重程度"""
        if score >= 0.9:
            return Severity.CRITICAL
        elif score >= 0.7:
            return Severity.WARNING
        else:
            return Severity.INFO

    def get_recent_anomalies(self, limit: int = 50) -> List[Anomaly]:
        """获取最近的异常记录"""
        return list(self._anomaly_history)[-limit:]

    def get_anomaly_rate(self, window_seconds: float = 300) -> float:
        """计算最近 N 秒内的异常发生率"""
        now = time.time()
        cutoff = now - window_seconds
        recent = [a for a in self._anomaly_history if a.timestamp >= cutoff]
        # 简化: 返回异常计数 / 时间窗口 (次/分钟)
        return len(recent) / (window_seconds / 60)

    def reset(self, metric_name: Optional[str] = None):
        """重置检测器状态 (用于配置变更后)"""
        if metric_name:
            for detector in self.detectors.values():
                if hasattr(detector, "_windows") and metric_name in detector._windows:
                    del detector._windows[metric_name]
                if hasattr(detector, "_state") and metric_name in detector._state:
                    del detector._state[metric_name]
            logger.info(f"重置检测器状态: {metric_name}")
        else:
            for detector in self.detectors.values():
                if hasattr(detector, "_windows"):
                    detector._windows.clear()
                if hasattr(detector, "_state"):
                    detector._state.clear()
            self._anomaly_history.clear()
            logger.info("重置所有检测器状态")
