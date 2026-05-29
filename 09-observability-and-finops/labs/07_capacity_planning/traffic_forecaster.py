"""
流量预测器 — Holt-Winters 季节性时序预测
==========================================

基于历史指标数据预测未来流量,支撑容量规划:
1. Holt-Winters Triple Exponential Smoothing (日/周季节性)
2. 线性回归趋势预测 (简单但稳定)
3. 置信区间估算

依赖: numpy, scipy
"""

import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """预测结果"""
    timestamps: List[float]
    values: List[float]               # 预测值
    lower_bound: List[float]          # 置信区间下界
    upper_bound: List[float]          # 置信区间上界
    confidence_level: float           # 置信水平
    method: str                       # 使用的方法
    metrics: Dict                     # 模型评估指标


class HoltWintersForecaster:
    """Holt-Winters 三重指数平滑预测

    组件:
    - Level (l_t): 当前水平
    - Trend (b_t): 趋势 (增长/下降速率)
    - Seasonal (s_t): 季节性分量

    更新公式 (乘法季节性):
    l_t = α × (y_t / s_{t-m}) + (1-α) × (l_{t-1} + b_{t-1})
    b_t = β × (l_t - l_{t-1}) + (1-β) × b_{t-1}
    s_t = γ × (y_t / l_t) + (1-γ) × s_{t-m}

    预测:
    ŷ_{t+h} = (l_t + h × b_t) × s_{t-m+h}

    GPU 推理场景:
    - 日季节性: 工作时间 QPS 高, 凌晨低
    - 周季节性: 工作日高, 周末低
    - 趋势: 业务增长导致整体上升
    """

    def __init__(
        self,
        seasonal_period: int = 96,    # 季节周期 (96 = 24h × 4次/h)
        alpha: float = 0.3,           # Level 平滑系数
        beta: float = 0.1,            # Trend 平滑系数
        gamma: float = 0.2,           # Seasonal 平滑系数
    ):
        self.m = seasonal_period
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self._level = None
        self._trend = None
        self._seasonal = None
        self._fitted = False

    def fit(self, data: np.ndarray) -> "HoltWintersForecaster":
        """拟合历史数据

        Args:
            data: 等间隔时序数据, 长度至少 2 × seasonal_period
        """
        n = len(data)
        if n < 2 * self.m:
            logger.warning(
                f"数据长度 {n} < 2 × 季节周期 {self.m}, "
                f"退化为简单指数平滑"
            )
            self.m = 1

        # 初始化 Level: 第一个季节周期的均值
        self._level = np.mean(data[:self.m])

        # 初始化 Trend: 两个周期均值之差 / 周期长度
        if n >= 2 * self.m:
            first_season_mean = np.mean(data[:self.m])
            second_season_mean = np.mean(data[self.m:2*self.m])
            self._trend = (second_season_mean - first_season_mean) / self.m
        else:
            self._trend = 0.0

        # 初始化 Seasonal: 每个点与同周期均值的比值
        self._seasonal = np.ones(self.m)
        for i in range(self.m):
            if self._level != 0:
                values_at_i = data[i::self.m]
                self._seasonal[i] = np.mean(values_at_i) / self._level

        # 迭代更新
        self._levels_history = [self._level]
        self._trends_history = [self._trend]
        self._seasonals_history = list(self._seasonal)
        self._residuals = []

        for t in range(n):
            season_idx = t % self.m
            y = data[t]

            # 避免除零
            s_prev = self._seasonal[season_idx]
            if s_prev == 0:
                s_prev = 1e-10

            # Level 更新
            new_level = (
                self.alpha * (y / s_prev)
                + (1 - self.alpha) * (self._level + self._trend)
            )

            # Trend 更新
            new_trend = (
                self.beta * (new_level - self._level)
                + (1 - self.beta) * self._trend
            )

            # Seasonal 更新
            if new_level != 0:
                new_seasonal = (
                    self.gamma * (y / new_level)
                    + (1 - self.gamma) * s_prev
                )
            else:
                new_seasonal = s_prev

            # 拟合值
            fitted = (self._level + self._trend) * s_prev
            self._residuals.append(y - fitted)

            # 更新状态
            self._level = new_level
            self._trend = new_trend
            self._seasonal[season_idx] = new_seasonal

        self._fitted = True
        self._residual_std = np.std(self._residuals) if self._residuals else 0
        logger.info(
            f"Holt-Winters fitted: level={self._level:.2f}, "
            f"trend={self._trend:.4f}, residual_std={self._residual_std:.4f}"
        )
        return self

    def predict(
        self,
        steps: int,
        confidence: float = 0.95,
    ) -> ForecastResult:
        """预测未来 steps 步

        Args:
            steps: 预测步数
            confidence: 置信水平 (0.95 = 95%)

        Returns:
            ForecastResult
        """
        if not self._fitted:
            raise RuntimeError("模型未拟合, 请先调用 fit()")

        predictions = []
        lower = []
        upper = []

        # Z-score for confidence interval
        from scipy import stats
        z = stats.norm.ppf((1 + confidence) / 2)

        for h in range(1, steps + 1):
            season_idx = h % self.m
            # 点预测
            forecast = (self._level + h * self._trend) * self._seasonal[season_idx]
            predictions.append(forecast)

            # 置信区间 (随预测步数增大而变宽)
            interval_width = z * self._residual_std * np.sqrt(h)
            lower.append(max(0, forecast - interval_width))  # 非负
            upper.append(forecast + interval_width)

        return ForecastResult(
            timestamps=list(range(steps)),
            values=predictions,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence,
            method="holt_winters",
            metrics={
                "residual_std": round(self._residual_std, 4),
                "final_level": round(self._level, 4),
                "final_trend": round(self._trend, 6),
                "mape": self._compute_mape(),
            },
        )

    def _compute_mape(self) -> float:
        """计算 MAPE (Mean Absolute Percentage Error)"""
        if not self._residuals:
            return 0.0
        # 避免除零
        abs_errors = np.abs(self._residuals)
        return round(float(np.mean(abs_errors / (np.abs(np.mean(abs_errors)) + 1e-10)) * 100), 2)


class LinearTrendForecaster:
    """简单线性趋势预测 (鲁棒备选)

    当数据量不足以支撑 Holt-Winters 时使用。
    等效于 Prometheus 的 predict_linear()。
    """

    def __init__(self):
        self._slope = 0
        self._intercept = 0
        self._residual_std = 0
        self._fitted = False

    def fit(self, data: np.ndarray) -> "LinearTrendForecaster":
        x = np.arange(len(data))
        # 最小二乘拟合
        A = np.vstack([x, np.ones(len(x))]).T
        result = np.linalg.lstsq(A, data, rcond=None)
        self._slope, self._intercept = result[0]

        # 残差
        fitted = self._slope * x + self._intercept
        residuals = data - fitted
        self._residual_std = np.std(residuals)
        self._n = len(data)
        self._fitted = True
        return self

    def predict(self, steps: int, confidence: float = 0.95) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("模型未拟合")

        from scipy import stats
        z = stats.norm.ppf((1 + confidence) / 2)

        predictions = []
        lower = []
        upper = []

        for h in range(1, steps + 1):
            t = self._n + h
            forecast = self._slope * t + self._intercept
            interval = z * self._residual_std * np.sqrt(1 + 1/self._n + (h**2) / (self._n**2))
            predictions.append(forecast)
            lower.append(max(0, forecast - interval))
            upper.append(forecast + interval)

        return ForecastResult(
            timestamps=list(range(steps)),
            values=predictions,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence,
            method="linear_trend",
            metrics={
                "slope": round(self._slope, 6),
                "intercept": round(self._intercept, 4),
                "residual_std": round(self._residual_std, 4),
            },
        )

    def time_to_threshold(self, threshold: float) -> Optional[int]:
        """预测何时达到阈值 (步数)"""
        if not self._fitted or self._slope <= 0:
            return None
        current = self._slope * self._n + self._intercept
        if current >= threshold:
            return 0
        steps = int((threshold - current) / self._slope)
        return max(0, steps)


class CapacityForecaster:
    """容量预测引擎 — 综合多种方法"""

    def __init__(self, sampling_interval_s: int = 900):
        """
        Args:
            sampling_interval_s: 数据采样间隔 (秒), 默认 15 分钟
        """
        self.interval = sampling_interval_s
        self.points_per_day = 86400 // sampling_interval_s
        self.points_per_week = self.points_per_day * 7

    def forecast_metric(
        self,
        data: np.ndarray,
        forecast_hours: int = 24,
        metric_name: str = "",
    ) -> ForecastResult:
        """预测单个指标

        自动选择最佳方法:
        - 数据 > 2 周: Holt-Winters (周季节性)
        - 数据 > 2 天: Holt-Winters (日季节性)
        - 数据 < 2 天: 线性趋势
        """
        n = len(data)
        forecast_steps = int(forecast_hours * 3600 / self.interval)

        if n >= 2 * self.points_per_week:
            logger.info(f"Using Holt-Winters (weekly) for {metric_name}")
            model = HoltWintersForecaster(seasonal_period=self.points_per_week)
        elif n >= 2 * self.points_per_day:
            logger.info(f"Using Holt-Winters (daily) for {metric_name}")
            model = HoltWintersForecaster(seasonal_period=self.points_per_day)
        else:
            logger.info(f"Using linear trend for {metric_name} (insufficient data)")
            model = LinearTrendForecaster()

        model.fit(data)
        return model.predict(forecast_steps)

    def forecast_capacity_breach(
        self,
        data: np.ndarray,
        threshold: float,
        metric_name: str = "",
    ) -> Optional[float]:
        """预测指标何时达到阈值

        Returns:
            预计达到阈值的小时数, None 表示不会达到
        """
        linear = LinearTrendForecaster()
        linear.fit(data)
        steps = linear.time_to_threshold(threshold)
        if steps is None:
            return None
        hours = steps * self.interval / 3600
        return round(hours, 1)


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Traffic Forecaster Demo ===\n")

    # 模拟 3 天的 KV Cache 使用率 (15 分钟间隔)
    points_per_day = 96
    days = 3
    n = points_per_day * days

    t = np.arange(n)
    # 基线 + 日季节性 + 上升趋势 + 噪声
    baseline = 0.6
    trend = 0.0005 * t                        # 缓慢上升
    daily_season = 0.1 * np.sin(2 * np.pi * t / points_per_day)  # 日周期
    noise = np.random.normal(0, 0.02, n)
    data = baseline + trend + daily_season + noise
    data = np.clip(data, 0, 1)

    print(f"Data: {n} points ({days} days)")
    print(f"Current KV Cache: {data[-1]:.3f}")

    # 预测未来 24 小时
    forecaster = CapacityForecaster(sampling_interval_s=900)
    result = forecaster.forecast_metric(data, forecast_hours=24, metric_name="kv_cache")

    print(f"\nForecast (next 24h):")
    print(f"  Method: {result.method}")
    print(f"  +6h:  {result.values[24]:.3f} [{result.lower_bound[24]:.3f}, {result.upper_bound[24]:.3f}]")
    print(f"  +12h: {result.values[48]:.3f} [{result.lower_bound[48]:.3f}, {result.upper_bound[48]:.3f}]")
    print(f"  +24h: {result.values[-1]:.3f} [{result.lower_bound[-1]:.3f}, {result.upper_bound[-1]:.3f}]")

    # 何时达到 0.9?
    hours_to_90 = forecaster.forecast_capacity_breach(data, threshold=0.9, metric_name="kv_cache")
    if hours_to_90:
        print(f"\n预测 KV Cache 将在 {hours_to_90:.1f} 小时后达到 90%")
    else:
        print("\n当前趋势下 KV Cache 不会达到 90%")
