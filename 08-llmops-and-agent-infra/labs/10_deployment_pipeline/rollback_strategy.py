"""
Lab 10: 自动回滚策略
基于指标监控的自动回滚机制
"""
import time
import random
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class RollbackSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    severity: RollbackSeverity
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MetricsMonitor:
    """指标监控器 - 滑动窗口统计"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics: dict[str, deque] = {}

    def record(self, name: str, value: float):
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=self.window_size)
        self.metrics[name].append(value)

    def get_stats(self, name: str) -> dict:
        if name not in self.metrics or not self.metrics[name]:
            return {}
        values = list(self.metrics[name])
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "p50": np.percentile(values, 50),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99),
            "count": len(values),
        }


class RollbackManager:
    """自动回滚管理器"""

    def __init__(self):
        self.monitor = MetricsMonitor(window_size=200)
        self.alerts: list[Alert] = []
        self.rollback_triggered = False
        self.current_version = "v2.0"
        self.stable_version = "v1.0"

        # 回滚规则
        self.rules = {
            "error_rate": {
                "warning": 0.03,
                "critical": 0.05,
                "window": 50,
            },
            "quality_score": {
                "warning": 0.75,  # 低于此值告警
                "critical": 0.65,  # 低于此值回滚
                "direction": "below",
            },
            "latency_p95": {
                "warning": 2500,
                "critical": 5000,
            },
            "hallucination_rate": {
                "warning": 0.08,
                "critical": 0.15,
            },
        }

    def ingest_request(self, quality: float, latency: float,
                       is_error: bool, is_hallucination: bool):
        """接收请求指标"""
        self.monitor.record("quality_score", quality)
        self.monitor.record("latency", latency)
        self.monitor.record("error", 1.0 if is_error else 0.0)
        self.monitor.record("hallucination", 1.0 if is_hallucination else 0.0)

    def check_rules(self) -> list[Alert]:
        """检查所有规则"""
        new_alerts = []

        # 错误率检查
        error_stats = self.monitor.get_stats("error")
        if error_stats and error_stats["count"] >= 50:
            error_rate = error_stats["mean"]
            if error_rate > self.rules["error_rate"]["critical"]:
                new_alerts.append(Alert(
                    metric="error_rate", value=error_rate,
                    threshold=self.rules["error_rate"]["critical"],
                    severity=RollbackSeverity.CRITICAL,
                    message=f"错误率 {error_rate:.2%} 超过阈值 {self.rules['error_rate']['critical']:.2%}",
                ))
            elif error_rate > self.rules["error_rate"]["warning"]:
                new_alerts.append(Alert(
                    metric="error_rate", value=error_rate,
                    threshold=self.rules["error_rate"]["warning"],
                    severity=RollbackSeverity.WARNING,
                    message=f"错误率 {error_rate:.2%} 接近阈值",
                ))

        # 质量分数检查
        quality_stats = self.monitor.get_stats("quality_score")
        if quality_stats and quality_stats["count"] >= 50:
            avg_quality = quality_stats["mean"]
            if avg_quality < self.rules["quality_score"]["critical"]:
                new_alerts.append(Alert(
                    metric="quality_score", value=avg_quality,
                    threshold=self.rules["quality_score"]["critical"],
                    severity=RollbackSeverity.CRITICAL,
                    message=f"质量分数 {avg_quality:.3f} 低于阈值",
                ))

        # 延迟检查
        latency_stats = self.monitor.get_stats("latency")
        if latency_stats:
            p95 = latency_stats["p95"]
            if p95 > self.rules["latency_p95"]["critical"]:
                new_alerts.append(Alert(
                    metric="latency_p95", value=p95,
                    threshold=self.rules["latency_p95"]["critical"],
                    severity=RollbackSeverity.CRITICAL,
                    message=f"P95 延迟 {p95:.0f}ms 超过阈值",
                ))

        # 幻觉率检查
        hallucination_stats = self.monitor.get_stats("hallucination")
        if hallucination_stats and hallucination_stats["count"] >= 50:
            hallu_rate = hallucination_stats["mean"]
            if hallu_rate > self.rules["hallucination_rate"]["critical"]:
                new_alerts.append(Alert(
                    metric="hallucination_rate", value=hallu_rate,
                    threshold=self.rules["hallucination_rate"]["critical"],
                    severity=RollbackSeverity.CRITICAL,
                    message=f"幻觉率 {hallu_rate:.2%} 超过阈值",
                ))

        self.alerts.extend(new_alerts)
        return new_alerts

    def should_rollback(self) -> tuple[bool, str]:
        """判断是否需要回滚"""
        critical_alerts = [a for a in self.alerts[-10:]
                          if a.severity == RollbackSeverity.CRITICAL]
        if len(critical_alerts) >= 2:
            reasons = [a.message for a in critical_alerts[:3]]
            return True, "; ".join(reasons)
        return False, ""

    def execute_rollback(self, reason: str):
        """执行回滚"""
        print(f"\n{'!'*60}")
        print(f"自动回滚触发!")
        print(f"原因: {reason}")
        print(f"操作: {self.current_version} → {self.stable_version}")
        print(f"{'!'*60}")

        self.current_version = self.stable_version
        self.rollback_triggered = True


def simulate():
    """模拟自动回滚场景"""
    print("=" * 60)
    print("自动回滚策略模拟")
    print("=" * 60)

    manager = RollbackManager()

    # Phase 1: 正常运行
    print("\n--- Phase 1: 正常运行 (200 请求) ---")
    for i in range(200):
        manager.ingest_request(
            quality=0.88 + random.gauss(0, 0.05),
            latency=1200 + random.gauss(0, 200),
            is_error=random.random() < 0.02,
            is_hallucination=random.random() < 0.05,
        )
    alerts = manager.check_rules()
    stats = manager.monitor.get_stats("quality_score")
    print(f"  质量: {stats['mean']:.3f}, 告警: {len(alerts)}")

    # Phase 2: 质量开始下降
    print("\n--- Phase 2: 质量下降 (100 请求) ---")
    for i in range(100):
        manager.ingest_request(
            quality=0.70 + random.gauss(0, 0.08),  # 质量下降
            latency=2000 + random.gauss(0, 500),     # 延迟升高
            is_error=random.random() < 0.06,          # 错误率升高
            is_hallucination=random.random() < 0.12,  # 幻觉率升高
        )

    alerts = manager.check_rules()
    print(f"  新增告警: {len(alerts)}")
    for alert in alerts:
        print(f"    [{alert.severity.value}] {alert.message}")

    should_rollback, reason = manager.should_rollback()
    if should_rollback:
        manager.execute_rollback(reason)
    else:
        print("  暂未触发回滚")

    # Phase 3: 继续恶化（如果未回滚）
    if not manager.rollback_triggered:
        print("\n--- Phase 3: 继续恶化 (50 请求) ---")
        for i in range(50):
            manager.ingest_request(
                quality=0.55 + random.gauss(0, 0.1),
                latency=3000 + random.gauss(0, 800),
                is_error=random.random() < 0.10,
                is_hallucination=random.random() < 0.20,
            )
        alerts = manager.check_rules()
        should_rollback, reason = manager.should_rollback()
        if should_rollback:
            manager.execute_rollback(reason)

    # 总结
    print(f"\n{'='*60}")
    print("监控总结")
    print(f"{'='*60}")
    print(f"  总告警数: {len(manager.alerts)}")
    print(f"  Critical: {sum(1 for a in manager.alerts if a.severity == RollbackSeverity.CRITICAL)}")
    print(f"  Warning: {sum(1 for a in manager.alerts if a.severity == RollbackSeverity.WARNING)}")
    print(f"  回滚触发: {'是' if manager.rollback_triggered else '否'}")
    print(f"  当前版本: {manager.current_version}")


if __name__ == "__main__":
    simulate()
