"""告警管理器 — 告警生命周期管理与路由

职责:
1. 接收异常检测引擎的告警
2. 去重 & 聚合 (同一 metric 在 suppression window 内只告警一次)
3. 根据路由规则分发到不同通道
4. 维护告警状态 (firing / acknowledged / resolved)

告警流转状态机:
    ┌──────────┐    ack     ┌──────────────┐
    │ FIRING   │───────────▶│ ACKNOWLEDGED │
    └──────────┘            └──────────────┘
         │                         │
         │  resolve                │ resolve
         ▼                         ▼
    ┌──────────┐            ┌──────────────┐
    │ RESOLVED │◀───────────│   RESOLVED   │
    └──────────┘            └──────────────┘
"""

import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class AlertState(str, Enum):
    FIRING = "firing"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    PAGE = "page"  # 需要立即响应的紧急告警


@dataclass
class Alert:
    """告警对象"""
    id: str = ""
    name: str = ""
    severity: AlertSeverity = AlertSeverity.WARNING
    state: AlertState = AlertState.FIRING
    metric_name: str = ""
    value: float = 0
    threshold: float = 0
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    fired_at: float = 0
    resolved_at: float = 0
    acknowledged_by: str = ""
    fingerprint: str = ""  # 用于去重的指纹

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
        if not self.fired_at:
            self.fired_at = time.time()
        if not self.fingerprint:
            # 使用 name + labels 生成指纹
            label_str = "&".join(f"{k}={v}" for k, v in sorted(self.labels.items()))
            self.fingerprint = f"{self.name}:{label_str}"


@dataclass
class RouteRule:
    """路由规则 — 决定告警发送到哪个通道"""
    name: str
    match_severity: Optional[List[AlertSeverity]] = None
    match_labels: Optional[Dict[str, str]] = None
    channels: List[str] = field(default_factory=list)
    # 例: channels = ["feishu_oncall", "pagerduty", "email_team"]


class AlertManager:
    """告警管理器"""

    def __init__(
        self,
        suppression_window: float = 300.0,  # 5 分钟内同一告警不重复发送
        auto_resolve_after: float = 600.0,  # 10 分钟无更新自动 resolve
    ):
        self.suppression_window = suppression_window
        self.auto_resolve_after = auto_resolve_after

        # 告警存储: fingerprint → Alert
        self._alerts: Dict[str, Alert] = {}
        # 上次触发时间: fingerprint → timestamp
        self._last_fired: Dict[str, float] = {}
        # 路由规则
        self._routes: List[RouteRule] = []
        # 通知回调: channel_name → callback
        self._notifiers: Dict[str, Callable] = {}
        # 告警计数 (用于 metrics)
        self._stats: Dict[str, int] = defaultdict(int)

    def add_route(self, route: RouteRule):
        """添加路由规则"""
        self._routes.append(route)
        logger.info(f"添加路由规则: {route.name} → {route.channels}")

    def register_notifier(self, channel: str, callback: Callable):
        """注册通知通道回调"""
        self._notifiers[channel] = callback
        logger.info(f"注册通知通道: {channel}")

    def fire(self, alert: Alert) -> bool:
        """触发告警

        Returns:
            True 如果实际发送了通知, False 如果被抑制
        """
        now = time.time()

        # 去重检查: suppression window 内不重复发送
        if alert.fingerprint in self._last_fired:
            elapsed = now - self._last_fired[alert.fingerprint]
            if elapsed < self.suppression_window:
                logger.debug(
                    f"告警被抑制 (距上次 {elapsed:.0f}s < {self.suppression_window}s): "
                    f"{alert.name}"
                )
                self._stats["suppressed"] += 1
                return False

        # 存储告警
        alert.state = AlertState.FIRING
        alert.fired_at = now
        self._alerts[alert.fingerprint] = alert
        self._last_fired[alert.fingerprint] = now
        self._stats["fired"] += 1

        logger.warning(
            f"告警触发: [{alert.severity.value}] {alert.name} "
            f"| {alert.metric_name}={alert.value} (threshold={alert.threshold}) "
            f"| labels={alert.labels}"
        )

        # 路由 & 通知
        self._route_and_notify(alert)
        return True

    def acknowledge(self, fingerprint: str, user: str) -> bool:
        """确认告警"""
        if fingerprint not in self._alerts:
            return False

        alert = self._alerts[fingerprint]
        if alert.state == AlertState.RESOLVED:
            return False

        alert.state = AlertState.ACKNOWLEDGED
        alert.acknowledged_by = user
        self._stats["acknowledged"] += 1
        logger.info(f"告警已确认: {alert.name} by {user}")
        return True

    def resolve(self, fingerprint: str) -> bool:
        """解除告警"""
        if fingerprint not in self._alerts:
            return False

        alert = self._alerts[fingerprint]
        alert.state = AlertState.RESOLVED
        alert.resolved_at = time.time()
        self._stats["resolved"] += 1
        logger.info(f"告警已解除: {alert.name}")
        return True

    def get_firing_alerts(self) -> List[Alert]:
        """获取所有正在触发的告警"""
        return [a for a in self._alerts.values() if a.state == AlertState.FIRING]

    def get_all_alerts(self, limit: int = 100) -> List[Alert]:
        """获取所有告警 (按时间倒序)"""
        alerts = sorted(self._alerts.values(), key=lambda a: a.fired_at, reverse=True)
        return alerts[:limit]

    def cleanup_resolved(self, max_age: float = 86400):
        """清理已解除超过 max_age 秒的告警"""
        now = time.time()
        to_remove = [
            fp for fp, alert in self._alerts.items()
            if alert.state == AlertState.RESOLVED
            and alert.resolved_at > 0
            and (now - alert.resolved_at) > max_age
        ]
        for fp in to_remove:
            del self._alerts[fp]
        if to_remove:
            logger.info(f"清理已解除告警: {len(to_remove)} 条")

    def auto_resolve_stale(self):
        """自动解除超时未更新的告警"""
        now = time.time()
        for fp, alert in self._alerts.items():
            if alert.state == AlertState.FIRING:
                last_fire = self._last_fired.get(fp, alert.fired_at)
                if (now - last_fire) > self.auto_resolve_after:
                    self.resolve(fp)
                    logger.info(f"自动解除超时告警: {alert.name}")

    def get_stats(self) -> Dict:
        """获取告警统计"""
        return {
            "total_firing": len(self.get_firing_alerts()),
            "total_alerts": len(self._alerts),
            **dict(self._stats),
        }

    def _route_and_notify(self, alert: Alert):
        """根据路由规则分发通知"""
        matched_channels = set()

        for route in self._routes:
            if self._match_route(alert, route):
                matched_channels.update(route.channels)

        # 如果没有匹配任何路由, 使用默认通道
        if not matched_channels:
            matched_channels = {"default"}

        for channel in matched_channels:
            if channel in self._notifiers:
                try:
                    self._notifiers[channel](alert)
                except Exception as e:
                    logger.error(f"通知发送失败 [{channel}]: {e}")
            else:
                logger.warning(f"通知通道未注册: {channel}")

    def _match_route(self, alert: Alert, route: RouteRule) -> bool:
        """检查告警是否匹配路由规则"""
        # 匹配严重程度
        if route.match_severity and alert.severity not in route.match_severity:
            return False

        # 匹配 labels
        if route.match_labels:
            for key, value in route.match_labels.items():
                if alert.labels.get(key) != value:
                    return False

        return True
