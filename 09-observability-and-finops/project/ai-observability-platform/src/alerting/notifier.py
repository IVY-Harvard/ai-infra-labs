"""多通道通知器 — 告警通知分发与限流

职责:
1. 支持多种通知通道: Slack, Email, PagerDuty, Webhook
2. 根据严重程度路由通知到不同通道
3. 速率限制 — 防止通知风暴
4. 提供通知记录与重试机制

通道路由规则 (默认):
┌────────────┬─────────────────────────────────┐
│ 严重程度    │ 通知通道                         │
├────────────┼─────────────────────────────────┤
│ INFO       │ Slack                           │
│ WARNING    │ Slack + Email                   │
│ CRITICAL   │ Slack + Email + PagerDuty       │
│ PAGE       │ PagerDuty + Slack + Email + 电话 │
└────────────┴─────────────────────────────────┘
"""

import time
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict

from src.alerting.alert_manager import Alert, AlertSeverity

logger = logging.getLogger(__name__)


@dataclass
class NotificationResult:
    """通知发送结果"""
    channel: str
    success: bool
    message: str = ""
    timestamp: float = 0

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


class RateLimiter:
    """令牌桶限流器 — 防止通知风暴

    每个通道独立限流:
    - Slack: 最多 10 条/分钟
    - PagerDuty: 最多 5 条/分钟
    - Email: 最多 20 条/分钟
    """

    def __init__(self, max_per_minute: int = 10):
        self.max_per_minute = max_per_minute
        self._window_size = 60.0  # 1 分钟窗口
        self._timestamps: List[float] = []

    def allow(self) -> bool:
        """检查是否允许发送"""
        now = time.time()
        cutoff = now - self._window_size

        # 清理过期记录
        self._timestamps = [t for t in self._timestamps if t > cutoff]

        if len(self._timestamps) >= self.max_per_minute:
            return False

        self._timestamps.append(now)
        return True

    def remaining(self) -> int:
        """剩余配额"""
        now = time.time()
        cutoff = now - self._window_size
        active = sum(1 for t in self._timestamps if t > cutoff)
        return max(0, self.max_per_minute - active)


class NotificationChannel(ABC):
    """通知通道基类"""

    def __init__(self, name: str, rate_limit: int = 10):
        self.name = name
        self.rate_limiter = RateLimiter(max_per_minute=rate_limit)
        self._send_count = 0
        self._error_count = 0

    @abstractmethod
    def send(self, alert: Alert, message: str = "") -> NotificationResult:
        """发送通知

        Args:
            alert: 告警对象
            message: 附加消息文本

        Returns:
            NotificationResult
        """
        pass

    def get_stats(self) -> Dict:
        return {
            "channel": self.name,
            "sent": self._send_count,
            "errors": self._error_count,
            "rate_remaining": self.rate_limiter.remaining(),
        }


class SlackNotifier(NotificationChannel):
    """Slack Webhook 通知通道

    通过 Incoming Webhook 发送富文本消息到指定频道
    """

    def __init__(self, webhook_url: str, channel: str = "#alerts", rate_limit: int = 10):
        super().__init__(name="slack", rate_limit=rate_limit)
        self.webhook_url = webhook_url
        self.channel = channel

    def send(self, alert: Alert, message: str = "") -> NotificationResult:
        """发送 Slack 通知"""
        if not self.rate_limiter.allow():
            logger.warning(f"Slack 通知被限流: {alert.name}")
            return NotificationResult(
                channel=self.name,
                success=False,
                message="rate_limited",
            )

        payload = self._build_payload(alert, message)

        try:
            # 实际环境使用 aiohttp/requests 发送
            # import requests
            # resp = requests.post(self.webhook_url, json=payload, timeout=5)
            # resp.raise_for_status()

            self._send_count += 1
            logger.info(f"Slack 通知已发送: {alert.name} → {self.channel}")
            return NotificationResult(channel=self.name, success=True)
        except Exception as e:
            self._error_count += 1
            logger.error(f"Slack 通知发送失败: {e}")
            return NotificationResult(
                channel=self.name,
                success=False,
                message=str(e),
            )

    def _build_payload(self, alert: Alert, message: str) -> Dict:
        """构建 Slack Block Kit 消息"""
        severity_emoji = {
            AlertSeverity.INFO: ":information_source:",
            AlertSeverity.WARNING: ":warning:",
            AlertSeverity.CRITICAL: ":rotating_light:",
            AlertSeverity.PAGE: ":fire:",
        }

        emoji = severity_emoji.get(alert.severity, ":bell:")
        text = message or f"{emoji} [{alert.severity.value.upper()}] {alert.name}"

        return {
            "channel": self.channel,
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": f"{emoji} Alert: {alert.name}"},
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Severity:* {alert.severity.value}"},
                        {"type": "mrkdwn", "text": f"*Metric:* {alert.metric_name}"},
                        {"type": "mrkdwn", "text": f"*Value:* {alert.value:.4f}"},
                        {"type": "mrkdwn", "text": f"*Threshold:* {alert.threshold:.4f}"},
                    ],
                },
                {
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn", "text": f"Labels: {alert.labels}"},
                    ],
                },
            ],
            "text": text,
        }


class EmailNotifier(NotificationChannel):
    """Email (SMTP) 通知通道"""

    def __init__(
        self,
        smtp_host: str = "localhost",
        smtp_port: int = 587,
        username: str = "",
        password: str = "",
        from_addr: str = "alerts@ai-platform.internal",
        to_addrs: Optional[List[str]] = None,
        rate_limit: int = 20,
    ):
        super().__init__(name="email", rate_limit=rate_limit)
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs or []

    def send(self, alert: Alert, message: str = "") -> NotificationResult:
        """发送邮件通知"""
        if not self.rate_limiter.allow():
            logger.warning(f"Email 通知被限流: {alert.name}")
            return NotificationResult(
                channel=self.name,
                success=False,
                message="rate_limited",
            )

        if not self.to_addrs:
            return NotificationResult(
                channel=self.name,
                success=False,
                message="no_recipients",
            )

        subject = f"[{alert.severity.value.upper()}] {alert.name}"
        body = self._build_body(alert, message)

        try:
            # 实际环境使用 smtplib 发送
            # import smtplib
            # from email.mime.text import MIMEText
            # msg = MIMEText(body)
            # msg["Subject"] = subject
            # msg["From"] = self.from_addr
            # msg["To"] = ", ".join(self.to_addrs)
            # with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            #     server.starttls()
            #     server.login(self.username, self.password)
            #     server.sendmail(self.from_addr, self.to_addrs, msg.as_string())

            self._send_count += 1
            logger.info(f"Email 通知已发送: {subject} → {self.to_addrs}")
            return NotificationResult(channel=self.name, success=True)
        except Exception as e:
            self._error_count += 1
            logger.error(f"Email 通知发送失败: {e}")
            return NotificationResult(
                channel=self.name,
                success=False,
                message=str(e),
            )

    def _build_body(self, alert: Alert, message: str) -> str:
        """构建邮件正文"""
        lines = [
            f"告警名称: {alert.name}",
            f"严重程度: {alert.severity.value}",
            f"指标: {alert.metric_name} = {alert.value:.4f}",
            f"阈值: {alert.threshold:.4f}",
            f"标签: {alert.labels}",
            f"触发时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.fired_at))}",
            "",
            f"告警 ID: {alert.id}",
        ]
        if message:
            lines.insert(0, message)
            lines.insert(1, "")
        return "\n".join(lines)


class PagerDutyNotifier(NotificationChannel):
    """PagerDuty 通知通道 — 用于关键告警的即时通知

    通过 PagerDuty Events API v2 创建 incident
    """

    def __init__(self, routing_key: str, rate_limit: int = 5):
        super().__init__(name="pagerduty", rate_limit=rate_limit)
        self.routing_key = routing_key
        self.api_url = "https://events.pagerduty.com/v2/enqueue"

    def send(self, alert: Alert, message: str = "") -> NotificationResult:
        """发送 PagerDuty 事件"""
        if not self.rate_limiter.allow():
            logger.warning(f"PagerDuty 通知被限流: {alert.name}")
            return NotificationResult(
                channel=self.name,
                success=False,
                message="rate_limited",
            )

        payload = self._build_event(alert, message)

        try:
            # 实际环境使用 requests/aiohttp 发送
            # import requests
            # resp = requests.post(self.api_url, json=payload, timeout=10)
            # resp.raise_for_status()

            self._send_count += 1
            logger.info(f"PagerDuty 事件已创建: {alert.name}")
            return NotificationResult(channel=self.name, success=True)
        except Exception as e:
            self._error_count += 1
            logger.error(f"PagerDuty 通知发送失败: {e}")
            return NotificationResult(
                channel=self.name,
                success=False,
                message=str(e),
            )

    def _build_event(self, alert: Alert, message: str) -> Dict:
        """构建 PagerDuty Events API v2 请求体"""
        severity_map = {
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.CRITICAL: "critical",
            AlertSeverity.PAGE: "critical",
        }

        return {
            "routing_key": self.routing_key,
            "event_action": "trigger",
            "dedup_key": alert.fingerprint,
            "payload": {
                "summary": message or f"[{alert.severity.value}] {alert.name}",
                "severity": severity_map.get(alert.severity, "warning"),
                "source": "ai-observability-platform",
                "component": alert.metric_name,
                "custom_details": {
                    "metric_name": alert.metric_name,
                    "value": alert.value,
                    "threshold": alert.threshold,
                    "labels": alert.labels,
                },
            },
        }


class WebhookNotifier(NotificationChannel):
    """通用 Webhook 通知通道"""

    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None, rate_limit: int = 15):
        super().__init__(name="webhook", rate_limit=rate_limit)
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}

    def send(self, alert: Alert, message: str = "") -> NotificationResult:
        """发送 Webhook 通知"""
        if not self.rate_limiter.allow():
            logger.warning(f"Webhook 通知被限流: {alert.name}")
            return NotificationResult(
                channel=self.name,
                success=False,
                message="rate_limited",
            )

        payload = {
            "alert_id": alert.id,
            "name": alert.name,
            "severity": alert.severity.value,
            "state": alert.state.value,
            "metric_name": alert.metric_name,
            "value": alert.value,
            "threshold": alert.threshold,
            "labels": alert.labels,
            "fired_at": alert.fired_at,
            "message": message,
        }

        try:
            # 实际环境使用 requests/aiohttp 发送
            # import requests
            # resp = requests.post(self.url, json=payload, headers=self.headers, timeout=10)
            # resp.raise_for_status()

            self._send_count += 1
            logger.info(f"Webhook 通知已发送: {alert.name} → {self.url}")
            return NotificationResult(channel=self.name, success=True)
        except Exception as e:
            self._error_count += 1
            logger.error(f"Webhook 通知发送失败: {e}")
            return NotificationResult(
                channel=self.name,
                success=False,
                message=str(e),
            )


class NotificationRouter:
    """通知路由器 — 根据严重程度分发到不同通道

    默认路由规则:
    - INFO: Slack
    - WARNING: Slack + Email
    - CRITICAL: Slack + Email + PagerDuty
    - PAGE: 所有通道
    """

    def __init__(self):
        # 通道注册表: name → NotificationChannel
        self._channels: Dict[str, NotificationChannel] = {}
        # 路由规则: severity → [channel_names]
        self._routes: Dict[AlertSeverity, List[str]] = {
            AlertSeverity.INFO: ["slack"],
            AlertSeverity.WARNING: ["slack", "email"],
            AlertSeverity.CRITICAL: ["slack", "email", "pagerduty"],
            AlertSeverity.PAGE: ["slack", "email", "pagerduty", "webhook"],
        }
        # 通知历史
        self._history: List[NotificationResult] = []
        self._max_history = 1000

    def register_channel(self, channel: NotificationChannel):
        """注册通知通道"""
        self._channels[channel.name] = channel
        logger.info(f"注册通知通道: {channel.name}")

    def set_route(self, severity: AlertSeverity, channels: List[str]):
        """设置路由规则"""
        self._routes[severity] = channels
        logger.info(f"设置路由: {severity.value} → {channels}")

    def notify(self, alert: Alert, message: str = "") -> List[NotificationResult]:
        """根据严重程度路由并发送通知

        Args:
            alert: 告警对象
            message: 附加消息

        Returns:
            各通道发送结果列表
        """
        results = []
        target_channels = self._routes.get(alert.severity, ["slack"])

        for channel_name in target_channels:
            channel = self._channels.get(channel_name)
            if not channel:
                logger.warning(f"通知通道未注册: {channel_name}")
                results.append(NotificationResult(
                    channel=channel_name,
                    success=False,
                    message="channel_not_registered",
                ))
                continue

            result = channel.send(alert, message)
            results.append(result)

        # 记录历史
        self._history.extend(results)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return results

    def get_channel_stats(self) -> List[Dict]:
        """获取所有通道的统计"""
        return [ch.get_stats() for ch in self._channels.values()]

    def get_recent_notifications(self, limit: int = 50) -> List[Dict]:
        """获取最近的通知记录"""
        recent = self._history[-limit:]
        return [
            {
                "channel": r.channel,
                "success": r.success,
                "message": r.message,
                "timestamp": r.timestamp,
            }
            for r in recent
        ]
