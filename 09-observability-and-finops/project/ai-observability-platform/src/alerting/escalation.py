"""告警升级策略引擎 — 基于时间和严重程度的自动升级

职责:
1. 定义多级升级策略 (L1 → L2 → L3)
2. 如果告警在指定时间内未被确认, 自动升级到下一级
3. 管理 On-Call 轮值表
4. 提供升级历史追踪

升级流程:
┌──────────┐  timeout  ┌──────────┐  timeout  ┌──────────┐
│  L1      │──────────▶│   L2     │──────────▶│   L3     │
│  On-Call  │           │  Leader  │           │  Manager │
└──────────┘           └──────────┘           └──────────┘
  5 min                   15 min                 final
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from enum import Enum

from src.alerting.alert_manager import Alert, AlertState, AlertSeverity

logger = logging.getLogger(__name__)


class EscalationLevel(int, Enum):
    """升级层级"""
    L1 = 1  # 一线值班人员
    L2 = 2  # 团队负责人
    L3 = 3  # 管理层 / P0 响应团队


@dataclass
class OnCallPerson:
    """值班人员"""
    name: str
    email: str
    phone: str = ""
    slack_id: str = ""
    team: str = ""


@dataclass
class EscalationPolicy:
    """升级策略 — 定义每一级的超时和通知目标

    示例:
        L1: 5 分钟无确认 → 升级到 L2
        L2: 15 分钟无确认 → 升级到 L3
        L3: 最终级别, 持续通知直到确认
    """
    name: str
    levels: Dict[EscalationLevel, "EscalationStep"] = field(default_factory=dict)
    applicable_severities: List[AlertSeverity] = field(default_factory=list)

    def __post_init__(self):
        if not self.applicable_severities:
            self.applicable_severities = [AlertSeverity.CRITICAL, AlertSeverity.PAGE]


@dataclass
class EscalationStep:
    """单个升级步骤的配置"""
    level: EscalationLevel
    timeout_seconds: float  # 超过此时间未确认则升级
    notify_targets: List[str] = field(default_factory=list)  # 通知目标标识
    repeat_interval: float = 300.0  # 重复通知间隔 (秒)


@dataclass
class EscalationRecord:
    """升级记录 — 追踪告警的升级历史"""
    alert_id: str
    alert_fingerprint: str
    current_level: EscalationLevel = EscalationLevel.L1
    escalated_at: float = 0
    last_notified_at: float = 0
    escalation_history: List[Dict] = field(default_factory=list)
    acknowledged: bool = False

    def __post_init__(self):
        if self.escalated_at == 0:
            self.escalated_at = time.time()
        if self.last_notified_at == 0:
            self.last_notified_at = time.time()


@dataclass
class OnCallSchedule:
    """On-Call 轮值表"""
    team: str
    current_oncall: Dict[EscalationLevel, List[OnCallPerson]] = field(default_factory=dict)
    rotation_interval_hours: float = 168.0  # 默认一周轮一次
    last_rotation: float = 0


class EscalationEngine:
    """告警升级引擎

    核心逻辑:
    1. 新告警进入时, 从 L1 开始
    2. 定期检查未确认告警, 超时则升级
    3. 升级时通知下一级 on-call 人员
    4. 到达最高级别后持续通知直到确认
    """

    def __init__(self):
        # 升级策略: policy_name → EscalationPolicy
        self._policies: Dict[str, EscalationPolicy] = {}
        # 活跃升级记录: alert_fingerprint → EscalationRecord
        self._active_records: Dict[str, EscalationRecord] = {}
        # On-Call 轮值: team → OnCallSchedule
        self._schedules: Dict[str, OnCallSchedule] = {}
        # 通知回调
        self._notify_callback: Optional[Callable] = None
        # 统计
        self._stats: Dict[str, int] = {
            "total_escalations": 0,
            "l1_resolved": 0,
            "l2_resolved": 0,
            "l3_resolved": 0,
        }

        # 创建默认策略
        self._setup_default_policies()

    def _setup_default_policies(self):
        """设置默认升级策略"""
        # 关键告警策略: 5 分钟 → 15 分钟 → 最终
        critical_policy = EscalationPolicy(
            name="critical_default",
            levels={
                EscalationLevel.L1: EscalationStep(
                    level=EscalationLevel.L1,
                    timeout_seconds=300,  # 5 分钟
                    notify_targets=["oncall_l1"],
                    repeat_interval=120,
                ),
                EscalationLevel.L2: EscalationStep(
                    level=EscalationLevel.L2,
                    timeout_seconds=900,  # 15 分钟
                    notify_targets=["oncall_l2", "oncall_l1"],
                    repeat_interval=300,
                ),
                EscalationLevel.L3: EscalationStep(
                    level=EscalationLevel.L3,
                    timeout_seconds=0,  # 最终级别, 不再升级
                    notify_targets=["oncall_l3", "oncall_l2"],
                    repeat_interval=600,
                ),
            },
            applicable_severities=[AlertSeverity.CRITICAL, AlertSeverity.PAGE],
        )

        # 警告级别策略: 较宽松的超时
        warning_policy = EscalationPolicy(
            name="warning_default",
            levels={
                EscalationLevel.L1: EscalationStep(
                    level=EscalationLevel.L1,
                    timeout_seconds=1800,  # 30 分钟
                    notify_targets=["oncall_l1"],
                    repeat_interval=600,
                ),
                EscalationLevel.L2: EscalationStep(
                    level=EscalationLevel.L2,
                    timeout_seconds=3600,  # 1 小时
                    notify_targets=["oncall_l2"],
                    repeat_interval=1800,
                ),
            },
            applicable_severities=[AlertSeverity.WARNING],
        )

        self._policies["critical_default"] = critical_policy
        self._policies["warning_default"] = warning_policy

    def add_policy(self, policy: EscalationPolicy):
        """添加自定义升级策略"""
        self._policies[policy.name] = policy
        logger.info(f"添加升级策略: {policy.name}, 级别数: {len(policy.levels)}")

    def set_notify_callback(self, callback: Callable):
        """设置通知回调函数

        Args:
            callback: 接收 (alert, level, targets) 参数的回调
        """
        self._notify_callback = callback

    def set_oncall_schedule(self, schedule: OnCallSchedule):
        """设置 On-Call 轮值表"""
        self._schedules[schedule.team] = schedule
        logger.info(
            f"设置轮值表: team={schedule.team}, "
            f"levels={list(schedule.current_oncall.keys())}"
        )

    def register_alert(self, alert: Alert):
        """注册新告警到升级引擎

        告警进入后从 L1 开始, 立即通知一线值班人员
        """
        if alert.fingerprint in self._active_records:
            # 已存在的告警, 跳过
            return

        record = EscalationRecord(
            alert_id=alert.id,
            alert_fingerprint=alert.fingerprint,
            current_level=EscalationLevel.L1,
        )

        record.escalation_history.append({
            "level": EscalationLevel.L1.value,
            "timestamp": time.time(),
            "action": "initial",
        })

        self._active_records[alert.fingerprint] = record
        logger.info(f"告警注册到升级引擎: {alert.name} [{alert.severity.value}]")

        # 立即发送 L1 通知
        self._send_notification(alert, record)

    def acknowledge_alert(self, fingerprint: str, user: str):
        """标记告警为已确认, 停止升级"""
        if fingerprint not in self._active_records:
            return False

        record = self._active_records[fingerprint]
        record.acknowledged = True
        record.escalation_history.append({
            "level": record.current_level.value,
            "timestamp": time.time(),
            "action": "acknowledged",
            "user": user,
        })

        # 更新统计
        level_key = f"l{record.current_level.value}_resolved"
        if level_key in self._stats:
            self._stats[level_key] += 1

        logger.info(
            f"告警已确认, 停止升级: fingerprint={fingerprint}, "
            f"level=L{record.current_level.value}, user={user}"
        )
        return True

    def resolve_alert(self, fingerprint: str):
        """告警已解除, 从升级引擎中移除"""
        if fingerprint in self._active_records:
            del self._active_records[fingerprint]
            logger.info(f"告警已从升级引擎移除: {fingerprint}")

    def check_escalations(self, alerts: Dict[str, Alert]):
        """检查所有活跃告警是否需要升级

        此方法应由定时任务周期性调用 (建议每 30 秒)

        Args:
            alerts: fingerprint → Alert 的映射 (当前活跃告警)
        """
        now = time.time()

        for fingerprint, record in list(self._active_records.items()):
            # 跳过已确认的
            if record.acknowledged:
                continue

            # 获取对应告警
            alert = alerts.get(fingerprint)
            if not alert:
                continue

            # 如果告警已 resolved, 移除记录
            if alert.state == AlertState.RESOLVED:
                self.resolve_alert(fingerprint)
                continue

            # 获取适用策略
            policy = self._find_policy(alert)
            if not policy:
                continue

            # 获取当前级别配置
            current_step = policy.levels.get(record.current_level)
            if not current_step:
                continue

            # 检查是否需要升级
            time_in_level = now - record.escalated_at
            if current_step.timeout_seconds > 0 and time_in_level > current_step.timeout_seconds:
                self._escalate(alert, record, policy)
            else:
                # 检查是否需要重复通知
                time_since_notify = now - record.last_notified_at
                if time_since_notify > current_step.repeat_interval:
                    self._send_notification(alert, record)

    def _find_policy(self, alert: Alert) -> Optional[EscalationPolicy]:
        """为告警找到适用的升级策略"""
        for policy in self._policies.values():
            if alert.severity in policy.applicable_severities:
                return policy
        return None

    def _escalate(self, alert: Alert, record: EscalationRecord, policy: EscalationPolicy):
        """执行升级"""
        next_level_value = record.current_level.value + 1

        # 检查是否已到最高级别
        try:
            next_level = EscalationLevel(next_level_value)
        except ValueError:
            # 已是最高级别, 保持重复通知
            self._send_notification(alert, record)
            return

        if next_level not in policy.levels:
            # 策略中无此级别, 保持当前级别重复通知
            self._send_notification(alert, record)
            return

        # 升级
        record.current_level = next_level
        record.escalated_at = time.time()
        record.escalation_history.append({
            "level": next_level.value,
            "timestamp": time.time(),
            "action": "escalated",
        })

        self._stats["total_escalations"] += 1

        logger.warning(
            f"告警升级: {alert.name} L{next_level_value - 1} → L{next_level_value} "
            f"(未确认超时)"
        )

        # 发送升级通知
        self._send_notification(alert, record)

    def _send_notification(self, alert: Alert, record: EscalationRecord):
        """发送通知到当前级别的目标"""
        record.last_notified_at = time.time()

        policy = self._find_policy(alert)
        if not policy:
            return

        step = policy.levels.get(record.current_level)
        if not step:
            return

        if self._notify_callback:
            try:
                self._notify_callback(alert, record.current_level, step.notify_targets)
            except Exception as e:
                logger.error(f"升级通知发送失败: {e}")
        else:
            logger.warning(
                f"升级通知 (无回调): alert={alert.name}, "
                f"level=L{record.current_level.value}, "
                f"targets={step.notify_targets}"
            )

    def get_oncall_for_level(self, team: str, level: EscalationLevel) -> List[OnCallPerson]:
        """获取指定级别的当前值班人员"""
        schedule = self._schedules.get(team)
        if not schedule:
            return []
        return schedule.current_oncall.get(level, [])

    def get_active_escalations(self) -> List[Dict]:
        """获取所有活跃升级记录"""
        results = []
        for fp, record in self._active_records.items():
            results.append({
                "alert_id": record.alert_id,
                "fingerprint": fp,
                "current_level": record.current_level.value,
                "escalated_at": record.escalated_at,
                "acknowledged": record.acknowledged,
                "history_count": len(record.escalation_history),
            })
        return results

    def get_stats(self) -> Dict:
        """获取升级引擎统计"""
        return {
            "active_escalations": len([
                r for r in self._active_records.values() if not r.acknowledged
            ]),
            "total_tracked": len(self._active_records),
            **self._stats,
        }
