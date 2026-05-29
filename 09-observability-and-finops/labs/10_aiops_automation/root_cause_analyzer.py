"""
告警根因分析器 — 从告警风暴中找到真正的根因
=============================================

问题: 一个 GPU 故障可能同时触发 20+ 条告警
解决: 通过拓扑图和时序分析将告警聚合为根因事件

依赖: numpy
"""

import time
import logging
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """告警"""
    alert_name: str
    severity: str           # "info" | "warning" | "critical"
    instance: str
    labels: Dict = field(default_factory=dict)
    annotations: Dict = field(default_factory=dict)
    fired_at: float = field(default_factory=time.time)
    value: float = 0


@dataclass
class RootCauseIncident:
    """根因事件 (聚合多条告警)"""
    incident_id: str
    root_cause: str
    root_alert: Alert
    correlated_alerts: List[Alert] = field(default_factory=list)
    severity: str = "warning"
    impact_description: str = ""
    recommended_actions: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


# 告警关联规则 (基于领域知识)
ALERT_CORRELATION_RULES = {
    # 根因 → 可能派生的告警
    "GPUECCErrorsUncorrectable": [
        "GPUXidError",
        "VLLMServiceDown",
        "GenerationThroughputZero",
        "RequestFailureRateCritical",
        "TTFTp99Critical",
    ],
    "GPUThermalThrottleActive": [
        "GPUTemperatureHigh",
        "GPUClockAbnormallyLow",
        "TPOTp99ExceedsSLO",
        "ThroughputDecliningTrend",
    ],
    "KVCacheNearlyFull": [
        "KVCacheUsageHigh",
        "PreemptionRateHigh",
        "TTFTp99ExceedsSLO",
        "TTFTp99Critical",
        "RequestQueueBacklog",
        "RequestQueueCritical",
        "KVCacheSwapActive",
    ],
    "VLLMServiceDown": [
        "GenerationThroughputZero",
        "GPUNotResponding",
        "RequestFailureRateCritical",
    ],
    "NVLinkBandwidthLow": [
        "TPOTp99ExceedsSLO",
        "ThroughputDecliningTrend",
    ],
    "RequestVolumeGrowthHigh": [
        "RequestQueueBacklog",
        "KVCacheUsageHigh",
        "TTFTp99ExceedsSLO",
        "PreemptionRateHigh",
    ],
}


class RootCauseAnalyzer:
    """告警根因分析器

    分析步骤:
    1. 时间窗口聚合: 5 分钟内同一实例的告警视为相关
    2. 拓扑关联: 基于因果图判断哪些告警是同一根因的派生
    3. 根因排序: 最上游 + 最早触发 = 最可能的根因
    4. 事件生成: 将 N 条告警聚合为 1 个根因事件
    """

    def __init__(self, correlation_window_s: float = 300):
        self.correlation_window = correlation_window_s
        self._pending_alerts: List[Alert] = []
        self._incidents: List[RootCauseIncident] = []
        self._incident_counter = 0

    def ingest_alert(self, alert: Alert) -> Optional[RootCauseIncident]:
        """摄入告警, 判断是否形成新的根因事件

        Returns:
            如果形成新事件则返回, 否则 None
        """
        self._pending_alerts.append(alert)

        # 清理过期告警
        now = time.time()
        self._pending_alerts = [
            a for a in self._pending_alerts
            if now - a.fired_at < self.correlation_window
        ]

        # 尝试关联分析
        incident = self._try_correlate(alert)
        if incident:
            self._incidents.append(incident)
            # 从 pending 中移除已关联的告警
            correlated_names = {a.alert_name for a in incident.correlated_alerts}
            correlated_names.add(incident.root_alert.alert_name)
            self._pending_alerts = [
                a for a in self._pending_alerts
                if a.alert_name not in correlated_names or a.instance != alert.instance
            ]
            return incident

        return None

    def _try_correlate(self, trigger_alert: Alert) -> Optional[RootCauseIncident]:
        """尝试将新告警与 pending 告警关联"""
        # 查找该实例同时间窗口内的所有告警
        instance_alerts = [
            a for a in self._pending_alerts
            if a.instance == trigger_alert.instance
            and abs(a.fired_at - trigger_alert.fired_at) < self.correlation_window
        ]

        if len(instance_alerts) < 2:
            return None  # 单条告警不需要关联

        # 基于规则查找根因
        alert_names = {a.alert_name for a in instance_alerts}
        best_root = None
        best_coverage = 0

        for root_name, derived_names in ALERT_CORRELATION_RULES.items():
            if root_name in alert_names:
                coverage = len(alert_names.intersection(set(derived_names)))
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_root = root_name

        if best_root and best_coverage >= 1:
            # 找到根因告警
            root_alert = next(
                (a for a in instance_alerts if a.alert_name == best_root),
                trigger_alert,
            )
            correlated = [a for a in instance_alerts if a.alert_name != best_root]

            self._incident_counter += 1
            incident = RootCauseIncident(
                incident_id=f"INC-{self._incident_counter:04d}",
                root_cause=self._describe_root_cause(best_root),
                root_alert=root_alert,
                correlated_alerts=correlated,
                severity=root_alert.severity,
                impact_description=self._describe_impact(best_root, correlated),
                recommended_actions=self._recommend_actions(best_root),
            )
            return incident

        # 没有匹配规则, 使用启发式: 最早的告警 = 根因
        if len(instance_alerts) >= 3:
            sorted_alerts = sorted(instance_alerts, key=lambda a: a.fired_at)
            self._incident_counter += 1
            return RootCauseIncident(
                incident_id=f"INC-{self._incident_counter:04d}",
                root_cause=f"多告警聚合 (最早: {sorted_alerts[0].alert_name})",
                root_alert=sorted_alerts[0],
                correlated_alerts=sorted_alerts[1:],
                severity=max(a.severity for a in instance_alerts),
                impact_description=f"{len(instance_alerts)} 条关联告警",
                recommended_actions=["人工排查 — 未匹配到已知模式"],
            )

        return None

    def _describe_root_cause(self, root_name: str) -> str:
        descriptions = {
            "GPUECCErrorsUncorrectable": "GPU 不可纠正 ECC 错误 — 硬件故障",
            "GPUThermalThrottleActive": "GPU 热限频 — 散热系统问题",
            "KVCacheNearlyFull": "KV Cache 耗尽 — 容量不足",
            "VLLMServiceDown": "vLLM 服务崩溃",
            "NVLinkBandwidthLow": "NVLink 通信异常 — 互联故障",
            "RequestVolumeGrowthHigh": "流量突增超过集群容量",
        }
        return descriptions.get(root_name, root_name)

    def _describe_impact(self, root_name: str, correlated: List[Alert]) -> str:
        if root_name == "GPUECCErrorsUncorrectable":
            return "GPU 计算可能不正确, 服务需要立即摘除"
        elif root_name == "KVCacheNearlyFull":
            return f"服务降级: TTFT 升高, 可能有 {len(correlated)} 个关联影响"
        elif root_name == "GPUThermalThrottleActive":
            return "GPU 性能下降 20-40%, 影响延迟和吞吐"
        return f"影响 {len(correlated)} 个关联指标"

    def _recommend_actions(self, root_name: str) -> List[str]:
        actions = {
            "GPUECCErrorsUncorrectable": [
                "1. 立即从 LB 摘除",
                "2. 运行 nvidia-smi -q 确认错误",
                "3. 尝试 GPU Reset",
                "4. 通知 Infra 安排 RMA",
            ],
            "GPUThermalThrottleActive": [
                "1. 检查机房温度和冷却系统",
                "2. 临时降低 max_num_seqs",
                "3. 检查 GPU 风扇状态",
            ],
            "KVCacheNearlyFull": [
                "1. 启用限流",
                "2. 触发 HPA 扩容",
                "3. 检查是否有异常长请求",
            ],
            "VLLMServiceDown": [
                "1. 检查 Pod 状态和日志",
                "2. 检查 GPU 状态",
                "3. 尝试重启 Pod",
            ],
        }
        return actions.get(root_name, ["人工排查"])

    def get_active_incidents(self) -> List[Dict]:
        """获取活跃事件"""
        return [
            {
                "id": i.incident_id,
                "root_cause": i.root_cause,
                "severity": i.severity,
                "instance": i.root_alert.instance,
                "alert_count": 1 + len(i.correlated_alerts),
                "impact": i.impact_description,
                "actions": i.recommended_actions,
                "created_at": i.created_at,
            }
            for i in self._incidents[-10:]
        ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    analyzer = RootCauseAnalyzer()

    print("=== Root Cause Analysis Demo ===\n")

    # 模拟: GPU ECC 错误引发连锁告警
    now = time.time()
    alerts = [
        Alert("GPUECCErrorsUncorrectable", "critical", "gpu-node-3", fired_at=now),
        Alert("GPUXidError", "critical", "gpu-node-3", fired_at=now + 2),
        Alert("GenerationThroughputZero", "critical", "gpu-node-3", fired_at=now + 10),
        Alert("RequestFailureRateCritical", "critical", "gpu-node-3", fired_at=now + 15),
        Alert("TTFTp99Critical", "warning", "gpu-node-3", fired_at=now + 20),
    ]

    print("Ingesting 5 alerts from gpu-node-3...")
    for alert in alerts:
        incident = analyzer.ingest_alert(alert)
        if incident:
            print(f"\n  ROOT CAUSE IDENTIFIED: [{incident.incident_id}]")
            print(f"    Root cause: {incident.root_cause}")
            print(f"    Root alert: {incident.root_alert.alert_name}")
            print(f"    Correlated: {len(incident.correlated_alerts)} alerts")
            print(f"    Impact: {incident.impact_description}")
            print(f"    Actions:")
            for a in incident.recommended_actions:
                print(f"      {a}")
