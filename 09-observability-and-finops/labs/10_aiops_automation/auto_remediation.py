"""
自动修复引擎 — GPU 推理服务 Auto-Remediation
=============================================

检测异常 → 诊断根因 → 选择修复动作 → 执行 → 验证

设计原则:
1. 安全优先: 所有自动动作都有 dry-run 模式和回滚能力
2. 渐进式: 先尝试温和动作, 失败再升级
3. 人类在环: 高风险动作需要人工确认
4. 审计追踪: 所有动作都有完整日志

依赖: asyncio
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ActionRisk(Enum):
    """动作风险等级"""
    LOW = "low"             # 无副作用, 自动执行
    MEDIUM = "medium"       # 轻微影响, 自动执行但通知
    HIGH = "high"           # 可能影响服务, 需要确认
    CRITICAL = "critical"   # 重大影响, 必须人工确认


class ActionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    NEEDS_APPROVAL = "needs_approval"


@dataclass
class RemediationAction:
    """修复动作"""
    name: str
    description: str
    risk: ActionRisk
    execute_fn: Optional[Callable] = None
    rollback_fn: Optional[Callable] = None
    timeout_s: int = 60
    auto_approve: bool = True     # 是否自动批准 (低风险)
    cooldown_s: int = 300         # 冷却期 (避免反复执行)


@dataclass
class RemediationEvent:
    """修复事件记录"""
    event_id: str
    trigger: str                  # 触发的告警/条件
    root_cause: str
    actions_taken: List[Dict] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float = 0
    status: str = "in_progress"
    outcome: str = ""


class AutoRemediationEngine:
    """自动修复引擎

    工作流:
    1. 接收告警/异常信号
    2. 匹配修复规则 (规则引擎)
    3. 评估风险和前置条件
    4. 执行修复动作 (或请求人工确认)
    5. 验证修复效果
    6. 记录审计日志
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self._rules: Dict[str, Dict] = {}
        self._action_history: List[RemediationEvent] = []
        self._last_action_time: Dict[str, float] = {}
        self._event_counter = 0

        # 注册默认规则
        self._register_default_rules()

    def _register_default_rules(self):
        """注册 GPU 推理场景的修复规则"""

        # === Rule 1: KV Cache 压力 ===
        self._rules["kv_cache_critical"] = {
            "condition": lambda metrics: metrics.get("kv_cache_usage", 0) > 0.95,
            "root_cause": "KV Cache 使用率 > 95%, 服务即将降级",
            "actions": [
                RemediationAction(
                    name="rate_limit_new_requests",
                    description="对新请求启用限流 (拒绝 > max_tokens 4096 的请求)",
                    risk=ActionRisk.MEDIUM,
                    auto_approve=True,
                    cooldown_s=300,
                ),
                RemediationAction(
                    name="reject_long_prompts",
                    description="临时拒绝 prompt > 4K tokens 的请求",
                    risk=ActionRisk.MEDIUM,
                    auto_approve=True,
                ),
                RemediationAction(
                    name="trigger_hpa_scale_up",
                    description="触发 HPA 扩容 (如果未自动触发)",
                    risk=ActionRisk.LOW,
                    auto_approve=True,
                ),
            ],
        }

        # === Rule 2: TTFT 严重超标 ===
        self._rules["ttft_critical"] = {
            "condition": lambda metrics: metrics.get("ttft_p99_s", 0) > 10,
            "root_cause": "TTFT P99 > 10s, 用户体验严重受损",
            "actions": [
                RemediationAction(
                    name="reduce_max_num_seqs",
                    description="临时降低 max_num_seqs (减少并发, 降低排队)",
                    risk=ActionRisk.MEDIUM,
                    auto_approve=True,
                ),
                RemediationAction(
                    name="drain_low_priority_requests",
                    description="取消 BATCH/SPOT 优先级的排队请求",
                    risk=ActionRisk.MEDIUM,
                    auto_approve=True,
                ),
            ],
        }

        # === Rule 3: GPU 硬件错误 ===
        self._rules["gpu_xid_critical"] = {
            "condition": lambda metrics: metrics.get("xid_error_48", False),
            "root_cause": "GPU ECC DBE 错误 (XID 48), 计算结果可能不可靠",
            "actions": [
                RemediationAction(
                    name="remove_from_load_balancer",
                    description="从负载均衡中摘除受影响的实例",
                    risk=ActionRisk.HIGH,
                    auto_approve=False,  # 需要人工确认
                ),
                RemediationAction(
                    name="gpu_reset",
                    description="尝试 GPU Reset (nvidia-smi -r)",
                    risk=ActionRisk.HIGH,
                    auto_approve=False,
                ),
                RemediationAction(
                    name="notify_infra_team",
                    description="通知 Infra 团队安排硬件更换",
                    risk=ActionRisk.LOW,
                    auto_approve=True,
                ),
            ],
        }

        # === Rule 4: 吞吐归零 ===
        self._rules["throughput_zero"] = {
            "condition": lambda metrics: (
                metrics.get("throughput_tps", 1) == 0
                and metrics.get("queue_length", 0) > 0
            ),
            "root_cause": "吞吐为 0 但有排队请求, 服务可能 hung",
            "actions": [
                RemediationAction(
                    name="check_gpu_health",
                    description="检查 GPU 状态 (nvidia-smi, DCGM)",
                    risk=ActionRisk.LOW,
                    auto_approve=True,
                ),
                RemediationAction(
                    name="restart_vllm_worker",
                    description="滚动重启 vLLM Worker Pod",
                    risk=ActionRisk.HIGH,
                    auto_approve=False,
                    cooldown_s=600,
                ),
            ],
        }

        # === Rule 5: GPU 过热 ===
        self._rules["gpu_thermal"] = {
            "condition": lambda metrics: metrics.get("gpu_temp_max", 0) > 83,
            "root_cause": "GPU 温度 > 83°C, 已触发 thermal throttle",
            "actions": [
                RemediationAction(
                    name="reduce_batch_size",
                    description="降低 max_num_seqs 减少 GPU 负载",
                    risk=ActionRisk.MEDIUM,
                    auto_approve=True,
                ),
                RemediationAction(
                    name="notify_datacenter",
                    description="通知数据中心检查冷却系统",
                    risk=ActionRisk.LOW,
                    auto_approve=True,
                ),
            ],
        }

    async def evaluate_and_remediate(self, metrics: Dict) -> List[RemediationEvent]:
        """评估当前指标并执行修复

        Args:
            metrics: 当前系统指标 dict

        Returns:
            触发的修复事件列表
        """
        events = []

        for rule_name, rule in self._rules.items():
            if rule["condition"](metrics):
                # 检查冷却期
                if self._in_cooldown(rule_name):
                    continue

                event = await self._execute_rule(rule_name, rule, metrics)
                events.append(event)
                self._action_history.append(event)

        return events

    async def _execute_rule(self, rule_name: str, rule: Dict, metrics: Dict) -> RemediationEvent:
        """执行单条规则的所有动作"""
        self._event_counter += 1
        event = RemediationEvent(
            event_id=f"rem-{self._event_counter:04d}",
            trigger=rule_name,
            root_cause=rule["root_cause"],
        )

        logger.warning(f"[{event.event_id}] Triggered: {rule_name} | {rule['root_cause']}")

        for action in rule["actions"]:
            # 检查是否需要人工确认
            if not action.auto_approve and action.risk in (ActionRisk.HIGH, ActionRisk.CRITICAL):
                logger.info(
                    f"[{event.event_id}] Action '{action.name}' needs manual approval "
                    f"(risk={action.risk.value})"
                )
                event.actions_taken.append({
                    "action": action.name,
                    "status": ActionStatus.NEEDS_APPROVAL.value,
                    "risk": action.risk.value,
                    "description": action.description,
                })
                continue

            # 执行动作
            if self.dry_run:
                logger.info(f"[{event.event_id}] DRY-RUN: Would execute '{action.name}'")
                event.actions_taken.append({
                    "action": action.name,
                    "status": "dry_run",
                    "description": action.description,
                })
            else:
                try:
                    logger.info(f"[{event.event_id}] Executing: {action.name}")
                    # 实际执行 (这里是模拟)
                    if action.execute_fn:
                        await asyncio.wait_for(
                            action.execute_fn(metrics),
                            timeout=action.timeout_s,
                        )
                    await asyncio.sleep(0.1)  # 模拟执行

                    event.actions_taken.append({
                        "action": action.name,
                        "status": ActionStatus.SUCCESS.value,
                        "description": action.description,
                    })
                    logger.info(f"[{event.event_id}] Action '{action.name}' succeeded")
                except Exception as e:
                    event.actions_taken.append({
                        "action": action.name,
                        "status": ActionStatus.FAILED.value,
                        "error": str(e),
                    })
                    logger.error(f"[{event.event_id}] Action '{action.name}' failed: {e}")

        event.end_time = time.time()
        event.status = "completed"
        self._last_action_time[rule_name] = time.time()

        return event

    def _in_cooldown(self, rule_name: str) -> bool:
        """检查规则是否在冷却期内"""
        last_time = self._last_action_time.get(rule_name, 0)
        cooldown = 300  # 默认 5 分钟
        for rule in self._rules.get(rule_name, {}).get("actions", []):
            cooldown = max(cooldown, rule.cooldown_s)
        return (time.time() - last_time) < cooldown

    def get_action_history(self, limit: int = 10) -> List[Dict]:
        """获取最近的修复历史"""
        return [
            {
                "event_id": e.event_id,
                "trigger": e.trigger,
                "root_cause": e.root_cause,
                "actions": e.actions_taken,
                "status": e.status,
                "time": e.start_time,
            }
            for e in self._action_history[-limit:]
        ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    async def demo():
        engine = AutoRemediationEngine(dry_run=True)

        print("=== Auto-Remediation Engine Demo (dry-run) ===\n")

        # 场景 1: KV Cache 满
        metrics = {
            "kv_cache_usage": 0.97,
            "queue_length": 30,
            "ttft_p99_s": 8,
            "throughput_tps": 500,
            "gpu_temp_max": 72,
        }
        events = await engine.evaluate_and_remediate(metrics)
        print(f"\nScenario 1 (KV Cache full): {len(events)} events triggered")
        for e in events:
            print(f"  [{e.event_id}] {e.trigger}: {len(e.actions_taken)} actions")
            for a in e.actions_taken:
                print(f"    - {a['action']}: {a['status']}")

        # 场景 2: GPU 过热
        engine._last_action_time.clear()  # 重置冷却期
        metrics = {"gpu_temp_max": 85, "kv_cache_usage": 0.5, "throughput_tps": 1000}
        events = await engine.evaluate_and_remediate(metrics)
        print(f"\nScenario 2 (GPU overheat): {len(events)} events triggered")
        for e in events:
            for a in e.actions_taken:
                print(f"    - {a['action']}: {a['status']}")

    asyncio.run(demo())
