"""
多租户计费引擎 — GPU 推理服务 Chargeback
==========================================

实现按使用量的公平计费:
1. Token 计费: 按消耗的 prompt + output tokens
2. GPU-Time 计费: 按占用的 GPU 时间
3. QoS 分级: 不同 SLA 等级不同价格
4. 报表生成: 按团队/项目/用户汇总

依赖: numpy
"""

import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class QoSTier(Enum):
    """服务等级"""
    PREMIUM = "premium"       # TTFT < 2s, 不被 preempt, 1.5x 价格
    STANDARD = "standard"     # TTFT < 5s, 可能被 preempt, 1.0x 价格
    BATCH = "batch"           # 无 SLO, 最低优先级, 0.5x 价格
    SPOT = "spot"             # 闲时使用, 随时可中断, 0.3x 价格


QOS_MULTIPLIER = {
    QoSTier.PREMIUM: 1.5,
    QoSTier.STANDARD: 1.0,
    QoSTier.BATCH: 0.5,
    QoSTier.SPOT: 0.3,
}


@dataclass
class UsageRecord:
    """单次使用记录"""
    request_id: str
    tenant_id: str                    # 租户/团队 ID
    project_id: str = ""              # 项目 ID
    user_id: str = ""                 # 用户 ID
    model: str = "Qwen2.5-72B"
    qos_tier: QoSTier = QoSTier.STANDARD
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    gpu_time_ms: float = 0            # GPU 占用时间 (毫秒)
    was_preempted: bool = False
    ttft_ms: float = 0
    e2e_latency_ms: float = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ChargebackConfig:
    """计费配置"""
    # 基础单价 ($/1M tokens)
    base_price_input_per_1m: float = 0.50     # Prompt tokens
    base_price_output_per_1m: float = 2.00    # Output tokens (更贵, 因为 decode 慢)

    # GPU-Time 单价 ($/GPU-second) — 作为兜底
    gpu_second_price: float = 0.002

    # 最小计费单位
    min_charge_per_request: float = 0.0001    # 最低收费 ($)

    # 折扣
    volume_discount_thresholds: Dict[int, float] = field(default_factory=lambda: {
        10_000_000: 0.95,    # > 10M tokens/月: 95 折
        100_000_000: 0.85,   # > 100M tokens/月: 85 折
        1_000_000_000: 0.70, # > 1B tokens/月: 70 折
    })


class ChargebackCalculator:
    """计费计算引擎

    计费公式:
    charge = max(
        (prompt_tokens × input_price + output_tokens × output_price) × qos_multiplier,
        gpu_time_s × gpu_second_price × qos_multiplier,
        min_charge
    )

    设计考量:
    - 双轨定价: Token-based + GPU-time-based 取较高者
      → Token 定价适合大多数场景
      → GPU-time 定价保护长 context / 低 output 场景
    - QoS 差异化: Premium 用户付更多但获得更好的 SLO
    - 被 Preempt 的请求: 不收重复 prefill 的费用
    """

    def __init__(self, config: ChargebackConfig = None):
        self.config = config or ChargebackConfig()
        self._usage_records: List[UsageRecord] = []
        self._tenant_totals: Dict[str, Dict] = defaultdict(lambda: {
            "total_prompt_tokens": 0,
            "total_output_tokens": 0,
            "total_requests": 0,
            "total_charge": 0.0,
            "total_gpu_time_ms": 0.0,
        })

    def record_usage(self, record: UsageRecord) -> float:
        """记录一次使用并计算费用

        Returns:
            计算出的费用 ($)
        """
        record.total_tokens = record.prompt_tokens + record.output_tokens

        # 计算 token-based 费用
        token_charge = (
            record.prompt_tokens * self.config.base_price_input_per_1m / 1e6
            + record.output_tokens * self.config.base_price_output_per_1m / 1e6
        )

        # 计算 GPU-time-based 费用
        gpu_time_s = record.gpu_time_ms / 1000
        gpu_charge = gpu_time_s * self.config.gpu_second_price

        # 取较高者
        base_charge = max(token_charge, gpu_charge)

        # QoS 乘数
        qos_mult = QOS_MULTIPLIER.get(record.qos_tier, 1.0)
        charge = base_charge * qos_mult

        # 被 preempt 的补偿: 退还 10% (因为用户体验受损)
        if record.was_preempted:
            charge *= 0.9

        # 最低收费
        charge = max(charge, self.config.min_charge_per_request)

        # 记录
        self._usage_records.append(record)
        tenant = self._tenant_totals[record.tenant_id]
        tenant["total_prompt_tokens"] += record.prompt_tokens
        tenant["total_output_tokens"] += record.output_tokens
        tenant["total_requests"] += 1
        tenant["total_charge"] += charge
        tenant["total_gpu_time_ms"] += record.gpu_time_ms

        return round(charge, 6)

    def apply_volume_discount(self, tenant_id: str) -> float:
        """应用月度用量折扣

        Returns:
            折扣后的总费用
        """
        tenant = self._tenant_totals.get(tenant_id)
        if not tenant:
            return 0.0

        total_tokens = tenant["total_prompt_tokens"] + tenant["total_output_tokens"]
        total_charge = tenant["total_charge"]

        discount = 1.0
        for threshold, disc in sorted(
            self.config.volume_discount_thresholds.items(), reverse=True
        ):
            if total_tokens >= threshold:
                discount = disc
                break

        return round(total_charge * discount, 4)

    def generate_tenant_report(self, tenant_id: str) -> Dict:
        """生成租户账单"""
        tenant = self._tenant_totals.get(tenant_id)
        if not tenant:
            return {"error": f"Tenant {tenant_id} not found"}

        total_tokens = tenant["total_prompt_tokens"] + tenant["total_output_tokens"]
        discounted = self.apply_volume_discount(tenant_id)

        return {
            "tenant_id": tenant_id,
            "period": "current",
            "usage": {
                "total_requests": tenant["total_requests"],
                "prompt_tokens": tenant["total_prompt_tokens"],
                "output_tokens": tenant["total_output_tokens"],
                "total_tokens": total_tokens,
                "gpu_hours": round(tenant["total_gpu_time_ms"] / 3600000, 3),
            },
            "billing": {
                "gross_charge": round(tenant["total_charge"], 4),
                "volume_discount": round(tenant["total_charge"] - discounted, 4),
                "net_charge": discounted,
                "avg_cost_per_request": round(
                    discounted / max(1, tenant["total_requests"]), 6
                ),
                "cost_per_1m_tokens": round(
                    discounted / max(1, total_tokens / 1e6), 4
                ),
            },
        }

    def generate_summary_report(self) -> Dict:
        """生成全局汇总报告"""
        total_charge = sum(t["total_charge"] for t in self._tenant_totals.values())
        total_requests = sum(t["total_requests"] for t in self._tenant_totals.values())
        total_tokens = sum(
            t["total_prompt_tokens"] + t["total_output_tokens"]
            for t in self._tenant_totals.values()
        )

        tenant_breakdown = {}
        for tid, data in self._tenant_totals.items():
            tenant_breakdown[tid] = {
                "charge": round(data["total_charge"], 4),
                "share_pct": round(data["total_charge"] / max(0.01, total_charge) * 100, 1),
                "requests": data["total_requests"],
            }

        return {
            "total_charge": round(total_charge, 2),
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "num_tenants": len(self._tenant_totals),
            "tenant_breakdown": tenant_breakdown,
        }


if __name__ == "__main__":
    import random

    calc = ChargebackCalculator()

    tenants = ["team-search", "team-chat", "team-code", "team-analytics"]
    models = ["Qwen2.5-72B"]

    print("=== Chargeback Calculator Demo ===\n")

    for i in range(1000):
        tenant = random.choice(tenants)
        qos = random.choices(
            [QoSTier.PREMIUM, QoSTier.STANDARD, QoSTier.BATCH],
            weights=[0.1, 0.7, 0.2]
        )[0]

        record = UsageRecord(
            request_id=f"req-{i:04d}",
            tenant_id=tenant,
            model="Qwen2.5-72B",
            qos_tier=qos,
            prompt_tokens=random.randint(100, 4096),
            output_tokens=random.randint(50, 1024),
            gpu_time_ms=random.uniform(100, 5000),
            was_preempted=random.random() < 0.05,
        )
        calc.record_usage(record)

    summary = calc.generate_summary_report()
    print(f"Total Charge: ${summary['total_charge']:.2f}")
    print(f"Total Requests: {summary['total_requests']}")
    print(f"Total Tokens: {summary['total_tokens']:,}")
    print(f"\nBreakdown by Tenant:")
    for tid, data in summary["tenant_breakdown"].items():
        print(f"  {tid}: ${data['charge']:.2f} ({data['share_pct']}%) - {data['requests']} reqs")

    print(f"\nDetailed Report for team-chat:")
    report = calc.generate_tenant_report("team-chat")
    for section, data in report.items():
        if isinstance(data, dict):
            print(f"  {section}:")
            for k, v in data.items():
                print(f"    {k}: {v}")
