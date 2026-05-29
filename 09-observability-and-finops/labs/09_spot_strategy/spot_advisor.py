"""
Spot Instance 选型顾问
======================

根据中断历史和价格数据推荐最优 Spot 配置:
1. 实例类型选择 (多 GPU 类型混合)
2. 可用区选择 (历史中断率最低)
3. 竞价策略 (基于价格历史)

依赖: numpy
"""

import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpotInstanceType:
    """Spot 实例类型信息"""
    instance_type: str
    gpu_model: str
    gpu_count: int
    gpu_memory_gb: float
    on_demand_price: float      # $/hour
    spot_price_avg: float       # $/hour (30 天均价)
    spot_price_min: float
    spot_price_max: float
    interruption_rate_pct: float  # 月均中断率 (%)
    availability_zones: List[str] = field(default_factory=list)


@dataclass
class SpotRecommendation:
    """Spot 推荐结果"""
    instance_type: str
    availability_zone: str
    bid_price: float
    expected_saving_pct: float
    interruption_risk: str        # "low" | "medium" | "high"
    confidence: float
    reasoning: str


# 模拟的 Spot 市场数据 (实际应从云 API 获取)
SPOT_MARKET_DATA = [
    SpotInstanceType(
        instance_type="p5.48xlarge",
        gpu_model="H100",
        gpu_count=8,
        gpu_memory_gb=80,
        on_demand_price=98.32,
        spot_price_avg=32.0,
        spot_price_min=25.0,
        spot_price_max=55.0,
        interruption_rate_pct=8.0,
        availability_zones=["us-east-1a", "us-east-1b", "us-east-1c"],
    ),
    SpotInstanceType(
        instance_type="p4d.24xlarge",
        gpu_model="A100",
        gpu_count=8,
        gpu_memory_gb=80,
        on_demand_price=32.77,
        spot_price_avg=12.0,
        spot_price_min=9.0,
        spot_price_max=22.0,
        interruption_rate_pct=12.0,
        availability_zones=["us-east-1a", "us-east-1b", "us-west-2a"],
    ),
    SpotInstanceType(
        instance_type="g5.48xlarge",
        gpu_model="A10G",
        gpu_count=8,
        gpu_memory_gb=24,
        on_demand_price=16.29,
        spot_price_avg=5.5,
        spot_price_min=3.8,
        spot_price_max=11.0,
        interruption_rate_pct=15.0,
        availability_zones=["us-east-1a", "us-east-1b", "us-east-1c", "us-west-2a"],
    ),
]


class SpotAdvisor:
    """Spot 选型顾问"""

    def __init__(self, market_data: List[SpotInstanceType] = None):
        self.market_data = market_data or SPOT_MARKET_DATA

    def recommend(
        self,
        min_gpu_memory_gb: float = 80,
        min_gpu_count: int = 8,
        max_interruption_rate_pct: float = 15,
        budget_per_hour: float = 50.0,
    ) -> List[SpotRecommendation]:
        """生成 Spot 推荐

        Args:
            min_gpu_memory_gb: 最低 GPU 显存需求
            min_gpu_count: 最少 GPU 数
            max_interruption_rate_pct: 可接受的最高中断率
            budget_per_hour: 每小时预算上限
        """
        recommendations = []

        for inst in self.market_data:
            # 过滤不满足需求的
            if inst.gpu_memory_gb < min_gpu_memory_gb:
                continue
            if inst.gpu_count < min_gpu_count:
                continue
            if inst.interruption_rate_pct > max_interruption_rate_pct:
                continue
            if inst.spot_price_avg > budget_per_hour:
                continue

            # 计算推荐竞价
            bid_price = min(inst.spot_price_max * 0.8, inst.on_demand_price * 0.5)

            # 计算节省
            saving = (1 - inst.spot_price_avg / inst.on_demand_price) * 100

            # 风险评估
            if inst.interruption_rate_pct < 5:
                risk = "low"
            elif inst.interruption_rate_pct < 10:
                risk = "medium"
            else:
                risk = "high"

            # 每个可用区一个推荐
            for az in inst.availability_zones:
                recommendations.append(SpotRecommendation(
                    instance_type=inst.instance_type,
                    availability_zone=az,
                    bid_price=round(bid_price, 2),
                    expected_saving_pct=round(saving, 1),
                    interruption_risk=risk,
                    confidence=round(1 - inst.interruption_rate_pct / 100, 3),
                    reasoning=(
                        f"{inst.gpu_model} x{inst.gpu_count}, "
                        f"avg ${inst.spot_price_avg:.1f}/h "
                        f"(vs ${inst.on_demand_price:.1f} on-demand), "
                        f"中断率 {inst.interruption_rate_pct}%"
                    ),
                ))

        # 按节省百分比 × 置信度排序
        recommendations.sort(
            key=lambda r: r.expected_saving_pct * r.confidence,
            reverse=True,
        )
        return recommendations

    def diversification_strategy(
        self, total_spot_instances: int = 4
    ) -> Dict:
        """Spot 多样化策略 (降低同时中断风险)

        核心思想: 不要把所有 Spot 放在同一个 AZ / 实例类型
        """
        # 跨 AZ 分布
        all_azs = set()
        for inst in self.market_data:
            all_azs.update(inst.availability_zones)

        # 跨实例类型分布
        type_groups = {}
        for inst in self.market_data:
            type_groups[inst.instance_type] = inst

        return {
            "strategy": "multi-az-multi-type",
            "description": "跨可用区 + 跨实例类型分散风险",
            "allocation": {
                "num_azs_used": min(3, len(all_azs)),
                "num_instance_types": min(2, len(type_groups)),
                "instances_per_az": max(1, total_spot_instances // 3),
            },
            "risk_reduction": {
                "single_az_failure": "仅影响 1/3 Spot 容量",
                "single_type_reclaim": "仅影响 1/2 Spot 容量",
                "simultaneous_probability": "< 1% (假设独立事件)",
            },
        }


if __name__ == "__main__":
    advisor = SpotAdvisor()

    print("=== Spot Instance Advisor ===\n")

    recommendations = advisor.recommend(
        min_gpu_memory_gb=80,
        min_gpu_count=8,
        max_interruption_rate_pct=15,
        budget_per_hour=50.0,
    )

    print(f"Top {min(5, len(recommendations))} Recommendations:")
    for i, rec in enumerate(recommendations[:5]):
        print(f"\n  #{i+1} {rec.instance_type} @ {rec.availability_zone}")
        print(f"     Bid: ${rec.bid_price}/h, Saving: {rec.expected_saving_pct}%")
        print(f"     Risk: {rec.interruption_risk}, Confidence: {rec.confidence}")
        print(f"     {rec.reasoning}")

    print(f"\n\n=== Diversification Strategy ===")
    strategy = advisor.diversification_strategy(total_spot_instances=4)
    for k, v in strategy.items():
        print(f"  {k}: {v}")
