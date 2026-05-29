"""
混合部署成本对比 — On-Demand vs Reserved vs Spot
=================================================

对比不同 GPU 部署策略的成本与风险。

依赖: numpy
"""

import logging
from typing import Dict, List
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DeploymentScenario:
    """部署场景"""
    name: str
    on_demand_instances: int = 0
    reserved_instances: int = 0
    spot_instances: int = 0
    gpus_per_instance: int = 8
    on_demand_price_per_gpu_hour: float = 4.0
    reserved_price_per_gpu_hour: float = 2.4    # 40% 折扣
    spot_price_per_gpu_hour: float = 1.2         # 70% 折扣
    spot_availability: float = 0.85              # Spot 可用性


def compare_scenarios(
    target_qps: float = 30.0,
    qps_per_instance: float = 10.0,
    hours_per_month: int = 730,
) -> List[Dict]:
    """对比多种部署方案"""
    min_instances = int(np.ceil(target_qps / qps_per_instance))

    scenarios = [
        DeploymentScenario(
            name="All On-Demand",
            on_demand_instances=min_instances,
        ),
        DeploymentScenario(
            name="All Reserved (1yr)",
            reserved_instances=min_instances,
        ),
        DeploymentScenario(
            name="Hybrid: 50% Reserved + 50% Spot",
            reserved_instances=min_instances // 2,
            spot_instances=min_instances - min_instances // 2 + 1,  # +1 补偿 Spot 中断
        ),
        DeploymentScenario(
            name="Hybrid: 30% OD + 40% RI + 30% Spot",
            on_demand_instances=max(1, int(min_instances * 0.3)),
            reserved_instances=int(min_instances * 0.4),
            spot_instances=int(min_instances * 0.3) + 1,
        ),
        DeploymentScenario(
            name="Aggressive Spot: 20% OD + 80% Spot",
            on_demand_instances=max(1, int(min_instances * 0.2)),
            spot_instances=int(min_instances * 0.8) + 2,  # +2 补偿中断
        ),
    ]

    results = []
    for s in scenarios:
        total_gpus = (
            (s.on_demand_instances + s.reserved_instances + s.spot_instances)
            * s.gpus_per_instance
        )

        monthly_cost = (
            s.on_demand_instances * s.gpus_per_instance * s.on_demand_price_per_gpu_hour * hours_per_month
            + s.reserved_instances * s.gpus_per_instance * s.reserved_price_per_gpu_hour * hours_per_month
            + s.spot_instances * s.gpus_per_instance * s.spot_price_per_gpu_hour * hours_per_month
        )

        # 有效容量 (考虑 Spot 中断)
        effective_instances = (
            s.on_demand_instances
            + s.reserved_instances
            + s.spot_instances * s.spot_availability
        )
        effective_qps = effective_instances * qps_per_instance

        # 最差情况 (所有 Spot 同时中断)
        worst_case_qps = (s.on_demand_instances + s.reserved_instances) * qps_per_instance
        slo_met_worst_case = worst_case_qps >= target_qps * 0.7  # 允许降级到 70%

        results.append({
            "scenario": s.name,
            "instances": {
                "on_demand": s.on_demand_instances,
                "reserved": s.reserved_instances,
                "spot": s.spot_instances,
                "total": s.on_demand_instances + s.reserved_instances + s.spot_instances,
            },
            "total_gpus": total_gpus,
            "monthly_cost": round(monthly_cost, 0),
            "yearly_cost": round(monthly_cost * 12, 0),
            "cost_per_gpu_hour_blended": round(
                monthly_cost / (total_gpus * hours_per_month), 3
            ),
            "effective_qps": round(effective_qps, 1),
            "worst_case_qps": round(worst_case_qps, 1),
            "slo_met_worst_case": slo_met_worst_case,
            "vs_all_on_demand_pct": None,  # 后面填充
        })

    # 计算相对节省
    baseline_cost = results[0]["monthly_cost"]
    for r in results:
        r["vs_all_on_demand_pct"] = round(
            (1 - r["monthly_cost"] / baseline_cost) * 100, 1
        ) if baseline_cost > 0 else 0

    return results


if __name__ == "__main__":
    print("=== GPU Deployment Cost Comparison ===\n")
    print(f"Target: 30 QPS, 10 QPS/instance\n")

    results = compare_scenarios(target_qps=30.0, qps_per_instance=10.0)

    for r in results:
        print(f"--- {r['scenario']} ---")
        print(f"  Instances: OD={r['instances']['on_demand']}, "
              f"RI={r['instances']['reserved']}, Spot={r['instances']['spot']}")
        print(f"  Monthly: ${r['monthly_cost']:,.0f} | Yearly: ${r['yearly_cost']:,.0f}")
        print(f"  Saving: {r['vs_all_on_demand_pct']}% vs All On-Demand")
        print(f"  Effective QPS: {r['effective_qps']} | Worst Case: {r['worst_case_qps']}")
        print(f"  SLO (worst case): {'OK' if r['slo_met_worst_case'] else 'AT RISK'}")
        print()
