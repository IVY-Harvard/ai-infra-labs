"""
GPU 成本模型 — TCO 与单位成本计算
==================================

计算 GPU 推理的完整成本:
1. TCO (Total Cost of Ownership): 硬件+电力+人力+运维
2. 单位成本: $/GPU-hour, $/million-tokens, $/request
3. 成本对比: 自建 vs 云 vs Spot

依赖: numpy
"""

import logging
from typing import Dict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class HardwareCost:
    """硬件成本"""
    gpu_unit_price: float = 15000       # 单 GPU 价格 ($)
    gpus_per_node: int = 8
    server_price: float = 30000         # 服务器 (不含 GPU) 价格 ($)
    nvswitch_price: float = 5000        # NVSwitch/互联价格 ($)
    networking_price: float = 10000     # 网络设备分摊 ($)
    depreciation_years: int = 3         # 折旧年限


@dataclass
class OperationalCost:
    """运营成本"""
    power_per_node_watts: float = 4000  # 节点功耗 (W)
    pue: float = 1.3                    # Power Usage Effectiveness
    electricity_rate: float = 0.08      # $/kWh
    hours_per_month: int = 730
    colo_cost_per_node_month: float = 500  # 机柜租金 ($)
    network_bandwidth_month: float = 200   # 网络费用 ($)


@dataclass
class StaffCost:
    """人力成本 (按节点分摊)"""
    sre_salary_yearly: float = 200000
    mlops_salary_yearly: float = 180000
    sre_nodes_ratio: int = 50           # 1 SRE 管 50 个节点
    mlops_nodes_ratio: int = 30         # 1 MLOps 管 30 个节点


class GPUCostModel:
    """GPU 推理成本模型

    计算层级:
    1. 节点级 TCO: 一个 8-GPU 节点的月度/年度成本
    2. GPU-Hour 成本: 单 GPU 每小时的完全成本
    3. Token 成本: 每百万 token 的成本
    4. 请求成本: 单个请求的成本 (基于平均 token 数)
    """

    def __init__(
        self,
        hardware: HardwareCost = None,
        operational: OperationalCost = None,
        staff: StaffCost = None,
    ):
        self.hw = hardware or HardwareCost()
        self.ops = operational or OperationalCost()
        self.staff = staff or StaffCost()

    def node_tco_monthly(self) -> Dict[str, float]:
        """单节点月度 TCO 分解"""
        # 硬件折旧
        total_hw = (
            self.hw.gpu_unit_price * self.hw.gpus_per_node
            + self.hw.server_price
            + self.hw.nvswitch_price
            + self.hw.networking_price
        )
        hw_monthly = total_hw / (self.hw.depreciation_years * 12)

        # 电力
        actual_power_kw = self.ops.power_per_node_watts * self.ops.pue / 1000
        power_monthly = actual_power_kw * self.ops.hours_per_month * self.ops.electricity_rate

        # 机房
        colo_monthly = self.ops.colo_cost_per_node_month

        # 网络
        network_monthly = self.ops.network_bandwidth_month

        # 人力分摊
        sre_monthly = self.staff.sre_salary_yearly / 12 / self.staff.sre_nodes_ratio
        mlops_monthly = self.staff.mlops_salary_yearly / 12 / self.staff.mlops_nodes_ratio
        staff_monthly = sre_monthly + mlops_monthly

        total = hw_monthly + power_monthly + colo_monthly + network_monthly + staff_monthly

        return {
            "hardware_depreciation": round(hw_monthly, 2),
            "electricity": round(power_monthly, 2),
            "colocation": round(colo_monthly, 2),
            "network": round(network_monthly, 2),
            "staff": round(staff_monthly, 2),
            "total_monthly": round(total, 2),
            "total_yearly": round(total * 12, 2),
        }

    def gpu_hour_cost(self) -> float:
        """单 GPU-Hour 完全成本"""
        monthly = self.node_tco_monthly()["total_monthly"]
        return round(monthly / self.hw.gpus_per_node / self.ops.hours_per_month, 4)

    def token_cost(
        self,
        throughput_tokens_per_s: float = 1500,
        utilization: float = 0.7,
    ) -> Dict[str, float]:
        """每百万 token 成本

        Args:
            throughput_tokens_per_s: 集群总吞吐 (tokens/s)
            utilization: 平均利用率 (0-1)
        """
        # 有效吞吐 = 名义吞吐 × 利用率
        effective_tps = throughput_tokens_per_s * utilization
        tokens_per_hour = effective_tps * 3600
        tokens_per_month = tokens_per_hour * self.ops.hours_per_month

        monthly_cost = self.node_tco_monthly()["total_monthly"]

        cost_per_1m = monthly_cost / (tokens_per_month / 1e6) if tokens_per_month > 0 else float('inf')

        return {
            "cost_per_1m_tokens": round(cost_per_1m, 4),
            "effective_throughput_tps": round(effective_tps, 1),
            "tokens_per_month": round(tokens_per_month, 0),
            "monthly_cost": round(monthly_cost, 2),
            "utilization": utilization,
        }

    def compare_deployment_options(
        self,
        monthly_cloud_on_demand: float = 25000,
        monthly_cloud_reserved_1y: float = 18000,
        monthly_cloud_spot: float = 8000,
        spot_availability: float = 0.85,
    ) -> Dict:
        """对比自建 vs 云部署成本"""
        on_prem = self.node_tco_monthly()["total_monthly"]

        return {
            "on_premise": {
                "monthly": round(on_prem, 2),
                "yearly": round(on_prem * 12, 2),
                "3_year": round(on_prem * 36, 2),
                "effective_availability": 0.999,
            },
            "cloud_on_demand": {
                "monthly": monthly_cloud_on_demand,
                "yearly": monthly_cloud_on_demand * 12,
                "vs_on_prem": f"{(monthly_cloud_on_demand/on_prem - 1)*100:+.0f}%",
            },
            "cloud_reserved_1y": {
                "monthly": monthly_cloud_reserved_1y,
                "yearly": monthly_cloud_reserved_1y * 12,
                "vs_on_prem": f"{(monthly_cloud_reserved_1y/on_prem - 1)*100:+.0f}%",
            },
            "cloud_spot": {
                "monthly_equivalent": round(monthly_cloud_spot / spot_availability, 2),
                "hourly_saving_vs_on_demand": f"{(1 - monthly_cloud_spot/monthly_cloud_on_demand)*100:.0f}%",
                "availability": spot_availability,
                "risk": "中断风险, 需要 fallback 策略",
            },
            "recommendation": self._deployment_recommendation(
                on_prem, monthly_cloud_reserved_1y
            ),
        }

    def _deployment_recommendation(self, on_prem: float, cloud_reserved: float) -> str:
        ratio = cloud_reserved / on_prem
        if ratio < 0.9:
            return "云 Reserved 更划算 (无需管理硬件)"
        elif ratio < 1.3:
            return "成本接近, 根据运维能力选择 (有强 Infra 团队选自建)"
        else:
            return "自建更划算 (前提: 有稳定长期需求)"


if __name__ == "__main__":
    model = GPUCostModel()

    print("=== GPU Cost Model (8×H20 Node) ===\n")

    tco = model.node_tco_monthly()
    print("Monthly TCO Breakdown:")
    for k, v in tco.items():
        print(f"  {k}: ${v:,.2f}")

    print(f"\nGPU-Hour Cost: ${model.gpu_hour_cost():.4f}")

    tokens = model.token_cost(throughput_tokens_per_s=1500, utilization=0.7)
    print(f"\nToken Cost (@70% util, 1500 tps):")
    for k, v in tokens.items():
        print(f"  {k}: {v}")

    print(f"\n=== Deployment Comparison ===")
    comparison = model.compare_deployment_options()
    print(f"  On-Premise: ${comparison['on_premise']['monthly']:,.0f}/month")
    print(f"  Cloud On-Demand: ${comparison['cloud_on_demand']['monthly']:,.0f}/month "
          f"({comparison['cloud_on_demand']['vs_on_prem']})")
    print(f"  Cloud Reserved: ${comparison['cloud_reserved_1y']['monthly']:,.0f}/month "
          f"({comparison['cloud_reserved_1y']['vs_on_prem']})")
    print(f"  Recommendation: {comparison['recommendation']}")
