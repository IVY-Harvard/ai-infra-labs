"""
公平共享调度算法

实现 Dominant Resource Fairness (DRF) 算法，
确保多个租户公平地共享 GPU 集群资源。
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TenantDemand:
    """租户的资源需求"""
    tenant_name: str
    gpu_demand: int       # 待满足的 GPU 需求
    cpu_demand: float     # 待满足的 CPU 需求
    gpu_allocated: int    # 已分配的 GPU
    cpu_allocated: float  # 已分配的 CPU
    weight: float = 1.0   # 租户权重（用于加权公平共享）


class FairShareCalculator:
    """
    基于 DRF (Dominant Resource Fairness) 的公平共享计算器。

    DRF 核心思想：
    - 每个租户有一个 "dominant share"（各资源维度中占比最高的那个）
    - 调度时优先满足 dominant share 最低的租户
    - 保证 Pareto 最优

    示例（8 GPU, 64 CPU 的集群）：
      Team A 的任务: 2 GPU + 8 CPU → GPU share=25%, CPU share=12.5% → dominant=25%
      Team B 的任务: 1 GPU + 16 CPU → GPU share=12.5%, CPU share=25% → dominant=25%
      → 两个团队的 dominant share 相同 → 公平
    """

    def __init__(self, total_gpus: int, total_cpus: float):
        self.total_gpus = total_gpus
        self.total_cpus = total_cpus

    def dominant_share(self, tenant: TenantDemand) -> float:
        """计算租户的 dominant resource share"""
        gpu_share = tenant.gpu_allocated / self.total_gpus if self.total_gpus > 0 else 0
        cpu_share = tenant.cpu_allocated / self.total_cpus if self.total_cpus > 0 else 0
        return max(gpu_share, cpu_share)

    def weighted_dominant_share(self, tenant: TenantDemand) -> float:
        """加权 dominant share（考虑租户权重）"""
        ds = self.dominant_share(tenant)
        return ds / tenant.weight if tenant.weight > 0 else float("inf")

    def compute_fair_allocation(
        self,
        tenants: list[TenantDemand],
    ) -> list[tuple[str, int]]:
        """
        计算公平的 GPU 分配方案。

        使用贪心算法逐步分配：
        1. 找到 weighted dominant share 最低的租户
        2. 给它分配一个 GPU
        3. 重复直到 GPU 用完或所有需求满足

        Returns:
            [(tenant_name, additional_gpus_to_allocate)]
        """
        remaining_gpus = self.total_gpus - sum(t.gpu_allocated for t in tenants)
        remaining_cpus = self.total_cpus - sum(t.cpu_allocated for t in tenants)

        if remaining_gpus <= 0:
            return [(t.tenant_name, 0) for t in tenants]

        # 工作副本
        alloc = {t.tenant_name: 0 for t in tenants}
        state = {t.tenant_name: t for t in tenants}

        for _ in range(remaining_gpus):
            # 找 weighted dominant share 最低且还有需求的租户
            best_tenant = None
            best_wds = float("inf")

            for t in tenants:
                unmet_demand = t.gpu_demand - alloc[t.tenant_name]
                if unmet_demand <= 0:
                    continue

                # 模拟分配后的 dominant share
                simulated = TenantDemand(
                    tenant_name=t.tenant_name,
                    gpu_demand=t.gpu_demand,
                    cpu_demand=t.cpu_demand,
                    gpu_allocated=t.gpu_allocated + alloc[t.tenant_name] + 1,
                    cpu_allocated=t.cpu_allocated,
                    weight=t.weight,
                )
                wds = self.weighted_dominant_share(simulated)

                if wds < best_wds:
                    best_wds = wds
                    best_tenant = t.tenant_name

            if best_tenant is None:
                break  # 所有需求已满足

            alloc[best_tenant] += 1

        result = [(name, count) for name, count in alloc.items()]
        logger.info(f"公平共享分配: {result}")
        return result

    def get_fairness_report(self, tenants: list[TenantDemand]) -> dict:
        """生成公平性报告"""
        shares = {}
        for t in tenants:
            ds = self.dominant_share(t)
            wds = self.weighted_dominant_share(t)
            shares[t.tenant_name] = {
                "gpu_allocated": t.gpu_allocated,
                "gpu_demand": t.gpu_demand,
                "dominant_share": round(ds * 100, 1),
                "weighted_dominant_share": round(wds * 100, 1),
                "weight": t.weight,
                "satisfied": t.gpu_allocated >= t.gpu_demand,
            }

        # 公平性指标：各租户 weighted_dominant_share 的方差
        wds_values = [s["weighted_dominant_share"] for s in shares.values()]
        if wds_values:
            mean_wds = sum(wds_values) / len(wds_values)
            variance = sum((v - mean_wds) ** 2 for v in wds_values) / len(wds_values)
        else:
            variance = 0

        return {
            "tenants": shares,
            "fairness_variance": round(variance, 2),
            "is_fair": variance < 100,  # 方差小于 100 认为基本公平
        }
