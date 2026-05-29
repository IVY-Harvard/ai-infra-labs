"""
多租户配额管理

管理各租户（团队）的 GPU 资源配额，包括：
  - 基础配额（guaranteed）
  - 弹性上限（burst limit）
  - 借用机制（从其他团队借用空闲配额）
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TenantQuota:
    """租户配额定义"""
    tenant_name: str
    gpu_quota: int               # 基础配额（保证可用）
    gpu_burst_limit: int         # 弹性上限（可以 burst 到的最大 GPU 数）
    cpu_quota: float = 0         # CPU 配额
    memory_quota_gb: float = 0   # 内存配额
    priority_default: int = 5    # 默认任务优先级
    can_borrow: bool = True      # 是否允许借用其他租户的空闲配额
    can_be_borrowed_from: bool = True  # 是否允许空闲配额被借用
    preemptible: bool = False    # 借用的资源是否可被回收


@dataclass
class TenantUsage:
    """租户当前资源使用情况"""
    tenant_name: str
    gpu_used: int = 0            # 当前使用的 GPU
    gpu_borrowed: int = 0        # 从其他租户借用的 GPU
    jobs_running: int = 0
    jobs_pending: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def gpu_total_used(self) -> int:
        return self.gpu_used + self.gpu_borrowed


class QuotaManager:
    """配额管理器"""

    def __init__(self, total_cluster_gpus: int = 8):
        self.total_cluster_gpus = total_cluster_gpus
        self._quotas: dict[str, TenantQuota] = {}
        self._usage: dict[str, TenantUsage] = {}

    def register_tenant(self, quota: TenantQuota):
        """注册租户"""
        self._quotas[quota.tenant_name] = quota
        self._usage[quota.tenant_name] = TenantUsage(tenant_name=quota.tenant_name)
        logger.info(
            f"注册租户 {quota.tenant_name}: "
            f"quota={quota.gpu_quota}, burst={quota.gpu_burst_limit}"
        )

    def check_quota(self, tenant_name: str, gpu_request: int) -> tuple[bool, str]:
        """
        检查租户是否有足够的配额。

        Returns:
            (allowed, reason)
        """
        quota = self._quotas.get(tenant_name)
        if quota is None:
            return False, f"租户 {tenant_name} 未注册"

        usage = self._usage[tenant_name]

        # 检查 1: 是否在基础配额内
        if usage.gpu_used + gpu_request <= quota.gpu_quota:
            return True, "在基础配额内"

        # 检查 2: 是否在 burst limit 内
        if usage.gpu_total_used + gpu_request <= quota.gpu_burst_limit:
            # 需要借用
            if not quota.can_borrow:
                return False, "超出基础配额且不允许借用"

            # 检查是否有可借用的空闲配额
            available_to_borrow = self._get_borrowable_gpus()
            borrow_needed = (usage.gpu_used + gpu_request) - quota.gpu_quota
            if borrow_needed <= available_to_borrow:
                return True, f"超出基础配额，需借用 {borrow_needed} GPU"
            else:
                return False, f"需要借用 {borrow_needed} GPU，但只有 {available_to_borrow} 可借"

        return False, f"超出 burst limit ({quota.gpu_burst_limit})"

    def allocate(self, tenant_name: str, gpu_count: int) -> bool:
        """分配 GPU 资源给租户"""
        allowed, reason = self.check_quota(tenant_name, gpu_count)
        if not allowed:
            logger.warning(f"配额拒绝: {tenant_name} 请求 {gpu_count} GPU: {reason}")
            return False

        usage = self._usage[tenant_name]
        quota = self._quotas[tenant_name]

        # 计算借用量
        within_quota = min(gpu_count, quota.gpu_quota - usage.gpu_used)
        borrowed = gpu_count - max(within_quota, 0)

        usage.gpu_used += max(within_quota, 0)
        usage.gpu_borrowed += max(borrowed, 0)
        usage.jobs_running += 1
        usage.last_updated = datetime.now()

        logger.info(
            f"配额分配: {tenant_name} +{gpu_count} GPU "
            f"(自有={max(within_quota, 0)}, 借用={max(borrowed, 0)})"
        )
        return True

    def release(self, tenant_name: str, gpu_count: int):
        """释放租户的 GPU 资源"""
        if tenant_name not in self._usage:
            return

        usage = self._usage[tenant_name]

        # 优先释放借用的 GPU
        borrowed_release = min(gpu_count, usage.gpu_borrowed)
        own_release = gpu_count - borrowed_release

        usage.gpu_borrowed -= borrowed_release
        usage.gpu_used -= own_release
        usage.jobs_running = max(0, usage.jobs_running - 1)
        usage.last_updated = datetime.now()

        logger.info(
            f"配额释放: {tenant_name} -{gpu_count} GPU "
            f"(自有={own_release}, 借用={borrowed_release})"
        )

    def _get_borrowable_gpus(self) -> int:
        """计算集群中可借用的空闲 GPU 数量"""
        total_idle = 0
        for tenant_name, quota in self._quotas.items():
            if not quota.can_be_borrowed_from:
                continue
            usage = self._usage[tenant_name]
            idle = max(0, quota.gpu_quota - usage.gpu_used)
            total_idle += idle
        return total_idle

    def get_recallable_gpus(self, tenant_name: str) -> int:
        """获取某租户可被回收的借用 GPU 数量"""
        usage = self._usage.get(tenant_name)
        if not usage:
            return 0
        quota = self._quotas.get(tenant_name)
        if not quota or not quota.preemptible:
            return 0
        return usage.gpu_borrowed

    def get_tenant_status(self, tenant_name: str) -> Optional[dict]:
        """获取租户状态"""
        quota = self._quotas.get(tenant_name)
        usage = self._usage.get(tenant_name)
        if not quota or not usage:
            return None

        return {
            "tenant": tenant_name,
            "quota": quota.gpu_quota,
            "burst_limit": quota.gpu_burst_limit,
            "gpu_used": usage.gpu_used,
            "gpu_borrowed": usage.gpu_borrowed,
            "total_used": usage.gpu_total_used,
            "utilization": round(usage.gpu_used / quota.gpu_quota * 100, 1) if quota.gpu_quota > 0 else 0,
            "jobs_running": usage.jobs_running,
            "jobs_pending": usage.jobs_pending,
        }

    def get_all_tenants_status(self) -> list[dict]:
        """获取所有租户状态"""
        return [
            self.get_tenant_status(name)
            for name in self._quotas
            if self.get_tenant_status(name)
        ]
