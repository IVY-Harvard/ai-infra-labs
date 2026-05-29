"""
GPU 任务抢占管理

当高优先级任务没有可用资源时，可以抢占低优先级任务。
抢占逻辑需要仔细设计，避免"抢占风暴"。
"""

import logging
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PreemptionConfig:
    """抢占策略配置"""
    enabled: bool = True
    min_priority_gap: int = 3          # 优先级差至少 3 才能抢占
    grace_period_seconds: int = 30     # 被抢占任务的优雅退出时间
    max_preemptions_per_cycle: int = 2 # 每个调度周期最多抢占 2 个任务
    protect_running_minutes: int = 5   # 刚启动 5 分钟内的任务不被抢占
    cooldown_seconds: int = 60         # 同一个任务被抢占后的冷却时间


class PreemptionManager:
    """抢占管理器"""

    def __init__(self, config: Optional[PreemptionConfig] = None):
        self.config = config or PreemptionConfig()
        self._preemption_history: list[dict] = []
        self._cooldown_map: dict[str, datetime] = {}  # job_id → last_preempted_time

    def try_preempt(
        self,
        job,      # GPUJob — 待调度的高优先级任务
        running_jobs: dict,  # {job_id: GPUJob}
        nodes: dict,         # {node_name: NodeInfo}
    ):
        """
        尝试抢占低优先级任务来释放资源。

        抢占策略：
        1. 找出所有可以被抢占的任务（优先级低于请求者）
        2. 按"抢占代价最小"排序（优先抢占优先级最低、运行时间最短的）
        3. 模拟抢占后是否有足够资源
        4. 执行抢占

        Returns:
            SchedulingResult
        """
        from .gpu_scheduler import SchedulingResult, JobState

        if not self.config.enabled:
            return SchedulingResult(
                job=job, success=False, reason="抢占未启用"
            )

        # 找出可抢占的候选任务
        candidates = self._find_preemptable_jobs(job, running_jobs)
        if not candidates:
            return SchedulingResult(
                job=job, success=False,
                reason="没有可抢占的低优先级任务"
            )

        logger.info(f"  找到 {len(candidates)} 个可抢占的任务")

        # 按抢占代价排序（代价 = 优先级 × 已运行时间）
        candidates.sort(key=lambda j: self._preemption_cost(j))

        # 逐个模拟抢占，直到资源满足
        preempted = []
        freed_gpus_by_node: dict[str, int] = {}

        for candidate in candidates:
            if len(preempted) >= self.config.max_preemptions_per_cycle:
                break

            node_name = candidate.assigned_node
            if node_name not in freed_gpus_by_node:
                freed_gpus_by_node[node_name] = 0

            freed_gpus_by_node[node_name] += candidate.gpu_count
            preempted.append(candidate)

            # 检查是否有节点能满足需求
            for nname, freed in freed_gpus_by_node.items():
                node = nodes.get(nname)
                if node and (node.available_gpus + freed) >= job.gpu_count:
                    # 找到了！执行抢占
                    self._execute_preemption(preempted, job)
                    return SchedulingResult(
                        job=job,
                        node=nname,
                        success=True,
                        reason=f"抢占 {[j.name for j in preempted]} 释放资源",
                        preempted_jobs=[j.id for j in preempted],
                    )

        return SchedulingResult(
            job=job, success=False,
            reason="抢占后资源仍然不足"
        )

    def _find_preemptable_jobs(self, requesting_job, running_jobs: dict) -> list:
        """找出可以被抢占的任务"""
        candidates = []
        now = datetime.now()

        for job_id, job in running_jobs.items():
            # 条件 1: 优先级差距够大
            if requesting_job.priority - job.priority < self.config.min_priority_gap:
                continue

            # 条件 2: 不在保护期内（刚启动的任务）
            if job.start_time:
                running_minutes = (now - job.start_time).total_seconds() / 60
                if running_minutes < self.config.protect_running_minutes:
                    continue

            # 条件 3: 不在冷却期
            if job_id in self._cooldown_map:
                cooldown_elapsed = (now - self._cooldown_map[job_id]).total_seconds()
                if cooldown_elapsed < self.config.cooldown_seconds:
                    continue

            # 条件 4: 同一租户的任务不互相抢占（可选策略）
            if job.tenant == requesting_job.tenant:
                continue

            candidates.append(job)

        return candidates

    def _preemption_cost(self, job) -> float:
        """计算抢占某个任务的代价（越低越优先被抢占）"""
        # 代价 = 优先级 × log(已运行时间+1)
        import math
        running_seconds = 0
        if job.start_time:
            running_seconds = (datetime.now() - job.start_time).total_seconds()

        return job.priority * math.log1p(running_seconds / 3600)

    def _execute_preemption(self, victims: list, requester):
        """执行抢占"""
        for victim in victims:
            logger.warning(
                f"  抢占: {victim.name} (priority={victim.priority}, "
                f"tenant={victim.tenant}) 被 {requester.name} "
                f"(priority={requester.priority}) 抢占"
            )

            # 记录抢占历史
            self._preemption_history.append({
                "time": datetime.now().isoformat(),
                "victim": victim.name,
                "victim_priority": victim.priority,
                "requester": requester.name,
                "requester_priority": requester.priority,
                "freed_gpus": victim.gpu_count,
            })

            # 设置冷却
            self._cooldown_map[victim.id] = datetime.now()

    def get_preemption_stats(self) -> dict:
        """获取抢占统计"""
        return {
            "total_preemptions": len(self._preemption_history),
            "recent_preemptions": self._preemption_history[-10:],
        }
