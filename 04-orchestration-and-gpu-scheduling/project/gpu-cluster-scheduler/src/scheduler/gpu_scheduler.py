"""
GPU 调度器核心 — 主调度循环

实现一个类似 kube-scheduler 的调度循环：
  1. 从队列获取待调度的 GPU Job
  2. Filter: 过滤不满足条件的节点
  3. Score: 对候选节点打分
  4. Bind: 将 Job 绑定到最优节点
  5. 处理抢占（如果没有可用节点）
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from datetime import datetime

from .topology_aware import TopologyScorer
from .bin_packing import BinPackingScorer
from .preemption import PreemptionManager

logger = logging.getLogger(__name__)


class SchedulingStrategy(Enum):
    BIN_PACKING = "bin_packing"
    SPREAD = "spread"
    TOPOLOGY_AWARE = "topology_aware"


class JobState(Enum):
    PENDING = "pending"
    SCHEDULING = "scheduling"
    RUNNING = "running"
    PREEMPTED = "preempted"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GPUJob:
    """GPU 任务描述"""
    id: str
    name: str
    tenant: str
    gpu_count: int
    gpu_memory_min_gb: int = 0
    cpu_request: float = 0.0
    memory_request_gb: float = 0.0
    priority: int = 5                    # 1-10
    prefer_nvlink: bool = True           # 是否优先 NVLink 互连
    state: JobState = JobState.PENDING
    assigned_node: Optional[str] = None
    assigned_gpus: list[int] = field(default_factory=list)
    submit_time: datetime = field(default_factory=datetime.now)
    start_time: Optional[datetime] = None
    image: str = ""
    command: list[str] = field(default_factory=list)
    tolerations: dict = field(default_factory=dict)


@dataclass
class NodeInfo:
    """节点信息"""
    name: str
    total_gpus: int
    allocated_gpus: int
    gpu_type: str
    gpu_memory_gb: int
    total_cpu: float
    allocated_cpu: float
    total_memory_gb: float
    allocated_memory_gb: float
    healthy: bool = True
    taints: list[str] = field(default_factory=list)
    labels: dict = field(default_factory=dict)

    @property
    def available_gpus(self) -> int:
        return self.total_gpus - self.allocated_gpus

    @property
    def available_cpu(self) -> float:
        return self.total_cpu - self.allocated_cpu

    @property
    def available_memory_gb(self) -> float:
        return self.total_memory_gb - self.allocated_memory_gb


@dataclass
class SchedulingResult:
    """调度结果"""
    job: GPUJob
    node: Optional[str] = None
    gpu_indices: list[int] = field(default_factory=list)
    success: bool = False
    reason: str = ""
    preempted_jobs: list[str] = field(default_factory=list)


class GPUScheduler:
    """GPU 集群调度器"""

    def __init__(
        self,
        strategy: SchedulingStrategy = SchedulingStrategy.TOPOLOGY_AWARE,
        topology_weight: float = 0.4,
        bin_packing_weight: float = 0.4,
        resource_balance_weight: float = 0.2,
        enable_preemption: bool = True,
        scheduling_interval: float = 1.0,
    ):
        self.strategy = strategy
        self.topology_weight = topology_weight
        self.bin_packing_weight = bin_packing_weight
        self.resource_balance_weight = resource_balance_weight
        self.enable_preemption = enable_preemption
        self.scheduling_interval = scheduling_interval

        # 组件
        self.topology_scorer = TopologyScorer()
        self.bin_packing_scorer = BinPackingScorer()
        self.preemption_manager = PreemptionManager()

        # 状态
        self._job_queue: list[GPUJob] = []
        self._nodes: dict[str, NodeInfo] = {}
        self._running_jobs: dict[str, GPUJob] = {}

        logger.info(f"GPU 调度器初始化: strategy={strategy.value}")

    def update_nodes(self, nodes: list[NodeInfo]):
        """更新集群节点信息"""
        self._nodes = {n.name: n for n in nodes}

    def submit_job(self, job: GPUJob):
        """提交 GPU 任务"""
        job.state = JobState.PENDING
        job.submit_time = datetime.now()
        self._job_queue.append(job)
        self._sort_queue()
        logger.info(f"任务 {job.name} 已提交 (tenant={job.tenant}, "
                    f"gpu={job.gpu_count}, priority={job.priority})")

    def _sort_queue(self):
        """按优先级排序队列"""
        self._job_queue.sort(key=lambda j: (-j.priority, j.submit_time))

    def schedule_one(self) -> Optional[SchedulingResult]:
        """调度队列中的下一个任务"""
        if not self._job_queue:
            return None

        job = self._job_queue[0]
        job.state = JobState.SCHEDULING

        logger.info(f"调度任务: {job.name} (gpu={job.gpu_count})")

        # Phase 1: Filter — 过滤不满足条件的节点
        feasible_nodes = self._filter_nodes(job)
        logger.info(f"  Filter: {len(feasible_nodes)}/{len(self._nodes)} 节点可用")

        if not feasible_nodes:
            # 没有可用节点 — 尝试抢占
            if self.enable_preemption:
                result = self._try_preemption(job)
                if result.success:
                    self._job_queue.pop(0)
                    return result
            # 无法调度，放回队列
            job.state = JobState.PENDING
            return SchedulingResult(job=job, success=False, reason="无可用节点")

        # Phase 2: Score — 对候选节点打分
        scored_nodes = self._score_nodes(job, feasible_nodes)
        best_node_name = scored_nodes[0][0]
        best_node = self._nodes[best_node_name]

        logger.info(f"  Score: 最优节点 {best_node_name} (分数 {scored_nodes[0][1]:.1f})")

        # Phase 3: Select GPUs — 选择具体的 GPU
        gpu_indices = self.topology_scorer.select_gpus(
            best_node_name, job.gpu_count
        )

        # Phase 4: Bind — 绑定
        self._bind_job(job, best_node_name, gpu_indices)
        self._job_queue.pop(0)

        return SchedulingResult(
            job=job,
            node=best_node_name,
            gpu_indices=gpu_indices,
            success=True,
            reason=f"调度到 {best_node_name}, GPU={gpu_indices}",
        )

    def _filter_nodes(self, job: GPUJob) -> list[NodeInfo]:
        """Filter 阶段：过滤不满足条件的节点"""
        feasible = []
        for node in self._nodes.values():
            # 检查节点健康
            if not node.healthy:
                continue

            # 检查 GPU 数量
            if node.available_gpus < job.gpu_count:
                continue

            # 检查 GPU 显存
            if job.gpu_memory_min_gb > 0 and node.gpu_memory_gb < job.gpu_memory_min_gb:
                continue

            # 检查 CPU/内存
            if node.available_cpu < job.cpu_request:
                continue
            if node.available_memory_gb < job.memory_request_gb:
                continue

            # 检查 Taint 容忍
            if not self._check_tolerations(job, node):
                continue

            feasible.append(node)

        return feasible

    def _score_nodes(
        self,
        job: GPUJob,
        nodes: list[NodeInfo],
    ) -> list[tuple[str, float]]:
        """Score 阶段：对候选节点打分"""
        scores = []

        for node in nodes:
            # 拓扑分数
            topo_score = self.topology_scorer.score(node.name, job.gpu_count)

            # Bin Packing 分数
            bp_score = self.bin_packing_scorer.score(
                allocated=node.allocated_gpus,
                total=node.total_gpus,
                requested=job.gpu_count,
            )

            # 资源均衡分数
            balance_score = self._resource_balance_score(node, job)

            # 加权总分
            total = (
                topo_score * self.topology_weight +
                bp_score * self.bin_packing_weight +
                balance_score * self.resource_balance_weight
            )

            scores.append((node.name, total))
            logger.debug(f"    {node.name}: topo={topo_score:.1f} "
                        f"bp={bp_score:.1f} bal={balance_score:.1f} "
                        f"total={total:.1f}")

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def _resource_balance_score(self, node: NodeInfo, job: GPUJob) -> float:
        """计算资源均衡分数"""
        if node.total_cpu == 0 or node.total_memory_gb == 0:
            return 50.0

        gpu_util = (node.allocated_gpus + job.gpu_count) / node.total_gpus
        cpu_util = (node.allocated_cpu + job.cpu_request) / node.total_cpu
        mem_util = (node.allocated_memory_gb + job.memory_request_gb) / node.total_memory_gb

        # 各资源利用率越接近越好（避免 GPU 满了但 CPU 空闲的不均衡）
        variance = (
            (gpu_util - cpu_util) ** 2 +
            (gpu_util - mem_util) ** 2 +
            (cpu_util - mem_util) ** 2
        ) / 3

        # variance 越小分数越高
        return max(0, (1 - variance) * 100)

    def _check_tolerations(self, job: GPUJob, node: NodeInfo) -> bool:
        """检查 Job 是否容忍节点的 Taint"""
        for taint in node.taints:
            if taint not in job.tolerations:
                return False
        return True

    def _bind_job(self, job: GPUJob, node_name: str, gpu_indices: list[int]):
        """将 Job 绑定到节点"""
        job.state = JobState.RUNNING
        job.assigned_node = node_name
        job.assigned_gpus = gpu_indices
        job.start_time = datetime.now()

        # 更新节点资源
        node = self._nodes[node_name]
        node.allocated_gpus += job.gpu_count
        node.allocated_cpu += job.cpu_request
        node.allocated_memory_gb += job.memory_request_gb

        self._running_jobs[job.id] = job
        logger.info(f"  Bind: {job.name} → {node_name} GPU={gpu_indices}")

    def _try_preemption(self, job: GPUJob) -> SchedulingResult:
        """尝试抢占低优先级任务"""
        return self.preemption_manager.try_preempt(
            job=job,
            running_jobs=self._running_jobs,
            nodes=self._nodes,
        )

    def release_job(self, job_id: str):
        """释放已完成/失败的 Job 资源"""
        if job_id not in self._running_jobs:
            return

        job = self._running_jobs.pop(job_id)
        if job.assigned_node and job.assigned_node in self._nodes:
            node = self._nodes[job.assigned_node]
            node.allocated_gpus -= job.gpu_count
            node.allocated_cpu -= job.cpu_request
            node.allocated_memory_gb -= job.memory_request_gb

        job.state = JobState.COMPLETED
        logger.info(f"任务 {job.name} 释放资源: node={job.assigned_node}")

    def get_queue_status(self) -> dict:
        """获取队列状态"""
        return {
            "pending": len(self._job_queue),
            "running": len(self._running_jobs),
            "queue": [
                {
                    "name": j.name,
                    "tenant": j.tenant,
                    "gpu_count": j.gpu_count,
                    "priority": j.priority,
                    "wait_time_sec": (datetime.now() - j.submit_time).total_seconds(),
                }
                for j in self._job_queue
            ],
        }

    def run(self):
        """主调度循环"""
        logger.info("GPU 调度器启动主循环")
        while True:
            try:
                result = self.schedule_one()
                if result and result.success:
                    logger.info(f"调度成功: {result.job.name} → {result.node}")
                elif result and not result.success:
                    logger.debug(f"调度失败: {result.job.name} — {result.reason}")
            except Exception as e:
                logger.error(f"调度循环异常: {e}", exc_info=True)

            time.sleep(self.scheduling_interval)
