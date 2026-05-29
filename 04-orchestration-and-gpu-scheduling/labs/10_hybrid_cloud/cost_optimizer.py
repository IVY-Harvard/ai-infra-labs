"""
混合云 GPU 成本优化调度器

根据以下因素做出调度决策：
  1. 各集群的 GPU 可用性
  2. 计算成本（私有集群 vs 云 GPU 按需/Spot）
  3. 数据传输成本和时间
  4. 任务优先级和 deadline
  5. GPU 类型匹配度

使用方式：
    python cost_optimizer.py --config=cluster_config.yaml
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta
from enum import Enum

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class GPUType(Enum):
    H20 = "H20"
    A100 = "A100"
    H100 = "H100"


class InstanceType(Enum):
    ON_PREMISE = "on_premise"     # 私有集群（固定成本）
    ON_DEMAND = "on_demand"       # 按需实例
    SPOT = "spot"                 # 竞价实例


@dataclass
class ClusterInfo:
    """集群信息"""
    name: str
    gpu_type: GPUType
    total_gpus: int
    available_gpus: int
    instance_type: InstanceType
    cost_per_gpu_hour: float       # 每 GPU 每小时成本（美元）
    data_transfer_cost_per_gb: float = 0.0  # 数据传输成本
    network_bandwidth_gbps: float = 10.0    # 到数据源的带宽
    spot_interruption_rate: float = 0.0     # Spot 中断概率 (0-1)
    region: str = "local"


@dataclass
class GPUTask:
    """GPU 任务"""
    name: str
    gpu_count: int
    estimated_hours: float
    gpu_memory_min_gb: int = 40
    priority: int = 5              # 1-10, 10 最高
    data_size_gb: float = 0.0      # 需要传输的数据量
    data_location: str = "local"   # 数据所在集群
    deadline: Optional[datetime] = None
    preferred_gpu_types: list[GPUType] = field(default_factory=lambda: list(GPUType))
    spot_acceptable: bool = True


@dataclass
class SchedulingDecision:
    """调度决策结果"""
    task_name: str
    target_cluster: str
    estimated_cost: float
    estimated_start_time: datetime
    estimated_completion_time: datetime
    reasoning: str
    score: float


class CostOptimizer:
    """成本优化调度器"""

    # GPU 显存规格 (GB)
    GPU_MEMORY = {
        GPUType.H20: 96,
        GPUType.A100: 80,
        GPUType.H100: 80,
    }

    # 训练性能相对系数（以 A100 为基准 1.0）
    GPU_PERFORMANCE = {
        GPUType.H20: 0.8,     # H20 训练性能约为 A100 的 80%
        GPUType.A100: 1.0,
        GPUType.H100: 2.0,    # H100 约为 A100 的 2x
    }

    def __init__(
        self,
        clusters: list[ClusterInfo],
        opportunity_cost_per_gpu_hour: float = 1.0,  # 等待的机会成本
        data_transfer_speed_gbps: float = 1.0,       # 默认传输速度
    ):
        self.clusters = clusters
        self.opportunity_cost = opportunity_cost_per_gpu_hour
        self.transfer_speed = data_transfer_speed_gbps

    def schedule(self, task: GPUTask) -> SchedulingDecision:
        """为任务做出最优调度决策"""
        logger.info(f"为任务 '{task.name}' 计算调度方案...")
        logger.info(f"  需要: {task.gpu_count} GPU, 预计 {task.estimated_hours}h, "
                    f"优先级 {task.priority}")

        candidates = []
        for cluster in self.clusters:
            score, cost, reasoning = self._evaluate_cluster(cluster, task)
            if score > 0:
                candidates.append((cluster, score, cost, reasoning))

        if not candidates:
            return SchedulingDecision(
                task_name=task.name,
                target_cluster="NONE",
                estimated_cost=0,
                estimated_start_time=datetime.now(),
                estimated_completion_time=datetime.now(),
                reasoning="无可用集群满足任务需求",
                score=0,
            )

        # 按分数排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_cluster, best_score, best_cost, best_reasoning = candidates[0]

        # 计算时间
        transfer_time = self._estimate_transfer_time(best_cluster, task)
        start_time = datetime.now() + timedelta(hours=transfer_time)
        completion_time = start_time + timedelta(hours=task.estimated_hours)

        decision = SchedulingDecision(
            task_name=task.name,
            target_cluster=best_cluster.name,
            estimated_cost=best_cost,
            estimated_start_time=start_time,
            estimated_completion_time=completion_time,
            reasoning=best_reasoning,
            score=best_score,
        )

        logger.info(f"  决策: → {decision.target_cluster}")
        logger.info(f"  成本: ${decision.estimated_cost:.2f}")
        logger.info(f"  原因: {decision.reasoning}")

        return decision

    def _evaluate_cluster(
        self,
        cluster: ClusterInfo,
        task: GPUTask,
    ) -> tuple[float, float, str]:
        """
        评估集群适合度。

        Returns:
            (score, estimated_cost, reasoning)
            score: 0-100, 越高越好
        """
        reasons = []

        # 检查硬约束
        if cluster.available_gpus < task.gpu_count:
            return 0, 0, f"GPU 不足 ({cluster.available_gpus} < {task.gpu_count})"

        if self.GPU_MEMORY[cluster.gpu_type] < task.gpu_memory_min_gb:
            return 0, 0, f"显存不足 ({self.GPU_MEMORY[cluster.gpu_type]}GB < {task.gpu_memory_min_gb}GB)"

        if cluster.instance_type == InstanceType.SPOT and not task.spot_acceptable:
            return 0, 0, "任务不接受 Spot 实例"

        # 1. 成本分数 (40%)
        compute_cost = cluster.cost_per_gpu_hour * task.gpu_count * task.estimated_hours
        transfer_cost = cluster.data_transfer_cost_per_gb * task.data_size_gb
        total_cost = compute_cost + transfer_cost

        # 归一化成本分数（成本越低分数越高）
        max_cost = 10.0 * task.gpu_count * task.estimated_hours  # 假设最高 $10/GPU/h
        cost_score = max(0, (1 - total_cost / max_cost)) * 100
        reasons.append(f"成本${total_cost:.1f}")

        # 2. 数据亲和性分数 (25%)
        if task.data_location == cluster.region or task.data_location == "local" and cluster.instance_type == InstanceType.ON_PREMISE:
            data_score = 100
            reasons.append("数据本地")
        elif task.data_size_gb < 100:
            data_score = 60
            reasons.append(f"需传输{task.data_size_gb:.0f}GB")
        else:
            data_score = 20
            reasons.append(f"大量数据传输{task.data_size_gb:.0f}GB")

        # 3. 等待时间分数 (20%)
        if cluster.available_gpus >= task.gpu_count:
            wait_score = 100
            reasons.append("立即可用")
        else:
            wait_score = 0

        # 4. GPU 匹配度分数 (10%)
        if task.preferred_gpu_types and cluster.gpu_type in task.preferred_gpu_types:
            match_score = 100
        else:
            match_score = 50
        reasons.append(f"GPU={cluster.gpu_type.value}")

        # 5. 风险分数 (5%)
        risk_score = (1 - cluster.spot_interruption_rate) * 100

        # 加权总分
        total_score = (
            cost_score * 0.40 +
            data_score * 0.25 +
            wait_score * 0.20 +
            match_score * 0.10 +
            risk_score * 0.05
        )

        reasoning = f"[{cluster.name}] " + ", ".join(reasons) + f" → 分数{total_score:.1f}"
        return total_score, total_cost, reasoning

    def _estimate_transfer_time(self, cluster: ClusterInfo, task: GPUTask) -> float:
        """估算数据传输时间（小时）"""
        if task.data_location == cluster.region:
            return 0.0
        if task.data_size_gb == 0:
            return 0.0

        # 传输时间 = 数据量 / 带宽
        transfer_hours = (task.data_size_gb * 8) / (cluster.network_bandwidth_gbps * 3600)
        return transfer_hours

    def batch_schedule(self, tasks: list[GPUTask]) -> list[SchedulingDecision]:
        """批量调度多个任务"""
        # 按优先级排序
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)

        decisions = []
        # 模拟资源消耗
        remaining_gpus = {c.name: c.available_gpus for c in self.clusters}

        for task in sorted_tasks:
            # 临时更新可用 GPU
            temp_clusters = []
            for c in self.clusters:
                temp = ClusterInfo(
                    name=c.name,
                    gpu_type=c.gpu_type,
                    total_gpus=c.total_gpus,
                    available_gpus=remaining_gpus[c.name],
                    instance_type=c.instance_type,
                    cost_per_gpu_hour=c.cost_per_gpu_hour,
                    data_transfer_cost_per_gb=c.data_transfer_cost_per_gb,
                    network_bandwidth_gbps=c.network_bandwidth_gbps,
                    spot_interruption_rate=c.spot_interruption_rate,
                    region=c.region,
                )
                temp_clusters.append(temp)

            optimizer = CostOptimizer(temp_clusters, self.opportunity_cost)
            decision = optimizer.schedule(task)
            decisions.append(decision)

            # 更新剩余资源
            if decision.target_cluster != "NONE":
                remaining_gpus[decision.target_cluster] -= task.gpu_count

        return decisions


def demo():
    """演示混合云调度"""
    # 定义集群
    clusters = [
        ClusterInfo(
            name="onprem-h20",
            gpu_type=GPUType.H20,
            total_gpus=8,
            available_gpus=4,           # 当前 4 GPU 空闲
            instance_type=InstanceType.ON_PREMISE,
            cost_per_gpu_hour=0.50,     # 边际成本（电费+折旧）
            data_transfer_cost_per_gb=0.0,
            region="local",
        ),
        ClusterInfo(
            name="aws-a100-spot",
            gpu_type=GPUType.A100,
            total_gpus=64,
            available_gpus=32,
            instance_type=InstanceType.SPOT,
            cost_per_gpu_hour=1.20,
            data_transfer_cost_per_gb=0.09,
            network_bandwidth_gbps=10.0,
            spot_interruption_rate=0.05,
            region="us-east-1",
        ),
        ClusterInfo(
            name="gcp-h100-ondemand",
            gpu_type=GPUType.H100,
            total_gpus=32,
            available_gpus=16,
            instance_type=InstanceType.ON_DEMAND,
            cost_per_gpu_hour=4.50,
            data_transfer_cost_per_gb=0.12,
            network_bandwidth_gbps=25.0,
            region="us-central1",
        ),
    ]

    optimizer = CostOptimizer(clusters)

    # 定义任务
    tasks = [
        GPUTask(
            name="llama-70b-finetune",
            gpu_count=8,
            estimated_hours=24.0,
            gpu_memory_min_gb=80,
            priority=8,
            data_size_gb=50.0,
            data_location="local",
            preferred_gpu_types=[GPUType.H20, GPUType.A100, GPUType.H100],
        ),
        GPUTask(
            name="small-model-experiment",
            gpu_count=2,
            estimated_hours=4.0,
            gpu_memory_min_gb=20,
            priority=3,
            data_size_gb=10.0,
            data_location="local",
            spot_acceptable=True,
        ),
        GPUTask(
            name="inference-benchmark",
            gpu_count=1,
            estimated_hours=2.0,
            gpu_memory_min_gb=40,
            priority=5,
            data_size_gb=5.0,
            data_location="local",
        ),
    ]

    print("=" * 70)
    print("混合云 GPU 成本优化调度 Demo")
    print("=" * 70)
    print(f"\n集群资源:")
    for c in clusters:
        print(f"  {c.name}: {c.available_gpus}/{c.total_gpus} GPU "
              f"({c.gpu_type.value}, ${c.cost_per_gpu_hour}/GPU/h)")

    print(f"\n待调度任务:")
    for t in tasks:
        print(f"  {t.name}: {t.gpu_count} GPU × {t.estimated_hours}h, 优先级{t.priority}")

    print("\n" + "=" * 70)
    print("调度决策:")
    print("=" * 70)

    decisions = optimizer.batch_schedule(tasks)

    total_cost = 0
    for d in decisions:
        print(f"\n任务: {d.task_name}")
        print(f"  → 集群: {d.target_cluster}")
        print(f"  → 成本: ${d.estimated_cost:.2f}")
        print(f"  → 开始: {d.estimated_start_time.strftime('%H:%M')}")
        print(f"  → 完成: {d.estimated_completion_time.strftime('%H:%M')}")
        print(f"  → 原因: {d.reasoning}")
        total_cost += d.estimated_cost

    print(f"\n总成本: ${total_cost:.2f}")
    print(f"对比全部云上: ${sum(t.gpu_count * t.estimated_hours * 4.5 for t in tasks):.2f}")


if __name__ == "__main__":
    demo()
