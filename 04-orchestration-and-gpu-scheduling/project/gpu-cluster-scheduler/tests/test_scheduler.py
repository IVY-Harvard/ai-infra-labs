"""
GPU 调度器单元测试
"""

import pytest
from datetime import datetime, timedelta

# 将 src 加入路径
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scheduler.gpu_scheduler import (
    GPUScheduler, GPUJob, NodeInfo, SchedulingStrategy, JobState
)
from src.scheduler.topology_aware import TopologyScorer, GPUTopologyInfo, build_h20_topology
from src.scheduler.bin_packing import BinPackingScorer
from src.scheduler.preemption import PreemptionManager, PreemptionConfig
from src.tenant.quota_manager import QuotaManager, TenantQuota
from src.tenant.fair_share import FairShareCalculator, TenantDemand


class TestTopologyScorer:
    """测试拓扑感知调度"""

    def setup_method(self):
        self.scorer = TopologyScorer()
        # 注册一个 8-GPU H20 拓扑
        topo = build_h20_topology("gpu-node-0")
        self.scorer.register_node(topo)

    def test_score_full_nvlink_group(self):
        """请求 4 GPU，应该获得 NVLink 组满分"""
        score = self.scorer.score("gpu-node-0", 4)
        assert score == TopologyScorer.SCORE_SAME_NVLINK_GROUP

    def test_score_cross_nvlink(self):
        """请求 6 GPU，需要跨 NVLink 组"""
        score = self.scorer.score("gpu-node-0", 6)
        assert score == TopologyScorer.SCORE_CROSS_NUMA

    def test_score_insufficient_gpus(self):
        """请求超过可用 GPU 数量"""
        # 标记 6 个 GPU 已分配
        self.scorer.update_allocation("gpu-node-0", {0, 1, 2, 3, 4, 5})
        score = self.scorer.score("gpu-node-0", 4)
        assert score == TopologyScorer.SCORE_INSUFFICIENT

    def test_select_gpus_prefers_nvlink(self):
        """GPU 选择应优先同一 NVLink 组"""
        gpus = self.scorer.select_gpus("gpu-node-0", 4)
        # 应该选 [0,1,2,3] 或 [4,5,6,7]（同一 NVLink 组）
        assert len(gpus) == 4
        assert all(g < 4 for g in gpus) or all(g >= 4 for g in gpus)

    def test_select_gpus_with_partial_allocation(self):
        """部分分配后的 GPU 选择"""
        self.scorer.update_allocation("gpu-node-0", {0, 1})
        gpus = self.scorer.select_gpus("gpu-node-0", 4)
        # NVLink 组 A 只剩 2 个(2,3)，不够 → 选组 B [4,5,6,7]
        assert len(gpus) == 4
        assert all(g >= 4 for g in gpus)

    def test_unknown_node(self):
        """未知节点返回默认分数"""
        score = self.scorer.score("unknown-node", 2)
        assert score == 50.0


class TestBinPackingScorer:
    """测试 Bin Packing 策略"""

    def setup_method(self):
        self.scorer = BinPackingScorer()

    def test_prefer_fuller_node(self):
        """应优先选择已经更满的节点"""
        score_full = self.scorer.score(allocated=6, total=8, requested=2)
        score_empty = self.scorer.score(allocated=0, total=8, requested=2)
        assert score_full > score_empty

    def test_insufficient_gpus(self):
        """GPU 不足时分数为 0"""
        score = self.scorer.score(allocated=7, total=8, requested=2)
        assert score == 0.0

    def test_perfect_fit(self):
        """恰好填满，无碎片"""
        score = self.scorer.score(allocated=6, total=8, requested=2)
        # 填满后 utilization = 100%，无碎片惩罚
        assert score == 100.0

    def test_fragmentation_penalty(self):
        """产生碎片时有惩罚"""
        # 分配后剩余 3 GPU（不是常见粒度的整数倍）
        score_3_remaining = self.scorer.score(allocated=3, total=8, requested=2)
        # 分配后剩余 4 GPU（是常见粒度）
        score_4_remaining = self.scorer.score(allocated=2, total=8, requested=2)
        # 剩 4 没惩罚，剩 3 有惩罚
        assert score_4_remaining >= score_3_remaining

    def test_rank_nodes(self):
        """多节点排名"""
        nodes = [
            {"name": "node-a", "allocated": 6, "total": 8},
            {"name": "node-b", "allocated": 2, "total": 8},
            {"name": "node-c", "allocated": 4, "total": 8},
        ]
        ranked = self.scorer.rank_nodes(nodes, requested_gpus=2)
        # 应该 node-a（最满）排第一
        assert ranked[0][0] == "node-a"


class TestQuotaManager:
    """测试配额管理"""

    def setup_method(self):
        self.qm = QuotaManager(total_cluster_gpus=8)
        self.qm.register_tenant(TenantQuota(
            "team-a", gpu_quota=4, gpu_burst_limit=6
        ))
        self.qm.register_tenant(TenantQuota(
            "team-b", gpu_quota=4, gpu_burst_limit=6,
            can_be_borrowed_from=True
        ))

    def test_within_quota(self):
        """在配额内的请求应该通过"""
        allowed, _ = self.qm.check_quota("team-a", 4)
        assert allowed

    def test_exceed_quota_with_borrowing(self):
        """超出配额但可以借用"""
        self.qm.allocate("team-a", 4)  # 用完基础配额
        # 请求额外 2 GPU（需要借用）
        allowed, reason = self.qm.check_quota("team-a", 2)
        assert allowed
        assert "借用" in reason

    def test_exceed_burst_limit(self):
        """超出 burst limit 被拒绝"""
        self.qm.allocate("team-a", 4)
        self.qm.allocate("team-a", 2)  # 现在 6 GPU（=burst limit）
        allowed, _ = self.qm.check_quota("team-a", 1)
        assert not allowed

    def test_release_resources(self):
        """释放资源后可以再次分配"""
        self.qm.allocate("team-a", 4)
        self.qm.release("team-a", 4)
        allowed, _ = self.qm.check_quota("team-a", 4)
        assert allowed

    def test_unknown_tenant(self):
        """未知租户被拒绝"""
        allowed, _ = self.qm.check_quota("unknown", 1)
        assert not allowed


class TestFairShare:
    """测试公平共享算法"""

    def setup_method(self):
        self.calc = FairShareCalculator(total_gpus=8, total_cpus=64)

    def test_equal_demand(self):
        """相同权重和需求应该平均分配"""
        tenants = [
            TenantDemand("a", gpu_demand=4, cpu_demand=32, gpu_allocated=0, cpu_allocated=0),
            TenantDemand("b", gpu_demand=4, cpu_demand=32, gpu_allocated=0, cpu_allocated=0),
        ]
        allocation = self.calc.compute_fair_allocation(tenants)
        alloc_dict = dict(allocation)
        assert alloc_dict["a"] == 4
        assert alloc_dict["b"] == 4

    def test_weighted_allocation(self):
        """权重高的租户应该获得更多资源"""
        tenants = [
            TenantDemand("a", gpu_demand=8, cpu_demand=64, gpu_allocated=0,
                        cpu_allocated=0, weight=2.0),
            TenantDemand("b", gpu_demand=8, cpu_demand=64, gpu_allocated=0,
                        cpu_allocated=0, weight=1.0),
        ]
        allocation = self.calc.compute_fair_allocation(tenants)
        alloc_dict = dict(allocation)
        # weight=2 的应该分配更多
        assert alloc_dict["a"] > alloc_dict["b"]

    def test_limited_demand(self):
        """需求有限的租户不会超分配"""
        tenants = [
            TenantDemand("a", gpu_demand=2, cpu_demand=16, gpu_allocated=0, cpu_allocated=0),
            TenantDemand("b", gpu_demand=8, cpu_demand=64, gpu_allocated=0, cpu_allocated=0),
        ]
        allocation = self.calc.compute_fair_allocation(tenants)
        alloc_dict = dict(allocation)
        assert alloc_dict["a"] == 2  # 不超过需求
        assert alloc_dict["b"] == 6  # 剩余都给 b


class TestGPUScheduler:
    """测试调度器主流程"""

    def setup_method(self):
        self.scheduler = GPUScheduler(
            strategy=SchedulingStrategy.TOPOLOGY_AWARE,
            enable_preemption=True,
        )
        # 注册拓扑
        topo = build_h20_topology("gpu-node-0")
        self.scheduler.topology_scorer.register_node(topo)
        topo2 = build_h20_topology("gpu-node-1")
        self.scheduler.topology_scorer.register_node(topo2)

        # 注册节点
        self.scheduler.update_nodes([
            NodeInfo("gpu-node-0", total_gpus=8, allocated_gpus=0,
                    gpu_type="H20", gpu_memory_gb=96,
                    total_cpu=64, allocated_cpu=0,
                    total_memory_gb=512, allocated_memory_gb=0),
            NodeInfo("gpu-node-1", total_gpus=8, allocated_gpus=4,
                    gpu_type="H20", gpu_memory_gb=96,
                    total_cpu=64, allocated_cpu=32,
                    total_memory_gb=512, allocated_memory_gb=256),
        ])

    def test_schedule_basic(self):
        """基本调度流程"""
        job = GPUJob(
            id="job-1", name="test-job", tenant="team-a",
            gpu_count=4, priority=5
        )
        self.scheduler.submit_job(job)
        result = self.scheduler.schedule_one()

        assert result is not None
        assert result.success
        assert result.node is not None
        assert len(result.gpu_indices) == 4

    def test_schedule_prefers_topology(self):
        """调度应该优先拓扑更好的节点"""
        job = GPUJob(
            id="job-2", name="topo-test", tenant="team-a",
            gpu_count=4, priority=5, prefer_nvlink=True
        )
        self.scheduler.submit_job(job)
        result = self.scheduler.schedule_one()

        # gpu-node-0 全空，拓扑更好（完整的 NVLink 组可用）
        assert result.success
        # 选出的 GPU 应该在同一 NVLink 组
        assert all(g < 4 for g in result.gpu_indices) or all(g >= 4 for g in result.gpu_indices)

    def test_schedule_insufficient_resources(self):
        """资源不足时调度失败"""
        # 两个节点分别只有 8 和 4 GPU，请求 10 GPU
        job = GPUJob(
            id="job-3", name="big-job", tenant="team-a",
            gpu_count=10, priority=5
        )
        self.scheduler.submit_job(job)
        result = self.scheduler.schedule_one()

        assert not result.success

    def test_priority_ordering(self):
        """高优先级任务应该先被调度"""
        low_job = GPUJob(id="low", name="low-pri", tenant="team-a",
                        gpu_count=2, priority=3)
        high_job = GPUJob(id="high", name="high-pri", tenant="team-a",
                         gpu_count=2, priority=9)

        self.scheduler.submit_job(low_job)
        self.scheduler.submit_job(high_job)

        # 第一个调度的应该是高优先级
        result = self.scheduler.schedule_one()
        assert result.job.name == "high-pri"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
