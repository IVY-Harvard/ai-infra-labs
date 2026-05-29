#!/usr/bin/env python3
"""UEC vs RoCE v2 vs InfiniBand 性能建模分析

本工具对比三种 RDMA 网络技术在 AI 训练场景中的理论性能:
1. InfiniBand NDR (400Gbps): 自适应路由 + 信用制流控
2. RoCE v2 (400GbE): ECMP 路由 + DCQCN 拥塞控制
3. UEC (800GbE): 包喷洒 + 多级 ECN + 选择性重传

分析维度:
- AllReduce 集合通信在不同拓扑下的性能
- Incast 场景下的吞吐与尾延迟
- 有效二分带宽利用率
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class NetworkConfig:
    """网络技术配置"""
    name: str
    link_speed_gbps: float          # 链路速率
    effective_bandwidth_ratio: float  # 有效带宽比 (考虑协议开销)
    multipath_efficiency: float      # 多路径效率 (0-1, 1=完美分散)
    congestion_overhead: float       # 拥塞控制开销 (占比)
    retransmit_overhead: float       # 重传开销 (占比)
    ecmp_collision_penalty: float    # ECMP 哈希冲突惩罚
    adaptive_routing: bool           # 是否支持自适应路由
    max_outstanding_bytes: int       # 最大在途字节数 (影响延迟隐藏)
    header_overhead_bytes: int       # 协议头开销

    @property
    def net_bandwidth_gbps(self) -> float:
        """净带宽 (扣除各项开销)"""
        return (self.link_speed_gbps
                * self.effective_bandwidth_ratio
                * (1.0 - self.congestion_overhead)
                * (1.0 - self.retransmit_overhead))


# 预定义网络配置
INFINIBAND_NDR = NetworkConfig(
    name="InfiniBand NDR",
    link_speed_gbps=400.0,
    effective_bandwidth_ratio=0.97,   # IB 协议开销低
    multipath_efficiency=0.85,        # 交换机自适应路由, 但非逐包
    congestion_overhead=0.02,         # 信用制低开销
    retransmit_overhead=0.001,        # 链路级可靠性, 极低重传
    ecmp_collision_penalty=0.0,       # 不使用 ECMP
    adaptive_routing=True,
    max_outstanding_bytes=256 * 1024,
    header_overhead_bytes=60,         # IB LRH + BTH + RETH
)

ROCE_V2 = NetworkConfig(
    name="RoCE v2 (400GbE)",
    link_speed_gbps=400.0,
    effective_bandwidth_ratio=0.94,   # UDP/IP 封装开销
    multipath_efficiency=0.60,        # ECMP 哈希冲突严重
    congestion_overhead=0.08,         # DCQCN 振荡开销
    retransmit_overhead=0.02,         # Go-back-N 重传浪费
    ecmp_collision_penalty=0.15,      # 哈希冲突导致热点
    adaptive_routing=False,
    max_outstanding_bytes=128 * 1024,
    header_overhead_bytes=78,         # ETH + IP + UDP + BTH + RETH
)

UEC_800G = NetworkConfig(
    name="UEC (800GbE)",
    link_speed_gbps=800.0,
    effective_bandwidth_ratio=0.95,   # 优化的传输协议
    multipath_efficiency=0.98,        # 逐包喷洒, 接近完美
    congestion_overhead=0.02,         # 多级 ECN 精确控制
    retransmit_overhead=0.005,        # 选择性重传, 仅重传丢失包
    ecmp_collision_penalty=0.0,       # 不使用 ECMP
    adaptive_routing=True,
    max_outstanding_bytes=512 * 1024,
    header_overhead_bytes=70,         # UEC 优化头部
)


@dataclass
class TopologyConfig:
    """网络拓扑配置"""
    name: str
    num_nodes: int
    num_switches: int
    num_paths: int           # 任意节点对间的等价路径数
    diameter_hops: int       # 最大跳数
    bisection_bandwidth_ratio: float  # 二分带宽比 (1.0 = 全二分)


# 常见拓扑
def make_fat_tree(k: int) -> TopologyConfig:
    """k-ary fat-tree 拓扑"""
    num_nodes = (k ** 3) // 4
    num_switches = 5 * (k ** 2) // 4
    num_paths = (k // 2) ** 2  # 任意对间的等价路径
    return TopologyConfig(
        name=f"Fat-tree (k={k})",
        num_nodes=num_nodes,
        num_switches=num_switches,
        num_paths=num_paths,
        diameter_hops=4,
        bisection_bandwidth_ratio=1.0,  # 全二分
    )


def make_dragonfly(groups: int, nodes_per_group: int) -> TopologyConfig:
    """Dragonfly 拓扑"""
    return TopologyConfig(
        name=f"Dragonfly ({groups}g × {nodes_per_group}n)",
        num_nodes=groups * nodes_per_group,
        num_switches=groups * (nodes_per_group // 4 + 1),
        num_paths=groups - 1,
        diameter_hops=3,
        bisection_bandwidth_ratio=0.5,  # 组间带宽通常为 1:2 收敛
    )


@dataclass
class AllReduceResult:
    """AllReduce 性能结果"""
    network: str
    topology: str
    algorithm: str
    num_nodes: int
    message_size_mb: float
    time_us: float
    bandwidth_gbps: float
    efficiency: float  # 相对于理论最优的效率


class PerformanceModeler:
    """性能建模分析工具"""

    def __init__(self):
        self.networks = [INFINIBAND_NDR, ROCE_V2, UEC_800G]

    def model_allreduce_ring(self, net: NetworkConfig, topo: TopologyConfig,
                             message_size_bytes: int) -> AllReduceResult:
        """建模 Ring AllReduce 性能

        Ring AllReduce 时间 = 2(n-1)/n × M/B + 2(n-1) × α
        其中 n=节点数, M=消息大小, B=带宽, α=延迟
        """
        n = topo.num_nodes
        # 有效带宽考虑多路径效率
        bw_bytes = net.net_bandwidth_gbps * 1e9 / 8
        # Ring 中每步传输 M/(n) 大小的块
        chunk_size = message_size_bytes / n
        # 基础延迟 (微秒)
        per_hop_latency_us = 0.5 * topo.diameter_hops
        # 总时间: 通信时间 + 延迟
        comm_time = 2 * (n - 1) / n * message_size_bytes / bw_bytes * 1e6
        latency_time = 2 * (n - 1) * per_hop_latency_us
        total_time = comm_time + latency_time

        # 实际带宽
        actual_bw = message_size_bytes * 8 / (total_time * 1e-6) / 1e9
        # 效率: 实际/理论
        theoretical_bw = net.net_bandwidth_gbps * 2 * (n - 1) / n
        efficiency = actual_bw / theoretical_bw if theoretical_bw > 0 else 0

        return AllReduceResult(
            network=net.name,
            topology=topo.name,
            algorithm="Ring",
            num_nodes=n,
            message_size_mb=message_size_bytes / 1e6,
            time_us=total_time,
            bandwidth_gbps=actual_bw,
            efficiency=efficiency,
        )

    def model_allreduce_tree(self, net: NetworkConfig, topo: TopologyConfig,
                             message_size_bytes: int) -> AllReduceResult:
        """建模 Tree (Recursive Halving-Doubling) AllReduce

        Tree AllReduce 时间 ≈ 2 × log2(n) × M/B + 2 × log2(n) × α
        受益于多路径: 每层可利用多条路径并行
        """
        n = topo.num_nodes
        log_n = math.ceil(math.log2(max(n, 2)))
        bw_bytes = net.net_bandwidth_gbps * 1e9 / 8

        # Tree 中每层传输 M/2 大小 (递归减半)
        # 多路径效率影响带宽
        effective_bw = bw_bytes * net.multipath_efficiency
        comm_time = 2 * log_n * message_size_bytes / effective_bw * 1e6
        per_hop_latency_us = 0.5 * topo.diameter_hops
        latency_time = 2 * log_n * per_hop_latency_us

        total_time = comm_time + latency_time
        actual_bw = message_size_bytes * 8 / (total_time * 1e-6) / 1e9
        theoretical_bw = net.net_bandwidth_gbps * net.multipath_efficiency
        efficiency = actual_bw / theoretical_bw if theoretical_bw > 0 else 0

        return AllReduceResult(
            network=net.name,
            topology=topo.name,
            algorithm="Tree",
            num_nodes=n,
            message_size_mb=message_size_bytes / 1e6,
            time_us=total_time,
            bandwidth_gbps=actual_bw,
            efficiency=efficiency,
        )

    def model_incast(self, net: NetworkConfig, num_senders: int,
                     message_size_bytes: int) -> Dict:
        """建模 Incast 场景 (多对一)

        Incast 是 AI 训练中 AllReduce reduce 阶段的典型模式
        """
        bw_bytes = net.net_bandwidth_gbps * 1e9 / 8

        # 总输入带宽
        total_input_bw = num_senders * bw_bytes

        # 过载比
        oversubscription = total_input_bw / bw_bytes

        # ECMP 场景: 哈希冲突加剧热点
        if not net.adaptive_routing:
            # 最繁忙路径的负载 (概率模型)
            # E[max_load] ≈ n/k + sqrt(2n*ln(k)/k) for k paths
            num_paths = 4  # 典型 ECMP 等价路径数
            expected_max = (num_senders / num_paths +
                           math.sqrt(2 * num_senders * math.log(num_paths) / num_paths))
            hotspot_factor = expected_max / (num_senders / num_paths)
        else:
            hotspot_factor = 1.05  # 自适应路由几乎完美均衡

        # 有效吞吐
        if net.multipath_efficiency > 0.9:
            # Spray: 均匀分散，队列浅
            effective_throughput = bw_bytes * 0.95
            queue_buildup_kb = 32  # 队列浅
        else:
            # ECMP: 热点路径深队列
            effective_throughput = bw_bytes * (1.0 / hotspot_factor) * 0.85
            queue_buildup_kb = 256 * hotspot_factor  # 队列深

        # 完成时间
        total_bytes = num_senders * message_size_bytes
        completion_time_us = total_bytes / effective_throughput * 1e6

        # 尾延迟 (P99)
        base_latency_us = 5.0
        if net.multipath_efficiency > 0.9:
            tail_latency_us = base_latency_us * 1.5  # Spray 尾延迟小
        else:
            tail_latency_us = base_latency_us * hotspot_factor * 3  # ECMP 尾延迟大

        return {
            "network": net.name,
            "num_senders": num_senders,
            "oversubscription": f"{oversubscription:.1f}x",
            "hotspot_factor": hotspot_factor,
            "effective_throughput_gbps": effective_throughput * 8 / 1e9,
            "completion_time_us": completion_time_us,
            "tail_latency_us": tail_latency_us,
            "queue_buildup_kb": queue_buildup_kb,
        }

    def model_bisection_bandwidth(self, net: NetworkConfig,
                                  topo: TopologyConfig) -> Dict:
        """建模有效二分带宽"""
        # 理论二分带宽
        theoretical_bw = (topo.num_nodes / 2 * net.link_speed_gbps
                          * topo.bisection_bandwidth_ratio)

        # 多路径效率影响有效二分带宽
        if net.adaptive_routing or net.multipath_efficiency > 0.9:
            path_efficiency = net.multipath_efficiency
        else:
            # ECMP 在大规模下效率下降
            # 冲突概率 ≈ 1 - (1 - 1/k)^n
            k = topo.num_paths
            n = topo.num_nodes
            collision_prob = 1 - (1 - 1/max(k, 1))**min(n, 100)
            path_efficiency = net.multipath_efficiency * (1 - collision_prob * 0.3)

        effective_bw = theoretical_bw * path_efficiency * net.effective_bandwidth_ratio

        return {
            "network": net.name,
            "topology": topo.name,
            "theoretical_bisection_gbps": theoretical_bw,
            "effective_bisection_gbps": effective_bw,
            "utilization_ratio": effective_bw / theoretical_bw if theoretical_bw > 0 else 0,
        }


def print_allreduce_comparison(results: List[AllReduceResult]):
    """打印 AllReduce 对比表格"""
    print(f"\n{'网络技术':<22}{'算法':<8}{'节点数':<8}{'消息(MB)':<10}"
          f"{'时间(μs)':<12}{'带宽(Gbps)':<12}{'效率':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r.network:<22}{r.algorithm:<8}{r.num_nodes:<8}"
              f"{r.message_size_mb:<10.0f}{r.time_us:<12.1f}"
              f"{r.bandwidth_gbps:<12.2f}{r.efficiency:<8.2%}")


def print_incast_comparison(results: List[Dict]):
    """打印 Incast 对比表格"""
    print(f"\n{'网络技术':<22}{'发送者':<8}{'过载比':<10}{'热点系数':<10}"
          f"{'有效吞吐(Gbps)':<16}{'完成(μs)':<12}{'P99延迟(μs)':<12}")
    print("-" * 100)
    for r in results:
        print(f"{r['network']:<22}{r['num_senders']:<8}{r['oversubscription']:<10}"
              f"{r['hotspot_factor']:<10.2f}{r['effective_throughput_gbps']:<16.1f}"
              f"{r['completion_time_us']:<12.1f}{r['tail_latency_us']:<12.1f}")


def main():
    """主函数: 运行完整性能分析"""
    print("=" * 90)
    print("UEC vs RoCE v2 vs InfiniBand 性能建模分析")
    print("面向 AI 训练集合通信的网络技术对比")
    print("=" * 90)

    modeler = PerformanceModeler()

    # 拓扑配置
    fat_tree_32 = make_fat_tree(k=8)      # 128 节点
    fat_tree_128 = make_fat_tree(k=16)    # 1024 节点
    dragonfly = make_dragonfly(groups=16, nodes_per_group=32)  # 512 节点

    # ========== AllReduce 性能对比 ==========
    print("\n" + "=" * 90)
    print("1. AllReduce 性能对比")
    print("=" * 90)

    message_sizes = [64 * 1024 * 1024, 256 * 1024 * 1024]  # 64MB, 256MB
    topologies = [fat_tree_32, fat_tree_128]

    for topo in topologies:
        for msg_size in message_sizes:
            print(f"\n--- {topo.name}, 消息大小: {msg_size // (1024*1024)} MB ---")
            results = []
            for net in modeler.networks:
                results.append(modeler.model_allreduce_ring(net, topo, msg_size))
                results.append(modeler.model_allreduce_tree(net, topo, msg_size))
            print_allreduce_comparison(results)

    # ========== Incast 场景对比 ==========
    print("\n" + "=" * 90)
    print("2. Incast 场景对比 (多对一通信)")
    print("=" * 90)

    incast_configs = [
        (8, 1 * 1024 * 1024),    # 8:1 incast, 1MB
        (32, 1 * 1024 * 1024),   # 32:1 incast, 1MB
        (64, 256 * 1024),        # 64:1 incast, 256KB
    ]

    for num_senders, msg_size in incast_configs:
        print(f"\n--- {num_senders}:1 Incast, 每流 {msg_size // 1024} KB ---")
        results = []
        for net in modeler.networks:
            results.append(modeler.model_incast(net, num_senders, msg_size))
        print_incast_comparison(results)

    # ========== 二分带宽利用率 ==========
    print("\n" + "=" * 90)
    print("3. 有效二分带宽利用率")
    print("=" * 90)

    all_topos = [fat_tree_32, fat_tree_128, dragonfly]
    print(f"\n{'网络技术':<22}{'拓扑':<25}{'理论二分(Tbps)':<16}"
          f"{'有效二分(Tbps)':<16}{'利用率':<10}")
    print("-" * 90)
    for topo in all_topos:
        for net in modeler.networks:
            bw = modeler.model_bisection_bandwidth(net, topo)
            print(f"{bw['network']:<22}{bw['topology']:<25}"
                  f"{bw['theoretical_bisection_gbps']/1000:<16.2f}"
                  f"{bw['effective_bisection_gbps']/1000:<16.2f}"
                  f"{bw['utilization_ratio']:<10.1%}")

    # ========== 网络技术参数总览 ==========
    print("\n" + "=" * 90)
    print("4. 网络技术参数总览")
    print("=" * 90)

    configs = [INFINIBAND_NDR, ROCE_V2, UEC_800G]
    params = [
        ("链路速率 (Gbps)", lambda c: f"{c.link_speed_gbps:.0f}"),
        ("净带宽 (Gbps)", lambda c: f"{c.net_bandwidth_gbps:.1f}"),
        ("带宽效率", lambda c: f"{c.effective_bandwidth_ratio:.0%}"),
        ("多路径效率", lambda c: f"{c.multipath_efficiency:.0%}"),
        ("拥塞控制开销", lambda c: f"{c.congestion_overhead:.0%}"),
        ("重传开销", lambda c: f"{c.retransmit_overhead:.1%}"),
        ("自适应路由", lambda c: "是" if c.adaptive_routing else "否"),
        ("最大在途数据 (KB)", lambda c: f"{c.max_outstanding_bytes // 1024}"),
        ("协议头 (字节)", lambda c: f"{c.header_overhead_bytes}"),
    ]

    print(f"\n{'参数':<20}", end="")
    for c in configs:
        print(f"{c.name:<22}", end="")
    print()
    print("-" * 86)

    for label, getter in params:
        print(f"{label:<20}", end="")
        for c in configs:
            print(f"{getter(c):<22}", end="")
        print()

    # ========== 总结 ==========
    print("\n" + "=" * 90)
    print("总结与建议")
    print("=" * 90)
    print("""
  InfiniBand NDR/NDR200:
    ✓ 成熟稳定, 自适应路由表现优秀
    ✓ 信用制流控零丢包
    ✗ 封闭生态, 单一供应商 (NVIDIA/Mellanox)
    ✗ 400Gbps 已是当代上限, 路线图不明

  RoCE v2 (400GbE):
    ✓ 开放生态, 多供应商竞争
    ✓ 与现有以太网基础设施兼容
    ✗ ECMP 哈希冲突导致带宽利用率低 (60-70%)
    ✗ Go-back-N 重传效率低
    ✗ 需要 PFC, 存在死锁风险

  UEC (800GbE) - 目标规格:
    ✓ 开放标准 + AI 优化设计
    ✓ 包喷洒实现 ~98% 多路径效率 (vs ECMP 60%)
    ✓ 多级 ECN 精确拥塞控制
    ✓ 选择性重传消除 Go-back-N 浪费
    ✓ 800GbE+ 带宽路线图清晰
    ✗ 尚未量产 (预计 2026 年)
    ✗ 需要全新 NIC + 交换芯片

  对于新建 AI 训练集群:
    - 2024-2025: InfiniBand NDR 仍是最佳选择
    - 2026+: UEC 有望成为性价比最优方案
    - 过渡期: RoCE v2 + ECN 调优可作为折中方案
""")


if __name__ == "__main__":
    main()
