#!/usr/bin/env python3
"""UEC Transport Protocol 模拟器 - 模拟包喷洒(Packet Spraying)与ECMP路由对比

本模拟器演示 UEC 传输协议的核心创新：
1. Packet Spraying: 将数据包分散到所有可用路径，实现接近完美的链路利用率
2. 乱序交付: 接收端重排序缓冲区处理乱序到达的数据包
3. 与传统 ECMP 哈希路由的性能对比
"""

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import statistics


@dataclass
class Packet:
    """数据包"""
    flow_id: int          # 所属流 ID
    seq_num: int          # 序列号
    size: int             # 字节大小
    send_time: float      # 发送时间
    arrive_time: float = 0.0    # 到达时间
    path_id: int = -1     # 经过的路径 ID


@dataclass
class Path:
    """网络路径"""
    path_id: int
    capacity_gbps: float       # 链路容量 (Gbps)
    base_latency_us: float     # 基础延迟 (微秒)
    current_load: float = 0.0  # 当前负载 (0.0 - 1.0)
    queue_depth: int = 0       # 队列深度 (字节)
    max_queue: int = 256 * 1024  # 最大队列 (256KB)
    packets_sent: int = 0      # 已发送包数

    def get_latency(self) -> float:
        """计算当前延迟 (含排队延迟)"""
        queue_latency = (self.queue_depth / (self.capacity_gbps * 1e9 / 8)) * 1e6
        return self.base_latency_us + queue_latency

    def enqueue(self, packet_size: int) -> bool:
        """数据包入队，返回是否成功"""
        if self.queue_depth + packet_size > self.max_queue:
            return False  # 队列满，丢包
        self.queue_depth += packet_size
        self.packets_sent += 1
        return True

    def dequeue(self, packet_size: int):
        """数据包出队"""
        self.queue_depth = max(0, self.queue_depth - packet_size)
        self.current_load = self.queue_depth / self.max_queue


@dataclass
class Flow:
    """网络流"""
    flow_id: int
    total_packets: int
    packet_size: int = 4096   # 默认 4KB MTU
    ecmp_hash: int = 0        # ECMP 哈希值 (决定固定路径)


@dataclass
class ReorderBuffer:
    """接收端重排序缓冲区 - UEC 核心组件"""
    expected_seq: int = 0
    buffer: Dict[int, Packet] = field(default_factory=dict)
    max_buffer_size: int = 1024
    delivered_packets: List[Packet] = field(default_factory=list)
    reorder_events: int = 0

    def receive(self, packet: Packet) -> List[Packet]:
        """接收数据包，返回可交付的有序包列表"""
        delivered = []

        if packet.seq_num == self.expected_seq:
            # 按序到达
            delivered.append(packet)
            self.expected_seq += 1
            # 检查缓冲区中是否有后续包
            while self.expected_seq in self.buffer:
                delivered.append(self.buffer.pop(self.expected_seq))
                self.expected_seq += 1
        elif packet.seq_num > self.expected_seq:
            # 乱序到达，存入缓冲区
            self.buffer[packet.seq_num] = packet
            self.reorder_events += 1
        # seq_num < expected_seq: 重复包，丢弃

        self.delivered_packets.extend(delivered)
        return delivered


class ECMPRouter:
    """传统 ECMP 路由器 - 基于流哈希的路径选择"""

    def __init__(self, paths: List[Path]):
        self.paths = paths
        self.flow_table: Dict[int, int] = {}  # flow_id -> path_id

    def route(self, packet: Packet) -> Path:
        """根据流哈希选择路径 (同一流始终走同一路径)"""
        if packet.flow_id not in self.flow_table:
            # 五元组哈希 -> 路径选择
            hash_val = hash(packet.flow_id) % len(self.paths)
            self.flow_table[packet.flow_id] = hash_val
        path_idx = self.flow_table[packet.flow_id]
        return self.paths[path_idx]


class UECPacketSprayer:
    """UEC 包喷洒路由 - 逐包自适应路径选择"""

    def __init__(self, paths: List[Path]):
        self.paths = paths
        self.rr_counter: int = 0  # Round-robin 计数器

    def route(self, packet: Packet, strategy: str = "least_loaded") -> Path:
        """逐包选择最优路径

        策略:
        - least_loaded: 选择队列最短的路径
        - round_robin: 轮询所有路径
        - weighted: 根据剩余容量加权选择
        """
        if strategy == "least_loaded":
            return min(self.paths, key=lambda p: p.queue_depth)
        elif strategy == "round_robin":
            path = self.paths[self.rr_counter % len(self.paths)]
            self.rr_counter += 1
            return path
        elif strategy == "weighted":
            # 按剩余容量加权随机选择
            weights = [max(0.01, 1.0 - p.current_load) for p in self.paths]
            total = sum(weights)
            probs = [w / total for w in weights]
            return random.choices(self.paths, weights=probs, k=1)[0]
        else:
            return self.paths[0]


@dataclass
class SimulationResult:
    """模拟结果"""
    method: str
    total_packets: int
    total_time_us: float
    avg_latency_us: float
    p99_latency_us: float
    path_utilization: List[float]  # 各路径利用率
    utilization_std: float         # 利用率标准差 (越低越均衡)
    dropped_packets: int
    reorder_events: int
    effective_throughput_gbps: float


class UECTransportSimulator:
    """UEC 传输协议模拟器"""

    def __init__(self, num_paths: int = 8, path_capacity_gbps: float = 100.0,
                 base_latency_us: float = 5.0):
        """
        初始化模拟环境

        Args:
            num_paths: 可用路径数 (Fat-tree 拓扑中的等价路径)
            path_capacity_gbps: 每条路径容量
            base_latency_us: 基础传播延迟
        """
        self.num_paths = num_paths
        self.path_capacity_gbps = path_capacity_gbps
        self.base_latency_us = base_latency_us

    def _create_paths(self) -> List[Path]:
        """创建路径集合 (轻微不对称模拟真实网络)"""
        paths = []
        for i in range(self.num_paths):
            # 添加轻微的延迟差异模拟真实网络不对称性
            latency_jitter = random.uniform(-0.5, 0.5)
            paths.append(Path(
                path_id=i,
                capacity_gbps=self.path_capacity_gbps,
                base_latency_us=self.base_latency_us + latency_jitter,
            ))
        return paths

    def _create_flows(self, num_flows: int, packets_per_flow: int) -> List[Flow]:
        """创建流量模型"""
        flows = []
        for i in range(num_flows):
            flows.append(Flow(
                flow_id=i,
                total_packets=packets_per_flow,
                ecmp_hash=random.randint(0, 2**32 - 1),
            ))
        return flows

    def _simulate_draining(self, paths: List[Path], drain_per_step: int = 2048):
        """模拟链路排空 (每个时间步排出一定字节)"""
        for path in paths:
            path.dequeue(drain_per_step)

    def run_ecmp_simulation(self, flows: List[Flow]) -> SimulationResult:
        """运行 ECMP 路由模拟"""
        paths = self._create_paths()
        router = ECMPRouter(paths)
        all_latencies = []
        dropped = 0
        current_time = 0.0
        step_interval = 0.1  # 微秒

        for flow in flows:
            for seq in range(flow.total_packets):
                pkt = Packet(
                    flow_id=flow.flow_id,
                    seq_num=seq,
                    size=flow.packet_size,
                    send_time=current_time,
                )
                path = router.route(pkt)
                if path.enqueue(pkt.size):
                    latency = path.get_latency()
                    pkt.arrive_time = current_time + latency
                    pkt.path_id = path.path_id
                    all_latencies.append(latency)
                else:
                    dropped += 1

                current_time += step_interval
                # 周期性排空
                if int(current_time * 10) % 5 == 0:
                    self._simulate_draining(paths)

        # 计算结果
        path_packets = [p.packets_sent for p in paths]
        total_pkts = sum(path_packets) if sum(path_packets) > 0 else 1
        utilization = [p.packets_sent / (total_pkts / self.num_paths)
                       if total_pkts > 0 else 0.0 for p in paths]
        # 归一化利用率
        max_util = max(utilization) if utilization else 1.0
        utilization = [u / max_util for u in utilization]

        avg_lat = statistics.mean(all_latencies) if all_latencies else 0.0
        p99_lat = sorted(all_latencies)[int(len(all_latencies) * 0.99)] if all_latencies else 0.0

        total_bytes = sum(f.total_packets * f.packet_size for f in flows) - dropped * 4096
        total_time = current_time
        throughput = (total_bytes * 8 / (total_time * 1e-6)) / 1e9 if total_time > 0 else 0.0

        return SimulationResult(
            method="ECMP",
            total_packets=sum(f.total_packets for f in flows),
            total_time_us=total_time,
            avg_latency_us=avg_lat,
            p99_latency_us=p99_lat,
            path_utilization=utilization,
            utilization_std=statistics.stdev(utilization) if len(utilization) > 1 else 0.0,
            dropped_packets=dropped,
            reorder_events=0,  # ECMP 保证有序
            effective_throughput_gbps=throughput,
        )

    def run_spray_simulation(self, flows: List[Flow],
                             strategy: str = "least_loaded") -> SimulationResult:
        """运行 UEC Packet Spray 模拟"""
        paths = self._create_paths()
        sprayer = UECPacketSprayer(paths)
        reorder_buffers: Dict[int, ReorderBuffer] = defaultdict(ReorderBuffer)
        all_latencies = []
        dropped = 0
        current_time = 0.0
        step_interval = 0.1

        for flow in flows:
            for seq in range(flow.total_packets):
                pkt = Packet(
                    flow_id=flow.flow_id,
                    seq_num=seq,
                    size=flow.packet_size,
                    send_time=current_time,
                )
                path = sprayer.route(pkt, strategy=strategy)
                if path.enqueue(pkt.size):
                    latency = path.get_latency()
                    pkt.arrive_time = current_time + latency
                    pkt.path_id = path.path_id
                    all_latencies.append(latency)
                    # 模拟接收端重排序
                    reorder_buffers[flow.flow_id].receive(pkt)
                else:
                    dropped += 1

                current_time += step_interval
                if int(current_time * 10) % 5 == 0:
                    self._simulate_draining(paths)

        # 计算路径利用率
        path_packets = [p.packets_sent for p in paths]
        total_pkts = sum(path_packets) if sum(path_packets) > 0 else 1
        utilization = [p.packets_sent / (total_pkts / self.num_paths)
                       if total_pkts > 0 else 0.0 for p in paths]
        max_util = max(utilization) if utilization else 1.0
        utilization = [u / max_util for u in utilization]

        # 统计重排序事件
        total_reorder = sum(rb.reorder_events for rb in reorder_buffers.values())

        avg_lat = statistics.mean(all_latencies) if all_latencies else 0.0
        p99_lat = sorted(all_latencies)[int(len(all_latencies) * 0.99)] if all_latencies else 0.0

        total_bytes = sum(f.total_packets * f.packet_size for f in flows) - dropped * 4096
        total_time = current_time
        throughput = (total_bytes * 8 / (total_time * 1e-6)) / 1e9 if total_time > 0 else 0.0

        return SimulationResult(
            method=f"UEC Spray ({strategy})",
            total_packets=sum(f.total_packets for f in flows),
            total_time_us=total_time,
            avg_latency_us=avg_lat,
            p99_latency_us=p99_lat,
            path_utilization=utilization,
            utilization_std=statistics.stdev(utilization) if len(utilization) > 1 else 0.0,
            dropped_packets=dropped,
            reorder_events=total_reorder,
            effective_throughput_gbps=throughput,
        )

    def print_result(self, result: SimulationResult):
        """打印单个模拟结果"""
        print(f"\n  方法: {result.method}")
        print(f"  总数据包: {result.total_packets}")
        print(f"  平均延迟: {result.avg_latency_us:.2f} μs")
        print(f"  P99 延迟: {result.p99_latency_us:.2f} μs")
        print(f"  丢包数: {result.dropped_packets}")
        print(f"  重排序事件: {result.reorder_events}")
        print(f"  有效吞吐量: {result.effective_throughput_gbps:.2f} Gbps")
        print(f"  路径利用率标准差: {result.utilization_std:.4f} (越低越均衡)")
        print(f"  各路径利用率: {[f'{u:.2f}' for u in result.path_utilization]}")


def run_comparison():
    """运行完整的 ECMP vs UEC Packet Spray 对比"""
    print("=" * 70)
    print("UEC Transport Protocol 模拟: Packet Spraying vs ECMP")
    print("=" * 70)

    # 场景 1: 少量大流 (ECMP 哈希冲突明显)
    print("\n" + "-" * 70)
    print("场景 1: 少量大流 (4 flows × 500 packets, 8 等价路径)")
    print("  ECMP 中流量集中在少数路径上 (哈希冲突)")
    print("-" * 70)

    sim = UECTransportSimulator(num_paths=8, path_capacity_gbps=100.0)
    flows = sim._create_flows(num_flows=4, packets_per_flow=500)

    ecmp_result = sim.run_ecmp_simulation(flows)
    spray_result = sim.run_spray_simulation(flows, strategy="least_loaded")

    sim.print_result(ecmp_result)
    sim.print_result(spray_result)

    improvement = ((spray_result.effective_throughput_gbps - ecmp_result.effective_throughput_gbps)
                   / ecmp_result.effective_throughput_gbps * 100
                   if ecmp_result.effective_throughput_gbps > 0 else 0)
    print(f"\n  >>> Spray 吞吐提升: {improvement:.1f}%")
    print(f"  >>> 负载均衡改善: "
          f"{ecmp_result.utilization_std:.4f} -> {spray_result.utilization_std:.4f}")

    # 场景 2: 大量小流 (ECMP 表现尚可)
    print("\n" + "-" * 70)
    print("场景 2: 大量小流 (64 flows × 50 packets, 8 等价路径)")
    print("  ECMP 在大量流时统计均衡较好，但仍不如 Spray")
    print("-" * 70)

    flows2 = sim._create_flows(num_flows=64, packets_per_flow=50)
    ecmp_result2 = sim.run_ecmp_simulation(flows2)
    spray_result2 = sim.run_spray_simulation(flows2, strategy="least_loaded")

    sim.print_result(ecmp_result2)
    sim.print_result(spray_result2)

    # 场景 3: Incast (多对一通信模式, AI训练典型场景)
    print("\n" + "-" * 70)
    print("场景 3: Incast 模式 (32 flows × 100 packets 同时发往同一目标)")
    print("  模拟 AllReduce 中的 reduce 阶段")
    print("-" * 70)

    sim_incast = UECTransportSimulator(num_paths=8, path_capacity_gbps=100.0)
    flows3 = sim_incast._create_flows(num_flows=32, packets_per_flow=100)
    ecmp_result3 = sim_incast.run_ecmp_simulation(flows3)
    spray_result3 = sim_incast.run_spray_simulation(flows3, strategy="least_loaded")

    sim_incast.print_result(ecmp_result3)
    sim_incast.print_result(spray_result3)

    # 场景 4: 不同 Spray 策略对比
    print("\n" + "-" * 70)
    print("场景 4: 不同 Packet Spray 策略对比")
    print("-" * 70)

    sim4 = UECTransportSimulator(num_paths=8)
    flows4 = sim4._create_flows(num_flows=16, packets_per_flow=200)

    strategies = ["least_loaded", "round_robin", "weighted"]
    for strategy in strategies:
        result = sim4.run_spray_simulation(flows4, strategy=strategy)
        sim4.print_result(result)

    # 总结
    print("\n" + "=" * 70)
    print("总结: UEC Packet Spraying 的优势")
    print("=" * 70)
    print("""
  1. 负载均衡: Spray 将流量均匀分散到所有路径，消除 ECMP 哈希冲突
  2. 链路利用率: 接近 100% 的等分带宽利用率 (ECMP 通常仅 60-70%)
  3. 尾延迟: 消除热点路径的排队，降低 P99 延迟
  4. Incast 容忍: 在多对一通信中表现尤为突出

  代价:
  - 需要接收端重排序缓冲区 (NIC 硬件实现)
  - 增加少量内存开销
  - 需要精确的路径状态信息
""")


def demo_reorder_buffer():
    """演示接收端重排序缓冲区工作原理"""
    print("\n" + "=" * 70)
    print("演示: UEC 接收端重排序缓冲区")
    print("=" * 70)

    rb = ReorderBuffer()
    # 模拟乱序到达的数据包
    arrival_order = [0, 2, 1, 4, 3, 5, 8, 6, 7, 9]
    print(f"\n  数据包到达顺序: {arrival_order}")
    print(f"  (期望顺序: 0, 1, 2, 3, ...)\n")

    for seq in arrival_order:
        pkt = Packet(flow_id=0, seq_num=seq, size=4096, send_time=0.0)
        delivered = rb.receive(pkt)
        if delivered:
            delivered_seqs = [p.seq_num for p in delivered]
            print(f"  收到 seq={seq:2d} -> 交付: {delivered_seqs}")
        else:
            print(f"  收到 seq={seq:2d} -> 缓冲 (等待 seq={rb.expected_seq})")

    print(f"\n  总重排序事件: {rb.reorder_events}")
    print(f"  已交付包数: {len(rb.delivered_packets)}")
    print(f"  缓冲区残留: {len(rb.buffer)} 个包")


def main():
    """主函数"""
    random.seed(42)  # 固定种子以便复现

    demo_reorder_buffer()
    run_comparison()


if __name__ == "__main__":
    main()
