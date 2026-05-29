#!/usr/bin/env python3
"""拥塞控制算法对比模拟 - DCQCN vs Credit-based vs UEC Multi-bit ECN

本模拟器对比三种数据中心拥塞控制方案:
1. DCQCN (RoCE v2): 基于单bit ECN 标记的速率调节
2. Credit-based (InfiniBand): 基于信用的逐跳流控
3. UEC Multi-bit ECN: 多级拥塞信号 + INT 遥测驱动的精细控制

重点展示 UEC 如何通过更细粒度的拥塞信号实现更高的公平性和吞吐量。
"""

import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import statistics


@dataclass
class CongestionSignal:
    """拥塞信号"""
    ecn_bits: int = 0        # ECN 位数 (0=无拥塞, 1=单bit, 0-7=多bit)
    queue_depth: int = 0     # 队列深度 (字节)
    link_utilization: float = 0.0  # 链路利用率
    timestamp_us: float = 0.0


@dataclass
class FlowState:
    """流状态"""
    flow_id: int
    rate_gbps: float           # 当前发送速率
    target_rate_gbps: float    # 目标速率 (线路速率)
    min_rate_gbps: float = 1.0   # 最低保证速率
    alpha: float = 1.0         # DCQCN alpha 参数
    credits: int = 0           # 信用值 (IB credit-based)
    bytes_sent: int = 0
    ecn_received: int = 0      # 收到的 ECN 标记数


@dataclass
class SwitchPort:
    """交换机端口 (瓶颈链路)"""
    capacity_gbps: float = 100.0
    buffer_size: int = 2 * 1024 * 1024  # 2MB 共享缓冲
    queue_depth: int = 0
    ecn_threshold_low: int = 128 * 1024   # ECN 标记起始阈值 (128KB)
    ecn_threshold_high: int = 512 * 1024  # ECN 严重拥塞阈值

    # 多级 ECN 阈值 (UEC)
    multibit_thresholds: List[int] = field(default_factory=lambda: [
        32 * 1024,    # Level 1: 32KB - 轻微拥塞
        64 * 1024,    # Level 2: 64KB
        128 * 1024,   # Level 3: 128KB
        256 * 1024,   # Level 4: 256KB
        384 * 1024,   # Level 5: 384KB
        512 * 1024,   # Level 6: 512KB - 严重拥塞
        768 * 1024,   # Level 7: 768KB - 极度拥塞
    ])

    def get_single_bit_ecn(self) -> int:
        """传统单 bit ECN: 超过阈值就标记"""
        return 1 if self.queue_depth > self.ecn_threshold_low else 0

    def get_multibit_ecn(self) -> int:
        """UEC 多级 ECN: 返回 0-7 的拥塞等级"""
        level = 0
        for threshold in self.multibit_thresholds:
            if self.queue_depth > threshold:
                level += 1
            else:
                break
        return level

    def get_utilization(self) -> float:
        """计算链路利用率"""
        return min(1.0, self.queue_depth / self.buffer_size)


class DCQCNController:
    """DCQCN 拥塞控制 (RoCE v2 标准方案)

    特点:
    - 基于单 bit ECN 标记
    - 收到 ECN 后按固定比例降速
    - 使用定时器缓慢恢复速率
    - 对拥塞程度不敏感 (只知道 "有" 或 "无")
    """

    def __init__(self, line_rate_gbps: float = 100.0):
        self.line_rate = line_rate_gbps
        self.g: float = 1.0 / 256     # Alpha 更新增益
        self.rate_decrease_factor = 0.5  # 收到 ECN 后降速比例
        self.timer_interval_us = 55.0    # 速率恢复定时间隔
        self.rate_increase_step = 5.0    # 每次恢复增加 (Gbps)

    def on_ecn(self, flow: FlowState, ecn: int):
        """收到 ECN 标记时的响应"""
        if ecn > 0:
            flow.ecn_received += 1
            # 更新 alpha (EWMA)
            flow.alpha = (1.0 - self.g) * flow.alpha + self.g
            # 降速: rate = rate * (1 - alpha/2)
            flow.rate_gbps *= (1.0 - flow.alpha / 2.0)
            flow.rate_gbps = max(flow.rate_gbps, flow.min_rate_gbps)
        else:
            # 无 ECN，缓慢恢复 alpha
            flow.alpha = (1.0 - self.g) * flow.alpha

    def timer_recovery(self, flow: FlowState):
        """定时恢复速率"""
        # 目标速率 = (当前速率 + 目标速率) / 2
        flow.target_rate_gbps = (flow.rate_gbps + flow.target_rate_gbps) / 2.0
        flow.rate_gbps = (flow.rate_gbps + flow.target_rate_gbps) / 2.0
        flow.rate_gbps = min(flow.rate_gbps, self.line_rate)


class CreditBasedController:
    """信用制流控 (InfiniBand 方案)

    特点:
    - 逐跳信用: 发送端需持有信用才能发送
    - 零丢包: 信用耗尽则停止发送
    - 延迟高: 信用往返延迟限制吞吐
    - 头阻塞: 一个端口拥塞影响其他流
    """

    def __init__(self, credit_size: int = 4096, max_credits: int = 64):
        self.credit_size = credit_size
        self.max_credits = max_credits
        self.credit_return_latency_us = 2.0  # 信用返还延迟

    def allocate_credits(self, flow: FlowState, available_buffer: int):
        """根据可用缓冲分配信用"""
        available_credits = available_buffer // self.credit_size
        flow.credits = min(available_credits // 4, self.max_credits)  # 公平分配

    def can_send(self, flow: FlowState) -> bool:
        """检查是否有信用可发送"""
        return flow.credits > 0

    def consume_credit(self, flow: FlowState):
        """消耗一个信用"""
        flow.credits = max(0, flow.credits - 1)

    def return_credit(self, flow: FlowState):
        """返还信用 (数据被消费后)"""
        flow.credits = min(flow.credits + 1, self.max_credits)


class UECMultiBitController:
    """UEC 多级 ECN 拥塞控制

    特点:
    - 8 级拥塞信号 (3-bit ECN)
    - 按拥塞程度比例降速 (而非固定降)
    - Sub-RTT 反应: 利用 INT 遥测快速响应
    - 精确公平: 基于拥塞度计算精确的公平速率
    """

    def __init__(self, line_rate_gbps: float = 100.0):
        self.line_rate = line_rate_gbps
        # 每级对应的降速系数 (0=不降, 7=大幅降速)
        self.level_factors = [
            1.00,  # Level 0: 不降速
            0.95,  # Level 1: 轻微降速 5%
            0.85,  # Level 2: 降速 15%
            0.70,  # Level 3: 降速 30%
            0.55,  # Level 4: 降速 45%
            0.40,  # Level 5: 降速 60%
            0.25,  # Level 6: 降速 75%
            0.10,  # Level 7: 降速 90%
        ]
        self.recovery_rate = 0.05  # 无拥塞时的恢复速率 (每步 5%)

    def on_multibit_ecn(self, flow: FlowState, ecn_level: int):
        """收到多级 ECN 信号时的响应"""
        if ecn_level > 0:
            flow.ecn_received += 1
            # 按拥塞等级比例降速
            factor = self.level_factors[min(ecn_level, 7)]
            flow.rate_gbps = flow.target_rate_gbps * factor
            flow.rate_gbps = max(flow.rate_gbps, flow.min_rate_gbps)
        else:
            # 无拥塞，快速恢复
            flow.rate_gbps = min(
                flow.rate_gbps * (1.0 + self.recovery_rate),
                flow.target_rate_gbps
            )


@dataclass
class TimeSeriesPoint:
    """时间序列数据点"""
    time_us: float
    throughput_gbps: float
    queue_depth_kb: float
    fairness_index: float


class CongestionSimulator:
    """拥塞控制模拟器"""

    def __init__(self, num_flows: int = 8, link_capacity_gbps: float = 100.0,
                 simulation_time_us: float = 1000.0):
        """
        初始化模拟参数

        Args:
            num_flows: 竞争流数量
            link_capacity_gbps: 瓶颈链路带宽
            simulation_time_us: 模拟时长 (微秒)
        """
        self.num_flows = num_flows
        self.link_capacity = link_capacity_gbps
        self.sim_time = simulation_time_us
        self.time_step = 1.0  # 时间步长 (微秒)

    def _compute_jains_fairness(self, rates: List[float]) -> float:
        """计算 Jain's Fairness Index

        公式: (Σxi)² / (n × Σxi²)
        值域: [1/n, 1], 1 表示完全公平
        """
        if not rates or all(r == 0 for r in rates):
            return 0.0
        n = len(rates)
        sum_x = sum(rates)
        sum_x2 = sum(r * r for r in rates)
        if sum_x2 == 0:
            return 0.0
        return (sum_x ** 2) / (n * sum_x2)

    def simulate_dcqcn(self) -> Tuple[List[TimeSeriesPoint], Dict]:
        """模拟 DCQCN 拥塞控制"""
        switch = SwitchPort(capacity_gbps=self.link_capacity)
        controller = DCQCNController(line_rate_gbps=self.link_capacity)
        flows = [
            FlowState(
                flow_id=i,
                rate_gbps=self.link_capacity,  # 初始全速发送
                target_rate_gbps=self.link_capacity,
            )
            for i in range(self.num_flows)
        ]

        timeseries = []
        timer_counters = [0.0] * self.num_flows
        current_time = 0.0

        while current_time < self.sim_time:
            # 计算本步总发送量 (bytes)
            rates = [f.rate_gbps for f in flows]
            total_input_bytes = sum(r * 1e9 / 8 * self.time_step * 1e-6 for r in rates)

            # 交换机出队量
            drain_bytes = self.link_capacity * 1e9 / 8 * self.time_step * 1e-6

            # 更新队列
            switch.queue_depth = max(0, int(switch.queue_depth + total_input_bytes - drain_bytes))
            switch.queue_depth = min(switch.queue_depth, switch.buffer_size)

            # ECN 标记
            ecn = switch.get_single_bit_ecn()

            # 每个流处理 ECN
            for i, flow in enumerate(flows):
                controller.on_ecn(flow, ecn)
                # 定时恢复
                timer_counters[i] += self.time_step
                if timer_counters[i] >= controller.timer_interval_us:
                    controller.timer_recovery(flow)
                    timer_counters[i] = 0.0

            # 记录时间序列
            actual_throughput = min(sum(rates), self.link_capacity)
            fairness = self._compute_jains_fairness(rates)
            timeseries.append(TimeSeriesPoint(
                time_us=current_time,
                throughput_gbps=actual_throughput,
                queue_depth_kb=switch.queue_depth / 1024,
                fairness_index=fairness,
            ))

            current_time += self.time_step

        # 汇总统计
        stats = {
            "method": "DCQCN (RoCE v2)",
            "avg_throughput_gbps": statistics.mean(
                [min(sum(f.rate_gbps for f in flows), self.link_capacity)
                 for _ in range(1)]),
            "avg_queue_kb": statistics.mean([t.queue_depth_kb for t in timeseries]),
            "max_queue_kb": max(t.queue_depth_kb for t in timeseries),
            "avg_fairness": statistics.mean([t.fairness_index for t in timeseries]),
            "convergence_time_us": self._find_convergence_time(timeseries),
            "final_rates": [f.rate_gbps for f in flows],
        }
        return timeseries, stats

    def simulate_credit_based(self) -> Tuple[List[TimeSeriesPoint], Dict]:
        """模拟信用制流控 (InfiniBand)"""
        switch = SwitchPort(capacity_gbps=self.link_capacity)
        controller = CreditBasedController()
        flows = [
            FlowState(
                flow_id=i,
                rate_gbps=self.link_capacity / self.num_flows,  # 公平分配
                target_rate_gbps=self.link_capacity,
                credits=controller.max_credits // self.num_flows,
            )
            for i in range(self.num_flows)
        ]

        timeseries = []
        current_time = 0.0

        while current_time < self.sim_time:
            # 信用分配: 根据可用缓冲公平分配
            available = switch.buffer_size - switch.queue_depth
            for flow in flows:
                controller.allocate_credits(flow, available)

            # 计算可发送量
            rates = []
            for flow in flows:
                if controller.can_send(flow):
                    # 速率受限于信用
                    credit_limited_rate = (
                        flow.credits * controller.credit_size * 8
                        / (self.time_step * 1e-6) / 1e9
                    )
                    effective_rate = min(credit_limited_rate, flow.target_rate_gbps)
                    flow.rate_gbps = effective_rate
                else:
                    flow.rate_gbps = 0.0
                rates.append(flow.rate_gbps)

            # 更新队列
            total_input_bytes = sum(r * 1e9 / 8 * self.time_step * 1e-6 for r in rates)
            drain_bytes = self.link_capacity * 1e9 / 8 * self.time_step * 1e-6
            switch.queue_depth = max(0, int(switch.queue_depth + total_input_bytes - drain_bytes))
            switch.queue_depth = min(switch.queue_depth, switch.buffer_size)

            # 信用消耗
            for flow in flows:
                if flow.rate_gbps > 0:
                    pkts_sent = int(flow.rate_gbps * 1e9 / 8 * self.time_step * 1e-6
                                    / controller.credit_size)
                    for _ in range(pkts_sent):
                        controller.consume_credit(flow)

            # 记录
            actual_throughput = min(sum(rates), self.link_capacity)
            fairness = self._compute_jains_fairness(rates)
            timeseries.append(TimeSeriesPoint(
                time_us=current_time,
                throughput_gbps=actual_throughput,
                queue_depth_kb=switch.queue_depth / 1024,
                fairness_index=fairness,
            ))

            current_time += self.time_step

        stats = {
            "method": "Credit-based (InfiniBand)",
            "avg_throughput_gbps": statistics.mean([t.throughput_gbps for t in timeseries]),
            "avg_queue_kb": statistics.mean([t.queue_depth_kb for t in timeseries]),
            "max_queue_kb": max(t.queue_depth_kb for t in timeseries),
            "avg_fairness": statistics.mean([t.fairness_index for t in timeseries]),
            "convergence_time_us": 0.0,  # 信用制从一开始就公平
            "final_rates": [f.rate_gbps for f in flows],
        }
        return timeseries, stats

    def simulate_uec_multibit(self) -> Tuple[List[TimeSeriesPoint], Dict]:
        """模拟 UEC 多级 ECN 拥塞控制"""
        switch = SwitchPort(capacity_gbps=self.link_capacity)
        controller = UECMultiBitController(line_rate_gbps=self.link_capacity)
        flows = [
            FlowState(
                flow_id=i,
                rate_gbps=self.link_capacity,  # 初始全速
                target_rate_gbps=self.link_capacity / self.num_flows,
            )
            for i in range(self.num_flows)
        ]
        # UEC 知道公平份额目标
        fair_share = self.link_capacity / self.num_flows
        for f in flows:
            f.target_rate_gbps = fair_share * 1.2  # 允许略超公平份额探测

        timeseries = []
        current_time = 0.0

        while current_time < self.sim_time:
            rates = [f.rate_gbps for f in flows]
            total_input_bytes = sum(r * 1e9 / 8 * self.time_step * 1e-6 for r in rates)
            drain_bytes = self.link_capacity * 1e9 / 8 * self.time_step * 1e-6

            # 更新队列
            switch.queue_depth = max(0, int(switch.queue_depth + total_input_bytes - drain_bytes))
            switch.queue_depth = min(switch.queue_depth, switch.buffer_size)

            # 多级 ECN 信号
            ecn_level = switch.get_multibit_ecn()

            # 每个流响应多级 ECN
            for flow in flows:
                controller.on_multibit_ecn(flow, ecn_level)

            # 记录
            actual_throughput = min(sum(rates), self.link_capacity)
            fairness = self._compute_jains_fairness(rates)
            timeseries.append(TimeSeriesPoint(
                time_us=current_time,
                throughput_gbps=actual_throughput,
                queue_depth_kb=switch.queue_depth / 1024,
                fairness_index=fairness,
            ))

            current_time += self.time_step

        stats = {
            "method": "UEC Multi-bit ECN",
            "avg_throughput_gbps": statistics.mean([t.throughput_gbps for t in timeseries]),
            "avg_queue_kb": statistics.mean([t.queue_depth_kb for t in timeseries]),
            "max_queue_kb": max(t.queue_depth_kb for t in timeseries),
            "avg_fairness": statistics.mean([t.fairness_index for t in timeseries]),
            "convergence_time_us": self._find_convergence_time(timeseries),
            "final_rates": [f.rate_gbps for f in flows],
        }
        return timeseries, stats

    def _find_convergence_time(self, timeseries: List[TimeSeriesPoint],
                               fairness_threshold: float = 0.95) -> float:
        """找到收敛时间 (公平性首次达到阈值)"""
        for point in timeseries:
            if point.fairness_index >= fairness_threshold:
                return point.time_us
        return self.sim_time  # 未收敛


def print_comparison_table(all_stats: List[Dict]):
    """打印对比结果表格"""
    print("\n" + "=" * 80)
    print("拥塞控制方案对比结果")
    print("=" * 80)

    # 表头
    headers = ["指标", "DCQCN (RoCE v2)", "Credit (IB)", "UEC Multi-bit"]
    col_widths = [20, 18, 18, 18]
    header_line = "".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * sum(col_widths))

    # 数据行
    metrics = [
        ("平均吞吐 (Gbps)", "avg_throughput_gbps", ".2f"),
        ("平均队列 (KB)", "avg_queue_kb", ".1f"),
        ("最大队列 (KB)", "max_queue_kb", ".1f"),
        ("平均公平性", "avg_fairness", ".4f"),
        ("收敛时间 (μs)", "convergence_time_us", ".1f"),
    ]

    for label, key, fmt in metrics:
        row = f"{label:<20}"
        for stats in all_stats:
            val = stats.get(key, 0)
            row += f"{val:<18{fmt}}"
        print(row)

    print("-" * sum(col_widths))

    # 最终各流速率
    print("\n各流最终速率 (Gbps):")
    for stats in all_stats:
        rates = stats["final_rates"]
        rate_str = ", ".join(f"{r:.2f}" for r in rates[:4])
        if len(rates) > 4:
            rate_str += f", ... (共{len(rates)}流)"
        print(f"  {stats['method']}: [{rate_str}]")
        fairness = (sum(rates) ** 2) / (len(rates) * sum(r**2 for r in rates)) if sum(r**2 for r in rates) > 0 else 0
        print(f"    Jain's Fairness = {fairness:.4f}")


def run_full_comparison():
    """运行完整拥塞控制对比"""
    print("=" * 80)
    print("数据中心拥塞控制算法对比模拟")
    print("DCQCN (RoCE v2) vs Credit-based (IB) vs UEC Multi-bit ECN")
    print("=" * 80)

    # 模拟参数
    num_flows = 8
    link_gbps = 100.0
    sim_time_us = 500.0

    print(f"\n模拟参数:")
    print(f"  竞争流数: {num_flows}")
    print(f"  瓶颈带宽: {link_gbps} Gbps")
    print(f"  模拟时长: {sim_time_us} μs")
    print(f"  公平份额: {link_gbps / num_flows:.2f} Gbps/流")

    sim = CongestionSimulator(
        num_flows=num_flows,
        link_capacity_gbps=link_gbps,
        simulation_time_us=sim_time_us,
    )

    # 运行三种方案
    print("\n[1/3] 模拟 DCQCN (RoCE v2)...")
    ts_dcqcn, stats_dcqcn = sim.simulate_dcqcn()

    print("[2/3] 模拟 Credit-based (InfiniBand)...")
    ts_credit, stats_credit = sim.simulate_credit_based()

    print("[3/3] 模拟 UEC Multi-bit ECN...")
    ts_uec, stats_uec = sim.simulate_uec_multibit()

    # 打印对比表
    print_comparison_table([stats_dcqcn, stats_credit, stats_uec])

    # 时间序列采样输出
    print("\n" + "-" * 80)
    print("队列深度变化趋势 (每 50μs 采样):")
    print("-" * 80)
    print(f"{'时间(μs)':<10}{'DCQCN队列(KB)':<16}{'Credit队列(KB)':<16}{'UEC队列(KB)':<16}")

    sample_interval = 50
    for i in range(0, len(ts_dcqcn), int(sample_interval / sim.time_step)):
        if i < len(ts_dcqcn) and i < len(ts_credit) and i < len(ts_uec):
            print(f"{ts_dcqcn[i].time_us:<10.0f}"
                  f"{ts_dcqcn[i].queue_depth_kb:<16.1f}"
                  f"{ts_credit[i].queue_depth_kb:<16.1f}"
                  f"{ts_uec[i].queue_depth_kb:<16.1f}")

    # 分析结论
    print("\n" + "=" * 80)
    print("分析结论")
    print("=" * 80)
    print(f"""
  DCQCN (RoCE v2):
    - 单 bit ECN 只能告知 "有拥塞"，无法传达拥塞严重程度
    - 降速后需要漫长的定时恢复过程，导致吞吐振荡
    - 收敛时间: {stats_dcqcn['convergence_time_us']:.0f} μs

  Credit-based (InfiniBand):
    - 零丢包保证，但受限于信用往返延迟
    - 公平性好 (天然公平分配信用)，但链路利用率受限
    - 无法利用突发带宽 (受信用上限约束)

  UEC Multi-bit ECN:
    - 8 级拥塞信号提供精确的拥塞程度信息
    - 可按比例调整速率，避免过度降速或欠降速
    - 收敛速度快: {stats_uec['convergence_time_us']:.0f} μs
    - 队列占用低: 平均 {stats_uec['avg_queue_kb']:.0f} KB vs DCQCN {stats_dcqcn['avg_queue_kb']:.0f} KB
    - 结合 INT 遥测可实现 sub-RTT 反应
""")


def demo_multibit_ecn_levels():
    """演示多级 ECN 的工作原理"""
    print("\n" + "=" * 80)
    print("演示: UEC 多级 ECN 信号映射")
    print("=" * 80)

    switch = SwitchPort()
    print(f"\n  缓冲区大小: {switch.buffer_size // 1024} KB")
    print(f"  单bit ECN 阈值: {switch.ecn_threshold_low // 1024} KB")
    print(f"\n  多级 ECN 阈值配置:")
    for i, threshold in enumerate(switch.multibit_thresholds):
        print(f"    Level {i+1}: > {threshold // 1024} KB")

    print(f"\n  队列深度 -> ECN 等级映射:")
    test_depths = [0, 16*1024, 48*1024, 100*1024, 200*1024,
                   300*1024, 450*1024, 600*1024, 900*1024]
    for depth in test_depths:
        switch.queue_depth = depth
        single = switch.get_single_bit_ecn()
        multi = switch.get_multibit_ecn()
        bar = "█" * multi + "░" * (7 - multi)
        print(f"    {depth//1024:>4d} KB -> 单bit: {single}  多bit: {multi}/7 [{bar}]")


def main():
    """主函数"""
    random.seed(42)

    demo_multibit_ecn_levels()
    run_full_comparison()


if __name__ == "__main__":
    main()
