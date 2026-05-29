"""
PFC死锁检测模块

监控Priority Flow Control (PFC)帧计数器，检测RoCE部署中的
潜在死锁模式。检查PFC风暴指标，分析PFC暂停帧的发送和接收
模式，识别可能导致网络死锁的异常状态。
"""

import logging
import subprocess
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime

logger = logging.getLogger(__name__)


class PFCStatus(Enum):
    """PFC状态"""
    NORMAL = "normal"
    ELEVATED = "elevated"       # PFC暂停帧偏多
    STORM = "storm"             # PFC风暴
    DEADLOCK_RISK = "deadlock_risk"  # 存在死锁风险
    DEADLOCK = "deadlock"       # 检测到死锁


class PFCPriority(Enum):
    """PFC优先级（IEEE 802.1Qbb）"""
    TC0 = 0
    TC1 = 1
    TC2 = 2
    TC3 = 3  # 通常用于RoCE
    TC4 = 4
    TC5 = 5
    TC6 = 6
    TC7 = 7


@dataclass
class PFCCounters:
    """PFC计数器"""
    node: str
    device: str
    port: int
    timestamp: str = ""
    # 每个优先级的PFC暂停帧发送计数
    tx_pause: Dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(8)})
    # 每个优先级的PFC暂停帧接收计数
    rx_pause: Dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(8)})
    # PFC暂停持续时间（纳秒）
    tx_pause_duration: Dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(8)})
    rx_pause_duration: Dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(8)})


@dataclass
class PFCSnapshot:
    """PFC状态快照（某一时刻的完整集群PFC状态）"""
    timestamp: str
    counters: Dict[str, PFCCounters] = field(default_factory=dict)  # key: node:device:port


@dataclass
class DeadlockPattern:
    """检测到的死锁模式"""
    pattern_type: str  # 模式类型描述
    involved_nodes: List[str]  # 涉及的节点
    involved_links: List[str]  # 涉及的链路
    priority: int  # 涉及的优先级
    description: str = ""
    severity: float = 0.0  # 0.0-1.0


@dataclass
class PFCAnalysisResult:
    """PFC分析结果"""
    check_time: str
    overall_status: PFCStatus
    node_status: Dict[str, PFCStatus] = field(default_factory=dict)
    hotspots: List[str] = field(default_factory=list)  # PFC暂停帧热点
    deadlock_patterns: List[DeadlockPattern] = field(default_factory=list)
    storm_alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    statistics: Dict[str, float] = field(default_factory=dict)


class PFCDeadlockDetector:
    """
    PFC死锁检测器

    监控RoCE网络中的PFC帧行为，检测以下异常:
    1. PFC风暴：某个端口持续发送大量PFC暂停帧
    2. 循环依赖：多个端口形成PFC暂停帧的循环链
    3. 头部阻塞(HOL Blocking)：PFC暂停导致的队列堆积
    4. 死锁模式：多节点间的PFC互锁

    检测原理:
    - 采集两次PFC计数器快照，计算增量
    - 分析PFC暂停帧的发送/接收比率
    - 检测是否存在循环依赖的PFC暂停链
    - 评估暂停持续时间是否超出阈值
    """

    def __init__(self, config: dict):
        """
        初始化PFC死锁检测器

        Args:
            config: 配置字典
        """
        self.nodes = config.get("nodes", [])
        self.ssh_user = config.get("ssh_user", "root")
        self.ssh_key = config.get("ssh_key", "~/.ssh/id_rsa")
        self.ssh_timeout = config.get("ssh_timeout", 30)
        self.max_workers = config.get("max_workers", 8)

        # PFC检测阈值
        pfc_thresholds = config.get("pfc_thresholds", {})
        self.pause_rate_warning = pfc_thresholds.get("pause_rate_warning", 100)  # 每秒
        self.pause_rate_storm = pfc_thresholds.get("pause_rate_storm", 1000)
        self.pause_duration_warning_us = pfc_thresholds.get("pause_duration_warning_us", 100)
        self.pause_duration_critical_us = pfc_thresholds.get("pause_duration_critical_us", 1000)
        self.deadlock_detection_cycles = pfc_thresholds.get("deadlock_detection_cycles", 3)

        # RoCE使用的优先级（通常是TC3）
        self.roce_priorities: List[int] = config.get("roce_priorities", [3])

        # 历史快照
        self.snapshots: List[PFCSnapshot] = []
        # 持续暂停计数（连续检测到高PFC率的次数）
        self.sustained_pause_count: Dict[str, int] = defaultdict(int)

    def _run_remote(self, host: str, command: str) -> Tuple[str, int]:
        """在远程主机上执行命令"""
        ssh_cmd = [
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", f"ConnectTimeout={self.ssh_timeout}",
            "-o", "BatchMode=yes",
            "-i", self.ssh_key,
            f"{self.ssh_user}@{host}",
            command
        ]
        try:
            result = subprocess.run(
                ssh_cmd, capture_output=True, text=True,
                timeout=self.ssh_timeout + 10
            )
            return result.stdout, result.returncode
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.error(f"远程命令执行失败 {host}: {e}")
            return "", -1

    def _collect_pfc_counters(self, hostname: str, ip_address: str,
                               device: str, port: int) -> Optional[PFCCounters]:
        """
        采集单个端口的PFC计数器

        通过ethtool或mlnx_qos获取PFC计数器数据
        """
        counters = PFCCounters(
            node=hostname,
            device=device,
            port=port,
            timestamp=datetime.datetime.now().isoformat()
        )

        # 方法1: 使用ethtool获取PFC统计（RoCE/Ethernet）
        netdev_cmd = f"cat /sys/class/infiniband/{device}/ports/{port}/gid_attrs/ndevs/0 2>/dev/null"
        stdout, rc = self._run_remote(ip_address, netdev_cmd)
        netdev = stdout.strip() if rc == 0 else ""

        if netdev:
            # 使用ethtool获取PFC计数器
            ethtool_cmd = f"ethtool -S {netdev} 2>/dev/null | grep -i pfc"
            stdout, rc = self._run_remote(ip_address, ethtool_cmd)
            if rc == 0:
                counters = self._parse_ethtool_pfc(stdout, counters)
        else:
            # 方法2: 使用mlnx_qos
            mlnx_cmd = f"mlnx_qos -i {device} --pfc_counters 2>/dev/null"
            stdout, rc = self._run_remote(ip_address, mlnx_cmd)
            if rc == 0:
                counters = self._parse_mlnx_qos_pfc(stdout, counters)

        return counters

    def _parse_ethtool_pfc(self, output: str, counters: PFCCounters) -> PFCCounters:
        """
        解析ethtool -S输出中的PFC相关计数器

        常见格式:
            rx_prio0_pause: 0
            rx_prio1_pause: 0
            ...
            tx_prio0_pause: 0
            tx_prio1_pause: 1234
            rx_prio0_pause_duration: 0
            ...
        """
        for line in output.split("\n"):
            line = line.strip()

            # 匹配 tx_prioN_pause
            tx_match = re.match(r"tx_prio(\d+)_pause:\s*(\d+)", line)
            if tx_match:
                prio = int(tx_match.group(1))
                value = int(tx_match.group(2))
                if 0 <= prio < 8:
                    counters.tx_pause[prio] = value
                continue

            # 匹配 rx_prioN_pause
            rx_match = re.match(r"rx_prio(\d+)_pause:\s*(\d+)", line)
            if rx_match:
                prio = int(rx_match.group(1))
                value = int(rx_match.group(2))
                if 0 <= prio < 8:
                    counters.rx_pause[prio] = value
                continue

            # 匹配 tx_prioN_pause_duration
            tx_dur_match = re.match(r"tx_prio(\d+)_pause_duration:\s*(\d+)", line)
            if tx_dur_match:
                prio = int(tx_dur_match.group(1))
                value = int(tx_dur_match.group(2))
                if 0 <= prio < 8:
                    counters.tx_pause_duration[prio] = value
                continue

            # 匹配 rx_prioN_pause_duration
            rx_dur_match = re.match(r"rx_prio(\d+)_pause_duration:\s*(\d+)", line)
            if rx_dur_match:
                prio = int(rx_dur_match.group(1))
                value = int(rx_dur_match.group(2))
                if 0 <= prio < 8:
                    counters.rx_pause_duration[prio] = value

        return counters

    def _parse_mlnx_qos_pfc(self, output: str, counters: PFCCounters) -> PFCCounters:
        """解析mlnx_qos --pfc_counters输出"""
        in_tx_section = False
        in_rx_section = False

        for line in output.split("\n"):
            line = line.strip().lower()
            if "tx" in line and "pause" in line:
                in_tx_section = True
                in_rx_section = False
                continue
            if "rx" in line and "pause" in line:
                in_rx_section = True
                in_tx_section = False
                continue

            # 解析数值行，格式如: "0  0  0  1234  0  0  0  0"
            values = re.findall(r"\d+", line)
            if len(values) == 8:
                for i, v in enumerate(values):
                    if in_tx_section:
                        counters.tx_pause[i] = int(v)
                    elif in_rx_section:
                        counters.rx_pause[i] = int(v)

        return counters

    def collect_snapshot(self) -> PFCSnapshot:
        """
        采集整个集群的PFC计数器快照

        Returns:
            PFCSnapshot对象
        """
        snapshot = PFCSnapshot(timestamp=datetime.datetime.now().isoformat())

        logger.info("采集集群PFC计数器快照...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for node in self.nodes:
                hostname = node.get("hostname", "")
                ip_address = node.get("ip", hostname)
                devices = node.get("rdma_devices", [{"name": "mlx5_0", "ports": [1]}])

                for dev in devices:
                    dev_name = dev.get("name", "mlx5_0")
                    for port_num in dev.get("ports", [1]):
                        future = executor.submit(
                            self._collect_pfc_counters,
                            hostname, ip_address, dev_name, port_num
                        )
                        key = f"{hostname}:{dev_name}:{port_num}"
                        futures[future] = key

            for future in as_completed(futures):
                key = futures[future]
                try:
                    result = future.result()
                    if result:
                        snapshot.counters[key] = result
                except Exception as e:
                    logger.error(f"采集PFC计数器失败 {key}: {e}")

        self.snapshots.append(snapshot)
        logger.info(f"PFC快照采集完成: {len(snapshot.counters)} 个端口")
        return snapshot

    def _compute_pfc_rates(self, prev: PFCSnapshot,
                            curr: PFCSnapshot) -> Dict[str, Dict[int, float]]:
        """
        计算两次快照之间的PFC暂停帧速率

        Returns:
            {link_key: {priority: rate_per_sec}}
        """
        rates = {}

        t_prev = datetime.datetime.fromisoformat(prev.timestamp)
        t_curr = datetime.datetime.fromisoformat(curr.timestamp)
        elapsed_seconds = max((t_curr - t_prev).total_seconds(), 1.0)

        for key, curr_counters in curr.counters.items():
            if key not in prev.counters:
                continue

            prev_counters = prev.counters[key]
            link_rates = {}

            for prio in range(8):
                tx_diff = curr_counters.tx_pause.get(prio, 0) - prev_counters.tx_pause.get(prio, 0)
                rx_diff = curr_counters.rx_pause.get(prio, 0) - prev_counters.rx_pause.get(prio, 0)
                total_diff = max(tx_diff + rx_diff, 0)
                link_rates[prio] = total_diff / elapsed_seconds

            rates[key] = link_rates

        return rates

    def _detect_pfc_storm(self, rates: Dict[str, Dict[int, float]]) -> List[str]:
        """
        检测PFC风暴

        当某个端口某个优先级的PFC暂停帧速率超过阈值时判定为风暴
        """
        storms = []
        for key, prio_rates in rates.items():
            for prio in self.roce_priorities:
                rate = prio_rates.get(prio, 0.0)
                if rate >= self.pause_rate_storm:
                    storms.append(
                        f"PFC风暴: {key} 优先级{prio} "
                        f"暂停帧速率={rate:.0f}/s (阈值={self.pause_rate_storm}/s)"
                    )
                    logger.warning(storms[-1])

        return storms

    def _detect_cyclic_dependency(self, snapshot: PFCSnapshot) -> List[DeadlockPattern]:
        """
        检测PFC暂停帧的循环依赖（潜在死锁）

        原理：如果节点A向B发送PFC暂停，B同时向C发送PFC暂停，
        C又向A发送PFC暂停，则形成循环依赖，可能导致死锁。

        简化检测：构建PFC暂停方向图，检测是否存在环
        """
        patterns = []

        for prio in self.roce_priorities:
            # 构建暂停方向图：高TX暂停意味着该节点在向对端发送暂停
            # 高RX暂停意味着该节点在接收对端的暂停
            active_senders = set()  # 正在发送大量PFC暂停帧的节点
            active_receivers = set()  # 正在接收大量PFC暂停帧的节点

            for key, counters in snapshot.counters.items():
                node = counters.node
                tx_count = counters.tx_pause.get(prio, 0)
                rx_count = counters.rx_pause.get(prio, 0)

                if tx_count > self.pause_rate_warning:
                    active_senders.add(node)
                if rx_count > self.pause_rate_warning:
                    active_receivers.add(node)

            # 同时是发送者和接收者的节点 = 潜在死锁参与者
            both = active_senders & active_receivers
            if len(both) >= 2:
                pattern = DeadlockPattern(
                    pattern_type="循环PFC暂停依赖",
                    involved_nodes=list(both),
                    involved_links=[],
                    priority=prio,
                    description=(
                        f"检测到 {len(both)} 个节点同时发送和接收PFC暂停帧 "
                        f"(优先级{prio})，可能形成死锁环路"
                    ),
                    severity=min(len(both) / 4.0, 1.0),
                )
                patterns.append(pattern)
                logger.warning(f"检测到潜在死锁模式: {pattern.description}")

        return patterns

    def _detect_sustained_pause(self, rates: Dict[str, Dict[int, float]]) -> List[str]:
        """
        检测持续的PFC暂停

        如果某个端口连续多个检测周期都有高PFC暂停率，
        说明可能已经进入死锁状态
        """
        hotspots = []

        for key, prio_rates in rates.items():
            for prio in self.roce_priorities:
                rate = prio_rates.get(prio, 0.0)
                counter_key = f"{key}:prio{prio}"

                if rate >= self.pause_rate_warning:
                    self.sustained_pause_count[counter_key] += 1
                    count = self.sustained_pause_count[counter_key]

                    if count >= self.deadlock_detection_cycles:
                        hotspots.append(
                            f"持续PFC暂停: {key} 优先级{prio}, "
                            f"已持续 {count} 个检测周期"
                        )
                else:
                    self.sustained_pause_count[counter_key] = 0

        return hotspots

    def analyze(self) -> PFCAnalysisResult:
        """
        执行PFC死锁检测分析

        需要至少两个快照才能计算速率

        Returns:
            PFCAnalysisResult分析结果
        """
        result = PFCAnalysisResult(
            check_time=datetime.datetime.now().isoformat(),
            overall_status=PFCStatus.NORMAL
        )

        if len(self.snapshots) < 2:
            result.recommendations.append("需要至少两个PFC快照才能进行分析，请再次采集")
            logger.warning("PFC快照不足，无法进行分析")
            return result

        prev_snapshot = self.snapshots[-2]
        curr_snapshot = self.snapshots[-1]

        # 计算PFC暂停帧速率
        rates = self._compute_pfc_rates(prev_snapshot, curr_snapshot)

        # 检测PFC风暴
        result.storm_alerts = self._detect_pfc_storm(rates)

        # 检测循环依赖
        result.deadlock_patterns = self._detect_cyclic_dependency(curr_snapshot)

        # 检测持续暂停
        result.hotspots = self._detect_sustained_pause(rates)

        # 计算统计信息
        all_rates = []
        for prio_rates in rates.values():
            for prio in self.roce_priorities:
                all_rates.append(prio_rates.get(prio, 0.0))

        if all_rates:
            result.statistics = {
                "avg_pfc_rate": sum(all_rates) / len(all_rates),
                "max_pfc_rate": max(all_rates),
                "ports_with_pfc": sum(1 for r in all_rates if r > 0),
                "total_ports": len(all_rates),
            }

        # 判断每个节点状态
        node_max_rates: Dict[str, float] = defaultdict(float)
        for key, prio_rates in rates.items():
            node = key.split(":")[0]
            for prio in self.roce_priorities:
                rate = prio_rates.get(prio, 0.0)
                node_max_rates[node] = max(node_max_rates[node], rate)

        for node, max_rate in node_max_rates.items():
            if max_rate >= self.pause_rate_storm:
                result.node_status[node] = PFCStatus.STORM
            elif max_rate >= self.pause_rate_warning:
                result.node_status[node] = PFCStatus.ELEVATED
            else:
                result.node_status[node] = PFCStatus.NORMAL

        # 确定整体状态
        if result.deadlock_patterns:
            max_severity = max(p.severity for p in result.deadlock_patterns)
            if max_severity >= 0.7:
                result.overall_status = PFCStatus.DEADLOCK
            else:
                result.overall_status = PFCStatus.DEADLOCK_RISK
        elif result.storm_alerts:
            result.overall_status = PFCStatus.STORM
        elif result.hotspots:
            result.overall_status = PFCStatus.ELEVATED
        else:
            result.overall_status = PFCStatus.NORMAL

        # 生成建议
        result.recommendations = self._generate_recommendations(result)

        logger.info(f"PFC分析完成: 整体状态={result.overall_status.value}")
        return result

    def _generate_recommendations(self, result: PFCAnalysisResult) -> List[str]:
        """根据分析结果生成运维建议"""
        recs = []

        if result.overall_status == PFCStatus.DEADLOCK:
            recs.append("紧急：检测到PFC死锁，建议立即介入排查")
            recs.append("临时措施：可以考虑暂时禁用受影响优先级的PFC")
            recs.append("检查是否存在路由环路（使用ibdiagnet --r检查）")
            recs.append("验证ECN是否正确配置并生效")

        if result.overall_status == PFCStatus.STORM:
            recs.append("检测到PFC风暴，检查是否有节点发送异常流量")
            recs.append("验证PFC watchdog是否已启用")
            recs.append("检查交换机buffer配置和水线设置")

        if result.overall_status in (PFCStatus.ELEVATED, PFCStatus.DEADLOCK_RISK):
            recs.append("PFC暂停帧偏多，建议检查网络拥塞情况")
            recs.append("确认DCQCN/ECN配置是否最优")
            recs.append("检查流量是否均匀分布在各条链路上")

        if result.hotspots:
            recs.append(f"发现 {len(result.hotspots)} 个PFC热点，建议重点排查")

        if not recs:
            recs.append("PFC状态正常，无需额外操作")

        return recs
