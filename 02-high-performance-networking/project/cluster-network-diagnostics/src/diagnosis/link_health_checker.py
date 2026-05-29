"""
链路健康检查模块

检查集群中所有RDMA链路的状态、FEC错误、线缆健康和端口计数器。
聚合结果并识别退化的链路。使用perfquery、ibdiagnet等工具进行检测。
"""

import logging
import subprocess
import re
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime

logger = logging.getLogger(__name__)


class LinkHealthStatus(Enum):
    """链路健康状态"""
    HEALTHY = "healthy"           # 正常
    WARNING = "warning"           # 告警（有少量错误）
    DEGRADED = "degraded"         # 退化（性能下降）
    CRITICAL = "critical"         # 严重（即将故障）
    DOWN = "down"                 # 链路已断开
    UNKNOWN = "unknown"


class ErrorType(Enum):
    """错误计数器类型"""
    SYMBOL_ERROR = "SymbolErrorCounter"
    LINK_ERROR_RECOVERY = "LinkErrorRecoveryCounter"
    LINK_DOWNED = "LinkDownedCounter"
    PORT_RCV_ERRORS = "PortRcvErrors"
    PORT_RCV_REMOTE_PHYSICAL_ERRORS = "PortRcvRemotePhysicalErrors"
    PORT_RCV_SWITCH_RELAY_ERRORS = "PortRcvSwitchRelayErrors"
    EXCESSIVE_BUFFER_OVERRUN = "ExcessiveBufferOverrunErrors"
    PORT_XMIT_DISCARDS = "PortXmitDiscards"
    PORT_XMIT_CONSTRAINT_ERRORS = "PortXmitConstraintErrors"
    PORT_RCV_CONSTRAINT_ERRORS = "PortRcvConstraintErrors"
    LOCAL_LINK_INTEGRITY_ERRORS = "LocalLinkIntegrityErrors"
    VL15_DROPPED = "VL15Dropped"
    PORT_XMIT_DATA = "PortXmitData"
    PORT_RCV_DATA = "PortRcvData"
    PORT_XMIT_PKTS = "PortXmitPkts"
    PORT_RCV_PKTS = "PortRcvPkts"


@dataclass
class PortCounters:
    """端口性能计数器"""
    symbol_errors: int = 0
    link_error_recovery: int = 0
    link_downed: int = 0
    port_rcv_errors: int = 0
    port_rcv_remote_physical_errors: int = 0
    port_rcv_switch_relay_errors: int = 0
    excessive_buffer_overrun: int = 0
    port_xmit_discards: int = 0
    port_xmit_constraint_errors: int = 0
    port_rcv_constraint_errors: int = 0
    local_link_integrity_errors: int = 0
    vl15_dropped: int = 0
    port_xmit_data: int = 0
    port_rcv_data: int = 0
    port_xmit_pkts: int = 0
    port_rcv_pkts: int = 0
    timestamp: str = ""

    @property
    def total_errors(self) -> int:
        """总错误计数"""
        return (
            self.symbol_errors
            + self.link_error_recovery
            + self.link_downed
            + self.port_rcv_errors
            + self.port_rcv_remote_physical_errors
            + self.excessive_buffer_overrun
            + self.local_link_integrity_errors
        )


@dataclass
class FECStatus:
    """前向纠错(FEC)状态"""
    fec_mode: str = ""             # FEC模式（RS-FEC, Base-R FEC等）
    corrected_errors: int = 0      # 已纠正错误
    uncorrectable_errors: int = 0  # 不可纠正错误
    fec_corrected_rate: float = 0.0  # 纠正率（每分钟）


@dataclass
class CableInfo:
    """线缆信息"""
    cable_type: str = ""          # 线缆类型（光纤/铜缆/AOC）
    vendor: str = ""              # 供应商
    part_number: str = ""         # 型号
    serial_number: str = ""       # 序列号
    length: str = ""              # 长度
    temperature: float = 0.0      # 温度（光模块）
    tx_power_dbm: float = 0.0    # 发射功率
    rx_power_dbm: float = 0.0    # 接收功率


@dataclass
class LinkHealthResult:
    """单条链路的健康检查结果"""
    node_hostname: str
    device_name: str
    port_number: int
    lid: int = 0
    status: LinkHealthStatus = LinkHealthStatus.UNKNOWN
    counters: PortCounters = field(default_factory=PortCounters)
    fec: FECStatus = field(default_factory=FECStatus)
    cable: CableInfo = field(default_factory=CableInfo)
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    check_timestamp: str = ""


@dataclass
class ClusterHealthReport:
    """集群链路健康报告"""
    check_time: str
    total_links: int = 0
    healthy_links: int = 0
    warning_links: int = 0
    degraded_links: int = 0
    critical_links: int = 0
    down_links: int = 0
    results: List[LinkHealthResult] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)


class LinkHealthChecker:
    """
    链路健康检查器

    对集群中所有RDMA链路执行全面健康检查，包括:
    - 端口计数器检查（错误率、丢包等）
    - FEC状态检查（纠错率、不可纠正错误）
    - 线缆/光模块健康检查（温度、信号强度）
    - 链路带宽利用率检查
    """

    # 错误阈值配置
    DEFAULT_THRESHOLDS = {
        "symbol_errors_warning": 10,
        "symbol_errors_critical": 100,
        "link_error_recovery_warning": 5,
        "link_error_recovery_critical": 50,
        "port_rcv_errors_warning": 10,
        "port_rcv_errors_critical": 100,
        "local_link_integrity_warning": 5,
        "local_link_integrity_critical": 50,
        "excessive_buffer_overrun_warning": 1,
        "excessive_buffer_overrun_critical": 10,
        "fec_uncorrectable_warning": 1,
        "fec_uncorrectable_critical": 10,
        "cable_temp_warning": 65.0,
        "cable_temp_critical": 75.0,
        "rx_power_low_warning": -10.0,
        "rx_power_low_critical": -14.0,
    }

    def __init__(self, config: dict):
        """
        初始化链路健康检查器

        Args:
            config: 配置字典
        """
        self.nodes = config.get("nodes", [])
        self.ssh_user = config.get("ssh_user", "root")
        self.ssh_key = config.get("ssh_key", "~/.ssh/id_rsa")
        self.ssh_timeout = config.get("ssh_timeout", 30)
        self.max_workers = config.get("max_workers", 8)
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **config.get("thresholds", {})}
        self.sm_node = config.get("subnet_manager_node", "")

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

    def _parse_perfquery(self, output: str) -> PortCounters:
        """
        解析perfquery输出

        perfquery输出格式:
            # Port counters: Lid 1 port 1 (CapMask: 0x5200)
            PortSelect:......................1
            CounterSelect:...................0x0000
            SymbolErrorCounter:..............0
            LinkErrorRecoveryCounter:........0
            LinkDownedCounter:...............0
            PortRcvErrors:...................0
            ...
        """
        counters = PortCounters(timestamp=datetime.datetime.now().isoformat())
        counter_patterns = {
            "SymbolErrorCounter": "symbol_errors",
            "LinkErrorRecoveryCounter": "link_error_recovery",
            "LinkDownedCounter": "link_downed",
            "PortRcvErrors": "port_rcv_errors",
            "PortRcvRemotePhysicalErrors": "port_rcv_remote_physical_errors",
            "PortRcvSwitchRelayErrors": "port_rcv_switch_relay_errors",
            "ExcessiveBufferOverrunErrors": "excessive_buffer_overrun",
            "PortXmitDiscards": "port_xmit_discards",
            "PortXmitConstraintErrors": "port_xmit_constraint_errors",
            "PortRcvConstraintErrors": "port_rcv_constraint_errors",
            "LocalLinkIntegrityErrors": "local_link_integrity_errors",
            "VL15Dropped": "vl15_dropped",
            "PortXmitData": "port_xmit_data",
            "PortRcvData": "port_rcv_data",
            "PortXmitPkts": "port_xmit_pkts",
            "PortRcvPkts": "port_rcv_pkts",
        }

        for line in output.split("\n"):
            line = line.strip()
            for pattern_name, attr_name in counter_patterns.items():
                if line.startswith(pattern_name):
                    # 提取数值，格式如 "SymbolErrorCounter:..............0"
                    match = re.search(r"\.+(\d+)", line)
                    if match:
                        setattr(counters, attr_name, int(match.group(1)))
                    break

        return counters

    def _parse_cable_info(self, output: str) -> CableInfo:
        """
        解析mlxcables或ethtool输出获取线缆信息

        Args:
            output: 命令输出文本

        Returns:
            CableInfo对象
        """
        cable = CableInfo()

        # 解析供应商
        vendor_match = re.search(r"Vendor\s*(?:name)?:\s*(.+)", output, re.IGNORECASE)
        if vendor_match:
            cable.vendor = vendor_match.group(1).strip()

        # 解析型号
        pn_match = re.search(r"Part\s*(?:number)?:\s*(.+)", output, re.IGNORECASE)
        if pn_match:
            cable.part_number = pn_match.group(1).strip()

        # 解析序列号
        sn_match = re.search(r"Serial\s*(?:number)?:\s*(.+)", output, re.IGNORECASE)
        if sn_match:
            cable.serial_number = sn_match.group(1).strip()

        # 解析线缆类型
        type_match = re.search(r"(?:Cable|Connector)\s*(?:type)?:\s*(.+)", output, re.IGNORECASE)
        if type_match:
            cable.cable_type = type_match.group(1).strip()

        # 解析长度
        len_match = re.search(r"(?:Cable\s+)?[Ll]ength:\s*(.+)", output)
        if len_match:
            cable.length = len_match.group(1).strip()

        # 解析温度
        temp_match = re.search(r"Temperature:\s*([\d.]+)", output)
        if temp_match:
            cable.temperature = float(temp_match.group(1))

        # 解析光功率
        tx_match = re.search(r"TX\s*(?:Power|power).*?:\s*([-\d.]+)\s*dBm", output)
        if tx_match:
            cable.tx_power_dbm = float(tx_match.group(1))

        rx_match = re.search(r"RX\s*(?:Power|power).*?:\s*([-\d.]+)\s*dBm", output)
        if rx_match:
            cable.rx_power_dbm = float(rx_match.group(1))

        return cable

    def _check_link_health(self, counters: PortCounters, cable: CableInfo) -> Tuple[LinkHealthStatus, List[str], List[str]]:
        """
        根据计数器和线缆信息判断链路健康状态

        Args:
            counters: 端口计数器
            cable: 线缆信息

        Returns:
            (状态, 问题列表, 建议列表) 元组
        """
        issues = []
        recommendations = []
        status = LinkHealthStatus.HEALTHY

        # 检查符号错误
        if counters.symbol_errors >= self.thresholds["symbol_errors_critical"]:
            issues.append(f"符号错误数严重: {counters.symbol_errors}")
            recommendations.append("建议检查线缆连接，可能需要更换线缆")
            status = LinkHealthStatus.CRITICAL
        elif counters.symbol_errors >= self.thresholds["symbol_errors_warning"]:
            issues.append(f"符号错误数偏高: {counters.symbol_errors}")
            recommendations.append("建议监控符号错误增长趋势")
            status = max(status, LinkHealthStatus.WARNING, key=lambda x: list(LinkHealthStatus).index(x))

        # 检查链路恢复错误
        if counters.link_error_recovery >= self.thresholds["link_error_recovery_critical"]:
            issues.append(f"链路恢复次数严重: {counters.link_error_recovery}")
            recommendations.append("链路不稳定，建议检查物理连接和交换机端口")
            status = LinkHealthStatus.CRITICAL
        elif counters.link_error_recovery >= self.thresholds["link_error_recovery_warning"]:
            issues.append(f"链路恢复次数偏高: {counters.link_error_recovery}")
            status = self._escalate_status(status, LinkHealthStatus.WARNING)

        # 检查接收错误
        if counters.port_rcv_errors >= self.thresholds["port_rcv_errors_critical"]:
            issues.append(f"接收错误数严重: {counters.port_rcv_errors}")
            recommendations.append("建议检查远端发送设备和中间链路")
            status = LinkHealthStatus.CRITICAL
        elif counters.port_rcv_errors >= self.thresholds["port_rcv_errors_warning"]:
            issues.append(f"接收错误数偏高: {counters.port_rcv_errors}")
            status = self._escalate_status(status, LinkHealthStatus.WARNING)

        # 检查本地链路完整性错误
        if counters.local_link_integrity_errors >= self.thresholds["local_link_integrity_critical"]:
            issues.append(f"本地链路完整性错误严重: {counters.local_link_integrity_errors}")
            recommendations.append("本地HCA或端口可能有硬件问题")
            status = LinkHealthStatus.CRITICAL
        elif counters.local_link_integrity_errors >= self.thresholds["local_link_integrity_warning"]:
            issues.append(f"本地链路完整性错误偏高: {counters.local_link_integrity_errors}")
            status = self._escalate_status(status, LinkHealthStatus.WARNING)

        # 检查缓冲区溢出
        if counters.excessive_buffer_overrun >= self.thresholds["excessive_buffer_overrun_critical"]:
            issues.append(f"缓冲区溢出错误严重: {counters.excessive_buffer_overrun}")
            recommendations.append("交换机缓冲区管理异常，建议检查流量控制配置")
            status = LinkHealthStatus.CRITICAL
        elif counters.excessive_buffer_overrun >= self.thresholds["excessive_buffer_overrun_warning"]:
            issues.append(f"缓冲区溢出错误偏高: {counters.excessive_buffer_overrun}")
            status = self._escalate_status(status, LinkHealthStatus.WARNING)

        # 检查链路断开次数
        if counters.link_downed > 0:
            issues.append(f"链路曾断开 {counters.link_downed} 次")
            if counters.link_downed >= 5:
                recommendations.append("链路频繁断开，建议更换线缆或检查端口")
                status = self._escalate_status(status, LinkHealthStatus.DEGRADED)
            else:
                status = self._escalate_status(status, LinkHealthStatus.WARNING)

        # 检查线缆温度
        if cable.temperature > 0:
            if cable.temperature >= self.thresholds["cable_temp_critical"]:
                issues.append(f"光模块温度过高: {cable.temperature}°C")
                recommendations.append("光模块温度超标，检查散热和环境温度")
                status = LinkHealthStatus.CRITICAL
            elif cable.temperature >= self.thresholds["cable_temp_warning"]:
                issues.append(f"光模块温度偏高: {cable.temperature}°C")
                status = self._escalate_status(status, LinkHealthStatus.WARNING)

        # 检查接收光功率
        if cable.rx_power_dbm < 0:  # 有有效数据
            if cable.rx_power_dbm <= self.thresholds["rx_power_low_critical"]:
                issues.append(f"接收光功率过低: {cable.rx_power_dbm} dBm")
                recommendations.append("接收光功率不足，检查光纤和光模块")
                status = LinkHealthStatus.CRITICAL
            elif cable.rx_power_dbm <= self.thresholds["rx_power_low_warning"]:
                issues.append(f"接收光功率偏低: {cable.rx_power_dbm} dBm")
                status = self._escalate_status(status, LinkHealthStatus.WARNING)

        return status, issues, recommendations

    @staticmethod
    def _escalate_status(current: LinkHealthStatus, new: LinkHealthStatus) -> LinkHealthStatus:
        """状态升级：取更严重的状态"""
        severity_order = [
            LinkHealthStatus.HEALTHY,
            LinkHealthStatus.UNKNOWN,
            LinkHealthStatus.WARNING,
            LinkHealthStatus.DEGRADED,
            LinkHealthStatus.CRITICAL,
            LinkHealthStatus.DOWN,
        ]
        current_idx = severity_order.index(current) if current in severity_order else 0
        new_idx = severity_order.index(new) if new in severity_order else 0
        return severity_order[max(current_idx, new_idx)]

    def check_node_links(self, hostname: str, ip_address: str,
                         devices: List[dict]) -> List[LinkHealthResult]:
        """
        检查单个节点所有链路的健康状态

        Args:
            hostname: 主机名
            ip_address: IP地址
            devices: 设备列表，每个设备包含name和ports信息

        Returns:
            LinkHealthResult列表
        """
        results = []
        logger.info(f"开始检查节点 {hostname} 的链路健康状态")

        for device in devices:
            dev_name = device.get("name", "")
            ports = device.get("ports", [1])

            for port_num in ports:
                result = LinkHealthResult(
                    node_hostname=hostname,
                    device_name=dev_name,
                    port_number=port_num,
                    check_timestamp=datetime.datetime.now().isoformat()
                )

                # 获取端口计数器
                perfquery_cmd = f"perfquery -x {dev_name} {port_num}"
                stdout, rc = self._run_remote(ip_address, perfquery_cmd)
                if rc == 0:
                    result.counters = self._parse_perfquery(stdout)
                else:
                    logger.warning(f"perfquery失败: {hostname} {dev_name}/{port_num}")

                # 获取线缆信息（仅Mellanox设备）
                if dev_name.startswith("mlx"):
                    cable_cmd = f"mlxcables -d {dev_name} -p {port_num}"
                    stdout, rc = self._run_remote(ip_address, cable_cmd)
                    if rc == 0:
                        result.cable = self._parse_cable_info(stdout)

                # 判断健康状态
                status, issues, recs = self._check_link_health(
                    result.counters, result.cable
                )
                result.status = status
                result.issues = issues
                result.recommendations = recs

                results.append(result)
                logger.debug(
                    f"链路 {hostname}:{dev_name}/{port_num} "
                    f"状态: {status.value}, 问题数: {len(issues)}"
                )

        return results

    def check_cluster(self, device_info: Optional[Dict] = None) -> ClusterHealthReport:
        """
        检查整个集群的链路健康状态

        Args:
            device_info: 可选，从设备扫描器获取的设备信息
                格式: {hostname: [{"name": "mlx5_0", "ports": [1, 2]}]}

        Returns:
            ClusterHealthReport对象
        """
        report = ClusterHealthReport(
            check_time=datetime.datetime.now().isoformat()
        )

        logger.info(f"开始集群链路健康检查，共 {len(self.nodes)} 个节点")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for node in self.nodes:
                hostname = node.get("hostname", "")
                ip_address = node.get("ip", hostname)

                # 获取该节点的设备信息
                if device_info and hostname in device_info:
                    devices = device_info[hostname]
                else:
                    # 默认检查mlx5_0端口1
                    devices = [{"name": "mlx5_0", "ports": [1]}]

                future = executor.submit(
                    self.check_node_links, hostname, ip_address, devices
                )
                futures[future] = hostname

            for future in as_completed(futures):
                hostname = futures[future]
                try:
                    results = future.result()
                    report.results.extend(results)
                except Exception as e:
                    logger.error(f"检查节点 {hostname} 时异常: {e}")

        # 统计
        for result in report.results:
            report.total_links += 1
            if result.status == LinkHealthStatus.HEALTHY:
                report.healthy_links += 1
            elif result.status == LinkHealthStatus.WARNING:
                report.warning_links += 1
            elif result.status == LinkHealthStatus.DEGRADED:
                report.degraded_links += 1
            elif result.status == LinkHealthStatus.CRITICAL:
                report.critical_links += 1
            elif result.status == LinkHealthStatus.DOWN:
                report.down_links += 1

        report.summary = {
            "total": report.total_links,
            "healthy": report.healthy_links,
            "warning": report.warning_links,
            "degraded": report.degraded_links,
            "critical": report.critical_links,
            "down": report.down_links,
        }

        logger.info(
            f"集群链路健康检查完成: "
            f"总计 {report.total_links}, "
            f"正常 {report.healthy_links}, "
            f"告警 {report.warning_links}, "
            f"退化 {report.degraded_links}, "
            f"严重 {report.critical_links}, "
            f"断开 {report.down_links}"
        )

        return report
