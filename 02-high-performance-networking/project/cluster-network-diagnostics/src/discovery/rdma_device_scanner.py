"""
RDMA设备扫描器模块

扫描集群节点上的RDMA设备(InfiniBand/RoCE)，收集HCA信息、端口状态、
链路速度、固件版本等关键信息。通过SSH远程执行ibv_devices、ibstat、
rdma link show等命令获取设备详情。
"""

import logging
import subprocess
import re
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class PortState(Enum):
    """端口状态枚举"""
    ACTIVE = "Active"
    DOWN = "Down"
    INIT = "Initializing"
    ARMED = "Armed"
    UNKNOWN = "Unknown"


class LinkLayer(Enum):
    """链路层类型"""
    INFINIBAND = "InfiniBand"
    ETHERNET = "Ethernet"  # RoCE
    UNKNOWN = "Unknown"


class LinkSpeed(Enum):
    """链路速度等级"""
    SDR = "SDR"       # 2.5 Gbps per lane
    DDR = "DDR"       # 5.0 Gbps per lane
    QDR = "QDR"       # 10.0 Gbps per lane
    FDR10 = "FDR10"   # 10.3125 Gbps per lane
    FDR = "FDR"       # 14.0625 Gbps per lane
    EDR = "EDR"       # 25.78125 Gbps per lane
    HDR = "HDR"       # 50.0 Gbps per lane
    NDR = "NDR"       # 100.0 Gbps per lane
    XDR = "XDR"       # 200.0 Gbps per lane (future)
    UNKNOWN = "Unknown"


@dataclass
class RDMAPort:
    """RDMA端口信息"""
    port_number: int
    state: PortState
    physical_state: str
    link_layer: LinkLayer
    link_speed: str
    link_width: str
    effective_bandwidth_gbps: float
    sm_lid: int = 0
    base_lid: int = 0
    gid: str = ""


@dataclass
class RDMADevice:
    """RDMA设备（HCA）信息"""
    device_name: str
    node_guid: str
    system_image_guid: str
    firmware_version: str
    hardware_version: str
    board_id: str
    num_ports: int
    ports: List[RDMAPort] = field(default_factory=list)
    driver: str = ""
    pci_address: str = ""


@dataclass
class NodeRDMAInfo:
    """单个节点的RDMA设备汇总信息"""
    hostname: str
    ip_address: str
    devices: List[RDMADevice] = field(default_factory=list)
    scan_timestamp: str = ""
    scan_success: bool = False
    error_message: str = ""
    total_ports: int = 0
    active_ports: int = 0

    def to_dict(self) -> dict:
        """转换为字典格式，方便序列化"""
        return asdict(self)


class RDMADeviceScanner:
    """
    RDMA设备扫描器

    通过SSH连接到集群各节点，执行RDMA诊断命令收集设备信息。
    支持并发扫描多个节点，汇总结果生成集群RDMA设备清单。
    """

    def __init__(self, config: dict):
        """
        初始化扫描器

        Args:
            config: 配置字典，包含节点列表、SSH参数等
        """
        self.nodes: List[dict] = config.get("nodes", [])
        self.ssh_user: str = config.get("ssh_user", "root")
        self.ssh_port: int = config.get("ssh_port", 22)
        self.ssh_key: str = config.get("ssh_key", "~/.ssh/id_rsa")
        self.ssh_timeout: int = config.get("ssh_timeout", 30)
        self.max_workers: int = config.get("max_scan_workers", 8)
        self.results: Dict[str, NodeRDMAInfo] = {}

    def _build_ssh_command(self, host: str, remote_cmd: str) -> List[str]:
        """
        构建SSH远程执行命令

        Args:
            host: 目标主机地址
            remote_cmd: 要在远程主机上执行的命令

        Returns:
            完整的SSH命令列表
        """
        cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout={}".format(self.ssh_timeout),
            "-o", "BatchMode=yes",
            "-i", self.ssh_key,
            "-p", str(self.ssh_port),
            f"{self.ssh_user}@{host}",
            remote_cmd
        ]
        return cmd

    def _run_remote_command(self, host: str, command: str) -> Tuple[str, str, int]:
        """
        在远程主机上执行命令

        Args:
            host: 目标主机
            command: 要执行的命令

        Returns:
            (stdout, stderr, returncode) 元组
        """
        ssh_cmd = self._build_ssh_command(host, command)
        logger.debug(f"执行远程命令: {host} -> {command}")

        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=self.ssh_timeout + 10
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            logger.error(f"命令超时: {host} -> {command}")
            return "", "Command timed out", -1
        except Exception as e:
            logger.error(f"执行命令异常: {host} -> {command}: {e}")
            return "", str(e), -1

    def _parse_ibv_devices(self, output: str) -> List[str]:
        """
        解析ibv_devices输出，获取设备名称列表

        示例输出:
            device                 node GUID
            ------              ----------------
            mlx5_0              7cfe900300e1a820
            mlx5_1              7cfe900300e1a821
        """
        devices = []
        for line in output.strip().split("\n"):
            line = line.strip()
            # 跳过标题行和分隔线
            if not line or "device" in line.lower() and "guid" in line.lower():
                continue
            if line.startswith("-"):
                continue
            parts = line.split()
            if len(parts) >= 1:
                device_name = parts[0]
                if device_name.startswith("mlx") or device_name.startswith("hfi"):
                    devices.append(device_name)
        return devices

    def _parse_ibstat_device(self, output: str, device_name: str) -> Optional[RDMADevice]:
        """
        解析单个设备的ibstat输出

        示例输出:
            CA 'mlx5_0'
                CA type: MT4123
                Number of ports: 2
                Firmware version: 20.31.1014
                Hardware version: 0
                Node GUID: 0x7cfe900300e1a820
                System image GUID: 0x7cfe900300e1a820
                Port 1:
                    State: Active
                    Physical state: LinkUp
                    Rate: 200
                    Base lid: 1
                    LMC: 0
                    SM lid: 1
                    Link layer: InfiniBand
        """
        device = RDMADevice(
            device_name=device_name,
            node_guid="",
            system_image_guid="",
            firmware_version="",
            hardware_version="",
            board_id="",
            num_ports=0,
            ports=[]
        )

        # 解析设备基本信息
        fw_match = re.search(r"Firmware version:\s*(.+)", output)
        if fw_match:
            device.firmware_version = fw_match.group(1).strip()

        hw_match = re.search(r"Hardware version:\s*(.+)", output)
        if hw_match:
            device.hardware_version = hw_match.group(1).strip()

        guid_match = re.search(r"Node GUID:\s*(0x[0-9a-fA-F]+)", output)
        if guid_match:
            device.node_guid = guid_match.group(1)

        sys_guid_match = re.search(r"System image GUID:\s*(0x[0-9a-fA-F]+)", output)
        if sys_guid_match:
            device.system_image_guid = sys_guid_match.group(1)

        ports_match = re.search(r"Number of ports:\s*(\d+)", output)
        if ports_match:
            device.num_ports = int(ports_match.group(1))

        board_match = re.search(r"Board id:\s*(.+)", output)
        if board_match:
            device.board_id = board_match.group(1).strip()

        # 解析端口信息
        port_sections = re.split(r"Port\s+(\d+):", output)
        for i in range(1, len(port_sections), 2):
            port_num = int(port_sections[i])
            port_text = port_sections[i + 1] if i + 1 < len(port_sections) else ""
            port = self._parse_port_info(port_num, port_text)
            device.ports.append(port)

        return device

    def _parse_port_info(self, port_number: int, port_text: str) -> RDMAPort:
        """解析端口信息文本"""
        state = PortState.UNKNOWN
        state_match = re.search(r"State:\s*(\w+)", port_text)
        if state_match:
            state_str = state_match.group(1)
            state_map = {
                "Active": PortState.ACTIVE,
                "Down": PortState.DOWN,
                "Initializing": PortState.INIT,
                "Armed": PortState.ARMED,
            }
            state = state_map.get(state_str, PortState.UNKNOWN)

        physical_state = ""
        phys_match = re.search(r"Physical state:\s*(.+)", port_text)
        if phys_match:
            physical_state = phys_match.group(1).strip()

        link_layer = LinkLayer.UNKNOWN
        ll_match = re.search(r"Link layer:\s*(.+)", port_text)
        if ll_match:
            ll_str = ll_match.group(1).strip()
            if "InfiniBand" in ll_str:
                link_layer = LinkLayer.INFINIBAND
            elif "Ethernet" in ll_str:
                link_layer = LinkLayer.ETHERNET

        # 解析速率
        rate_str = ""
        rate_match = re.search(r"Rate:\s*(\d+)", port_text)
        rate_gbps = 0.0
        if rate_match:
            rate_str = rate_match.group(1)
            rate_gbps = float(rate_str)

        # 解析LID
        base_lid = 0
        lid_match = re.search(r"Base lid:\s*(\d+)", port_text)
        if lid_match:
            base_lid = int(lid_match.group(1))

        sm_lid = 0
        sm_lid_match = re.search(r"SM lid:\s*(\d+)", port_text)
        if sm_lid_match:
            sm_lid = int(sm_lid_match.group(1))

        return RDMAPort(
            port_number=port_number,
            state=state,
            physical_state=physical_state,
            link_layer=link_layer,
            link_speed=rate_str,
            link_width="4x",  # 默认值，需要从rdma link获取精确值
            effective_bandwidth_gbps=rate_gbps,
            sm_lid=sm_lid,
            base_lid=base_lid
        )

    def _parse_rdma_link(self, output: str) -> Dict[str, dict]:
        """
        解析 rdma link show 输出，获取精确的链路宽度和速度

        示例输出:
            link mlx5_0/1 state ACTIVE physical_state LINK_UP netdev eth0
            link mlx5_0/2 state DOWN physical_state DISABLED
        """
        links = {}
        for line in output.strip().split("\n"):
            line = line.strip()
            if not line or not line.startswith("link"):
                continue
            # 解析 device/port
            match = re.match(r"link\s+(\w+)/(\d+)\s+state\s+(\w+)\s+physical_state\s+(\w+)", line)
            if match:
                dev_name = match.group(1)
                port_num = match.group(2)
                key = f"{dev_name}/{port_num}"
                links[key] = {
                    "state": match.group(3),
                    "physical_state": match.group(4),
                }
                # 检查是否有netdev绑定
                netdev_match = re.search(r"netdev\s+(\w+)", line)
                if netdev_match:
                    links[key]["netdev"] = netdev_match.group(1)
        return links

    def _get_pci_info(self, host: str, device_name: str) -> str:
        """获取设备的PCI地址"""
        cmd = f"readlink -f /sys/class/infiniband/{device_name}/device | xargs basename"
        stdout, _, rc = self._run_remote_command(host, cmd)
        if rc == 0:
            return stdout.strip()
        return ""

    def _get_driver_info(self, host: str, device_name: str) -> str:
        """获取设备驱动信息"""
        cmd = f"cat /sys/class/infiniband/{device_name}/device/driver/module/version 2>/dev/null || echo 'unknown'"
        stdout, _, rc = self._run_remote_command(host, cmd)
        if rc == 0:
            return stdout.strip()
        return "unknown"

    def scan_node(self, hostname: str, ip_address: str) -> NodeRDMAInfo:
        """
        扫描单个节点的RDMA设备信息

        Args:
            hostname: 节点主机名
            ip_address: 节点IP地址

        Returns:
            NodeRDMAInfo对象，包含该节点所有RDMA设备信息
        """
        import datetime
        node_info = NodeRDMAInfo(
            hostname=hostname,
            ip_address=ip_address,
            scan_timestamp=datetime.datetime.now().isoformat()
        )

        logger.info(f"开始扫描节点RDMA设备: {hostname} ({ip_address})")

        # 步骤1: 获取设备列表
        stdout, stderr, rc = self._run_remote_command(ip_address, "ibv_devices")
        if rc != 0:
            node_info.error_message = f"ibv_devices执行失败: {stderr}"
            logger.error(f"节点 {hostname} ibv_devices失败: {stderr}")
            return node_info

        device_names = self._parse_ibv_devices(stdout)
        if not device_names:
            node_info.error_message = "未发现RDMA设备"
            logger.warning(f"节点 {hostname} 未发现RDMA设备")
            return node_info

        logger.info(f"节点 {hostname} 发现 {len(device_names)} 个RDMA设备: {device_names}")

        # 步骤2: 获取每个设备的详细信息
        for dev_name in device_names:
            stdout, stderr, rc = self._run_remote_command(
                ip_address, f"ibstat {dev_name}"
            )
            if rc != 0:
                logger.warning(f"获取设备 {dev_name} 信息失败: {stderr}")
                continue

            device = self._parse_ibstat_device(stdout, dev_name)
            if device:
                # 获取PCI和驱动信息
                device.pci_address = self._get_pci_info(ip_address, dev_name)
                device.driver = self._get_driver_info(ip_address, dev_name)
                node_info.devices.append(device)

        # 步骤3: 获取rdma link补充信息
        stdout, stderr, rc = self._run_remote_command(ip_address, "rdma link show")
        if rc == 0:
            rdma_links = self._parse_rdma_link(stdout)
            # 用rdma link信息补充设备端口数据
            for device in node_info.devices:
                for port in device.ports:
                    key = f"{device.device_name}/{port.port_number}"
                    if key in rdma_links:
                        link_info = rdma_links[key]
                        logger.debug(f"补充 {key} 的链路信息: {link_info}")

        # 统计端口信息
        for device in node_info.devices:
            node_info.total_ports += len(device.ports)
            node_info.active_ports += sum(
                1 for p in device.ports if p.state == PortState.ACTIVE
            )

        node_info.scan_success = True
        logger.info(
            f"节点 {hostname} 扫描完成: "
            f"{len(node_info.devices)} 设备, "
            f"{node_info.active_ports}/{node_info.total_ports} 端口活跃"
        )
        return node_info

    def scan_cluster(self) -> Dict[str, NodeRDMAInfo]:
        """
        并发扫描整个集群的RDMA设备

        Returns:
            字典，key为主机名，value为NodeRDMAInfo
        """
        logger.info(f"开始集群RDMA设备扫描，共 {len(self.nodes)} 个节点")
        self.results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for node in self.nodes:
                hostname = node.get("hostname", "")
                ip_address = node.get("ip", hostname)
                future = executor.submit(self.scan_node, hostname, ip_address)
                futures[future] = hostname

            for future in as_completed(futures):
                hostname = futures[future]
                try:
                    result = future.result()
                    self.results[hostname] = result
                except Exception as e:
                    logger.error(f"扫描节点 {hostname} 异常: {e}")
                    self.results[hostname] = NodeRDMAInfo(
                        hostname=hostname,
                        ip_address="",
                        error_message=str(e)
                    )

        # 汇总统计
        total_devices = sum(len(n.devices) for n in self.results.values())
        total_active = sum(n.active_ports for n in self.results.values())
        total_ports = sum(n.total_ports for n in self.results.values())
        success_nodes = sum(1 for n in self.results.values() if n.scan_success)

        logger.info(
            f"集群扫描完成: {success_nodes}/{len(self.nodes)} 节点成功, "
            f"共 {total_devices} 设备, {total_active}/{total_ports} 端口活跃"
        )
        return self.results

    def get_cluster_summary(self) -> dict:
        """
        生成集群RDMA设备摘要

        Returns:
            包含集群设备统计信息的字典
        """
        summary = {
            "total_nodes": len(self.results),
            "successful_scans": sum(1 for n in self.results.values() if n.scan_success),
            "failed_scans": sum(1 for n in self.results.values() if not n.scan_success),
            "total_devices": sum(len(n.devices) for n in self.results.values()),
            "total_ports": sum(n.total_ports for n in self.results.values()),
            "active_ports": sum(n.active_ports for n in self.results.values()),
            "devices_by_type": {},
            "firmware_versions": {},
            "link_layers": {"InfiniBand": 0, "Ethernet": 0},
        }

        for node_info in self.results.values():
            for device in node_info.devices:
                # 按设备类型统计
                dev_type = device.board_id or device.device_name.split("_")[0]
                summary["devices_by_type"][dev_type] = (
                    summary["devices_by_type"].get(dev_type, 0) + 1
                )
                # 固件版本统计
                fw = device.firmware_version
                summary["firmware_versions"][fw] = (
                    summary["firmware_versions"].get(fw, 0) + 1
                )
                # 链路层类型统计
                for port in device.ports:
                    if port.link_layer == LinkLayer.INFINIBAND:
                        summary["link_layers"]["InfiniBand"] += 1
                    elif port.link_layer == LinkLayer.ETHERNET:
                        summary["link_layers"]["Ethernet"] += 1

        return summary

    def export_inventory(self, output_path: str) -> None:
        """
        导出设备清单到JSON文件

        Args:
            output_path: 输出文件路径
        """
        inventory = {
            "summary": self.get_cluster_summary(),
            "nodes": {}
        }
        for hostname, node_info in self.results.items():
            inventory["nodes"][hostname] = node_info.to_dict()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(inventory, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"设备清单已导出到: {output_path}")
