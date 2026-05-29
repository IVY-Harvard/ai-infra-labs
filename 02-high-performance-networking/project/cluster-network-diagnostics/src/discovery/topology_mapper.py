"""
网络拓扑映射模块

发现并映射集群网络拓扑结构，识别交换机、构建邻接图，
判断拓扑类型（Fat-Tree/Spine-Leaf等）。解析ibnetdiscover和
ibdiagnet的输出生成拓扑图数据结构。
"""

import logging
import subprocess
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """网络节点类型"""
    HOST = "host"           # 计算节点（HCA）
    SWITCH = "switch"       # 交换机
    ROUTER = "router"       # 路由器
    UNKNOWN = "unknown"


class TopologyType(Enum):
    """拓扑类型"""
    FAT_TREE = "fat_tree"
    SPINE_LEAF = "spine_leaf"
    DRAGONFLY = "dragonfly"
    TORUS = "torus"
    TREE = "tree"
    UNKNOWN = "unknown"


class SwitchTier(Enum):
    """交换机层级（Fat-Tree中的位置）"""
    LEAF = "leaf"           # 叶子交换机（连接主机）
    SPINE = "spine"         # 脊交换机
    CORE = "core"           # 核心交换机
    UNKNOWN = "unknown"


@dataclass
class NetworkPort:
    """网络端口"""
    port_number: int
    guid: str = ""
    state: str = "Active"
    speed: str = ""
    width: str = ""
    remote_guid: str = ""
    remote_port: int = 0


@dataclass
class NetworkNode:
    """网络中的一个节点（主机或交换机）"""
    guid: str
    name: str
    node_type: NodeType
    num_ports: int = 0
    ports: List[NetworkPort] = field(default_factory=list)
    tier: SwitchTier = SwitchTier.UNKNOWN
    description: str = ""
    lid: int = 0
    # 连接的邻居节点GUID列表
    neighbors: List[str] = field(default_factory=list)


@dataclass
class NetworkLink:
    """网络中的一条链路"""
    source_guid: str
    source_port: int
    dest_guid: str
    dest_port: int
    speed: str = ""
    width: str = ""
    state: str = "Active"
    link_id: str = ""

    def __post_init__(self):
        """生成链路唯一ID"""
        if not self.link_id:
            self.link_id = f"{self.source_guid}:{self.source_port}-{self.dest_guid}:{self.dest_port}"


@dataclass
class NetworkTopology:
    """完整的网络拓扑"""
    nodes: Dict[str, NetworkNode] = field(default_factory=dict)
    links: List[NetworkLink] = field(default_factory=list)
    adjacency: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    topology_type: TopologyType = TopologyType.UNKNOWN
    num_hosts: int = 0
    num_switches: int = 0
    num_links: int = 0
    switch_tiers: Dict[str, List[str]] = field(default_factory=dict)


class TopologyMapper:
    """
    网络拓扑映射器

    通过解析IB诊断工具的输出，构建完整的网络拓扑图，
    识别拓扑结构类型，计算路径和距离。
    """

    def __init__(self, config: dict):
        """
        初始化拓扑映射器

        Args:
            config: 配置字典
        """
        self.sm_node: str = config.get("subnet_manager_node", "")
        self.ssh_user: str = config.get("ssh_user", "root")
        self.ssh_key: str = config.get("ssh_key", "~/.ssh/id_rsa")
        self.ssh_timeout: int = config.get("ssh_timeout", 60)
        self.topology: NetworkTopology = NetworkTopology()

    def _run_command(self, command: str, host: Optional[str] = None) -> Tuple[str, int]:
        """
        执行命令（本地或远程）

        Args:
            command: 要执行的命令
            host: 如果指定，通过SSH远程执行

        Returns:
            (stdout, returncode) 元组
        """
        if host:
            full_cmd = [
                "ssh", "-o", "StrictHostKeyChecking=no",
                "-o", f"ConnectTimeout={self.ssh_timeout}",
                "-i", self.ssh_key,
                f"{self.ssh_user}@{host}",
                command
            ]
        else:
            full_cmd = command.split()

        try:
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=self.ssh_timeout + 30
            )
            return result.stdout, result.returncode
        except subprocess.TimeoutExpired:
            logger.error(f"命令超时: {command}")
            return "", -1
        except Exception as e:
            logger.error(f"命令执行异常: {command}: {e}")
            return "", -1

    def discover_topology(self) -> NetworkTopology:
        """
        执行拓扑发现流程

        Returns:
            NetworkTopology对象
        """
        logger.info("开始网络拓扑发现...")
        host = self.sm_node if self.sm_node else None

        # 步骤1: 运行ibnetdiscover获取拓扑原始数据
        stdout, rc = self._run_command("ibnetdiscover --ports", host)
        if rc == 0 and stdout:
            self._parse_ibnetdiscover(stdout)
            logger.info(f"ibnetdiscover解析完成: {len(self.topology.nodes)} 节点")
        else:
            logger.warning("ibnetdiscover执行失败，尝试使用备用方法")
            # 备用方法：使用ibswitches和ibhosts
            self._discover_via_individual_commands(host)

        # 步骤2: 构建邻接图
        self._build_adjacency_graph()

        # 步骤3: 识别交换机层级
        self._classify_switch_tiers()

        # 步骤4: 判断拓扑类型
        self._identify_topology_type()

        # 统计
        self.topology.num_hosts = sum(
            1 for n in self.topology.nodes.values() if n.node_type == NodeType.HOST
        )
        self.topology.num_switches = sum(
            1 for n in self.topology.nodes.values() if n.node_type == NodeType.SWITCH
        )
        self.topology.num_links = len(self.topology.links)

        logger.info(
            f"拓扑发现完成: {self.topology.num_hosts} 主机, "
            f"{self.topology.num_switches} 交换机, "
            f"{self.topology.num_links} 链路, "
            f"拓扑类型: {self.topology.topology_type.value}"
        )
        return self.topology

    def _parse_ibnetdiscover(self, output: str) -> None:
        """
        解析ibnetdiscover输出

        ibnetdiscover输出格式:
            Switch  36 "S-0002c90200400e30"  # "MF0;switch-ib01:IS5600/U1" enhanced port 0 lid 1
            [1]  "H-7cfe900300e1a820"[1]    # "node01 HCA-1" lid 2
            [2]  "H-7cfe900300e1a830"[1]    # "node02 HCA-1" lid 3
            ...

            Ca  2 "H-7cfe900300e1a820"  # "node01 HCA-1"
            [1]  "S-0002c90200400e30"[1]    # lid 1
        """
        current_node = None

        for line in output.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # 匹配交换机定义行
            switch_match = re.match(
                r'Switch\s+(\d+)\s+"([^"]+)"\s*#\s*"([^"]*)".*lid\s+(\d+)',
                line
            )
            if switch_match:
                num_ports = int(switch_match.group(1))
                guid = switch_match.group(2)
                name = switch_match.group(3)
                lid = int(switch_match.group(4))

                node = NetworkNode(
                    guid=guid,
                    name=name,
                    node_type=NodeType.SWITCH,
                    num_ports=num_ports,
                    lid=lid
                )
                self.topology.nodes[guid] = node
                current_node = node
                continue

            # 匹配主机(CA)定义行
            ca_match = re.match(
                r'Ca\s+(\d+)\s+"([^"]+)"\s*#\s*"([^"]*)"',
                line
            )
            if ca_match:
                num_ports = int(ca_match.group(1))
                guid = ca_match.group(2)
                name = ca_match.group(3)

                node = NetworkNode(
                    guid=guid,
                    name=name,
                    node_type=NodeType.HOST,
                    num_ports=num_ports
                )
                self.topology.nodes[guid] = node
                current_node = node
                continue

            # 匹配连接行（端口连接关系）
            conn_match = re.match(
                r'\[(\d+)\]\s+"([^"]+)"\[(\d+)\]',
                line
            )
            if conn_match and current_node:
                local_port = int(conn_match.group(1))
                remote_guid = conn_match.group(2)
                remote_port = int(conn_match.group(3))

                link = NetworkLink(
                    source_guid=current_node.guid,
                    source_port=local_port,
                    dest_guid=remote_guid,
                    dest_port=remote_port
                )
                self.topology.links.append(link)
                current_node.neighbors.append(remote_guid)

    def _discover_via_individual_commands(self, host: Optional[str]) -> None:
        """
        使用ibswitches和ibhosts单独发现节点

        当ibnetdiscover不可用时的备用方案
        """
        # 获取交换机列表
        stdout, rc = self._run_command("ibswitches", host)
        if rc == 0:
            self._parse_ibswitches(stdout)

        # 获取主机列表
        stdout, rc = self._run_command("ibhosts", host)
        if rc == 0:
            self._parse_ibhosts(stdout)

        # 获取链路信息
        stdout, rc = self._run_command("iblinkinfo", host)
        if rc == 0:
            self._parse_iblinkinfo(stdout)

    def _parse_ibswitches(self, output: str) -> None:
        """
        解析ibswitches输出

        格式: Switch : 0x0002c90200400e30 ports 36 "MF0;switch-01" enhanced port 0 lid 1
        """
        for line in output.strip().split("\n"):
            match = re.match(
                r'Switch\s*:\s*(0x[0-9a-fA-F]+)\s+ports\s+(\d+)\s+"([^"]*)".*lid\s+(\d+)',
                line
            )
            if match:
                guid = f"S-{match.group(1)[2:]}"
                num_ports = int(match.group(2))
                name = match.group(3)
                lid = int(match.group(4))

                node = NetworkNode(
                    guid=guid,
                    name=name,
                    node_type=NodeType.SWITCH,
                    num_ports=num_ports,
                    lid=lid
                )
                self.topology.nodes[guid] = node

    def _parse_ibhosts(self, output: str) -> None:
        """
        解析ibhosts输出

        格式: Ca : 0x7cfe900300e1a820 ports 2 "node01 HCA-1"
        """
        for line in output.strip().split("\n"):
            match = re.match(
                r'Ca\s*:\s*(0x[0-9a-fA-F]+)\s+ports\s+(\d+)\s+"([^"]*)"',
                line
            )
            if match:
                guid = f"H-{match.group(1)[2:]}"
                num_ports = int(match.group(2))
                name = match.group(3)

                node = NetworkNode(
                    guid=guid,
                    name=name,
                    node_type=NodeType.HOST,
                    num_ports=num_ports
                )
                self.topology.nodes[guid] = node

    def _parse_iblinkinfo(self, output: str) -> None:
        """
        解析iblinkinfo输出获取链路连接关系

        格式:
            0x0002c90200400e30      1    1[  ] ==( 4X 25.78125 Gbps Active)==>
                2    1[  ] "node01 HCA-1" (7cfe900300e1a820)
        """
        lines = output.strip().split("\n")
        i = 0
        while i < len(lines) - 1:
            line = lines[i].strip()
            # 匹配源端口信息
            src_match = re.match(
                r'(0x[0-9a-fA-F]+)\s+(\d+)\s+(\d+).*==\(.*\)==>',
                line
            )
            if src_match and i + 1 < len(lines):
                src_guid_raw = src_match.group(1)
                src_port = int(src_match.group(3))

                next_line = lines[i + 1].strip()
                dst_match = re.match(
                    r'(\d+)\s+(\d+).*\(([0-9a-fA-F]+)\)',
                    next_line
                )
                if dst_match:
                    dst_port = int(dst_match.group(2))
                    dst_guid_raw = dst_match.group(3)

                    link = NetworkLink(
                        source_guid=f"S-{src_guid_raw[2:]}",
                        source_port=src_port,
                        dest_guid=f"H-{dst_guid_raw}",
                        dest_port=dst_port
                    )
                    self.topology.links.append(link)
                i += 2
            else:
                i += 1

    def _build_adjacency_graph(self) -> None:
        """构建邻接图"""
        self.topology.adjacency = defaultdict(list)
        for link in self.topology.links:
            src = link.source_guid
            dst = link.dest_guid
            if dst not in self.topology.adjacency[src]:
                self.topology.adjacency[src].append(dst)
            if src not in self.topology.adjacency[dst]:
                self.topology.adjacency[dst].append(src)

        logger.debug(f"邻接图构建完成: {len(self.topology.adjacency)} 个节点有连接")

    def _classify_switch_tiers(self) -> None:
        """
        识别交换机层级

        在Fat-Tree拓扑中:
        - 叶子交换机(Leaf): 直接连接主机的交换机
        - 脊交换机(Spine): 连接叶子交换机的交换机
        - 核心交换机(Core): 最高层的交换机
        """
        switches = {
            guid: node for guid, node in self.topology.nodes.items()
            if node.node_type == NodeType.SWITCH
        }
        hosts = {
            guid for guid, node in self.topology.nodes.items()
            if node.node_type == NodeType.HOST
        }

        leaf_switches = set()
        spine_switches = set()
        core_switches = set()

        # 第一步: 找到连接主机的交换机 -> Leaf
        for sw_guid, sw_node in switches.items():
            neighbors = set(self.topology.adjacency.get(sw_guid, []))
            connected_hosts = neighbors & hosts
            if connected_hosts:
                leaf_switches.add(sw_guid)
                sw_node.tier = SwitchTier.LEAF

        # 第二步: 找到连接Leaf但不连接主机的交换机 -> Spine
        for sw_guid, sw_node in switches.items():
            if sw_guid in leaf_switches:
                continue
            neighbors = set(self.topology.adjacency.get(sw_guid, []))
            connected_leaves = neighbors & leaf_switches
            if connected_leaves:
                spine_switches.add(sw_guid)
                sw_node.tier = SwitchTier.SPINE

        # 第三步: 剩余的交换机 -> Core
        for sw_guid, sw_node in switches.items():
            if sw_guid not in leaf_switches and sw_guid not in spine_switches:
                core_switches.add(sw_guid)
                sw_node.tier = SwitchTier.CORE

        self.topology.switch_tiers = {
            "leaf": list(leaf_switches),
            "spine": list(spine_switches),
            "core": list(core_switches),
        }

        logger.info(
            f"交换机层级分类: Leaf={len(leaf_switches)}, "
            f"Spine={len(spine_switches)}, Core={len(core_switches)}"
        )

    def _identify_topology_type(self) -> None:
        """
        判断网络拓扑类型

        基于交换机层级结构和连接模式判断:
        - Fat-Tree: 明确的多层交换机层级，全对分带宽
        - Spine-Leaf: 两层结构（Leaf+Spine），每个Leaf连接所有Spine
        - 其他类型的简单启发式判断
        """
        tiers = self.topology.switch_tiers
        leaf_count = len(tiers.get("leaf", []))
        spine_count = len(tiers.get("spine", []))
        core_count = len(tiers.get("core", []))

        if leaf_count == 0 and spine_count == 0:
            self.topology.topology_type = TopologyType.UNKNOWN
            return

        # 检查是否为Spine-Leaf（2层结构）
        if leaf_count > 0 and spine_count > 0 and core_count == 0:
            # 验证每个Leaf是否连接到所有Spine
            all_spines = set(tiers.get("spine", []))
            is_full_mesh = True
            for leaf_guid in tiers.get("leaf", []):
                neighbors = set(self.topology.adjacency.get(leaf_guid, []))
                switch_neighbors = neighbors & all_spines
                if len(switch_neighbors) < len(all_spines) * 0.8:
                    is_full_mesh = False
                    break

            if is_full_mesh:
                self.topology.topology_type = TopologyType.SPINE_LEAF
            else:
                self.topology.topology_type = TopologyType.TREE
            return

        # 检查是否为Fat-Tree（3层结构）
        if leaf_count > 0 and spine_count > 0 and core_count > 0:
            self.topology.topology_type = TopologyType.FAT_TREE
            return

        # 默认
        self.topology.topology_type = TopologyType.UNKNOWN

    def get_shortest_path(self, src_guid: str, dst_guid: str) -> List[str]:
        """
        计算两个节点之间的最短路径（BFS）

        Args:
            src_guid: 源节点GUID
            dst_guid: 目标节点GUID

        Returns:
            路径上的节点GUID列表
        """
        if src_guid == dst_guid:
            return [src_guid]

        visited = {src_guid}
        queue = deque([(src_guid, [src_guid])])

        while queue:
            current, path = queue.popleft()
            for neighbor in self.topology.adjacency.get(current, []):
                if neighbor == dst_guid:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []  # 不可达

    def get_hop_count(self, src_guid: str, dst_guid: str) -> int:
        """
        计算两个节点之间的跳数

        Args:
            src_guid: 源节点GUID
            dst_guid: 目标节点GUID

        Returns:
            跳数，不可达返回-1
        """
        path = self.get_shortest_path(src_guid, dst_guid)
        if not path:
            return -1
        return len(path) - 1

    def get_all_pairs_distance(self) -> Dict[Tuple[str, str], int]:
        """
        计算所有主机对之间的跳数距离

        Returns:
            字典，key为(src, dst)元组，value为跳数
        """
        hosts = [
            guid for guid, node in self.topology.nodes.items()
            if node.node_type == NodeType.HOST
        ]

        distances = {}
        for i, src in enumerate(hosts):
            for dst in hosts[i + 1:]:
                hops = self.get_hop_count(src, dst)
                distances[(src, dst)] = hops
                distances[(dst, src)] = hops

        return distances

    def export_dot(self) -> str:
        """
        导出为DOT格式（Graphviz）

        Returns:
            DOT格式的拓扑图字符串
        """
        lines = ["graph cluster_topology {"]
        lines.append("    rankdir=TB;")
        lines.append("    node [shape=box];")
        lines.append("")

        # 节点定义
        for guid, node in self.topology.nodes.items():
            label = node.name or guid[:16]
            if node.node_type == NodeType.SWITCH:
                color = {
                    SwitchTier.LEAF: "lightblue",
                    SwitchTier.SPINE: "lightyellow",
                    SwitchTier.CORE: "lightcoral",
                }.get(node.tier, "lightgray")
                shape = "diamond"
            else:
                color = "lightgreen"
                shape = "box"

            safe_id = guid.replace("-", "_").replace(".", "_")
            lines.append(
                f'    "{safe_id}" [label="{label}", '
                f'shape={shape}, style=filled, fillcolor="{color}"];'
            )

        lines.append("")

        # 链路定义（去重）
        seen_links = set()
        for link in self.topology.links:
            pair = tuple(sorted([link.source_guid, link.dest_guid]))
            if pair not in seen_links:
                seen_links.add(pair)
                src_id = link.source_guid.replace("-", "_").replace(".", "_")
                dst_id = link.dest_guid.replace("-", "_").replace(".", "_")
                lines.append(f'    "{src_id}" -- "{dst_id}";')

        lines.append("}")
        return "\n".join(lines)

    def export_json(self) -> dict:
        """
        导出拓扑为JSON格式

        Returns:
            拓扑数据字典
        """
        nodes_data = []
        for guid, node in self.topology.nodes.items():
            nodes_data.append({
                "guid": guid,
                "name": node.name,
                "type": node.node_type.value,
                "tier": node.tier.value if node.node_type == NodeType.SWITCH else None,
                "num_ports": node.num_ports,
                "lid": node.lid,
            })

        links_data = []
        for link in self.topology.links:
            links_data.append({
                "source": link.source_guid,
                "source_port": link.source_port,
                "dest": link.dest_guid,
                "dest_port": link.dest_port,
                "speed": link.speed,
                "state": link.state,
            })

        return {
            "topology_type": self.topology.topology_type.value,
            "num_hosts": self.topology.num_hosts,
            "num_switches": self.topology.num_switches,
            "num_links": self.topology.num_links,
            "switch_tiers": self.topology.switch_tiers,
            "nodes": nodes_data,
            "links": links_data,
        }
