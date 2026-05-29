"""
拓扑可视化模块

生成网络拓扑的可视化输出，支持DOT格式（Graphviz）和简单的
ASCII文本拓扑图。显示节点、交换机、链路健康状态，
用不同颜色标注状态。
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """可视化配置"""
    # DOT图形配置
    graph_direction: str = "TB"     # TB=top-to-bottom, LR=left-to-right
    node_font_size: int = 10
    edge_font_size: int = 8
    show_port_numbers: bool = True
    show_bandwidth: bool = True
    show_link_state: bool = True
    # 颜色配置
    host_color: str = "#27ae60"        # 主机节点颜色（绿色）
    leaf_switch_color: str = "#3498db"  # 叶子交换机（蓝色）
    spine_switch_color: str = "#f39c12" # 脊交换机（黄色）
    core_switch_color: str = "#e74c3c"  # 核心交换机（红色）
    healthy_link_color: str = "#27ae60"
    warning_link_color: str = "#f39c12"
    error_link_color: str = "#e74c3c"
    down_link_color: str = "#95a5a6"


@dataclass
class NodeVisual:
    """可视化节点"""
    node_id: str
    label: str
    node_type: str          # "host", "switch"
    tier: str = ""          # "leaf", "spine", "core"
    health: str = "healthy" # "healthy", "warning", "critical", "down"
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class LinkVisual:
    """可视化链路"""
    source_id: str
    dest_id: str
    source_port: int = 0
    dest_port: int = 0
    health: str = "healthy"
    bandwidth: str = ""
    label: str = ""


class TopologyVisualizer:
    """
    拓扑可视化器

    将拓扑数据转换为可视化输出格式。
    支持:
    - DOT格式（可用Graphviz渲染为SVG/PNG）
    - ASCII文本拓扑图
    - SVG内联图（嵌入HTML报告）
    """

    def __init__(self, config: Optional[dict] = None):
        """
        初始化可视化器

        Args:
            config: 可视化配置
        """
        viz_config = config or {}
        self.config = VisualizationConfig(
            graph_direction=viz_config.get("graph_direction", "TB"),
            show_port_numbers=viz_config.get("show_port_numbers", True),
            show_bandwidth=viz_config.get("show_bandwidth", True),
        )
        self.nodes: List[NodeVisual] = []
        self.links: List[LinkVisual] = []

    def load_topology(self, topology_data: dict,
                      health_data: Optional[dict] = None) -> None:
        """
        加载拓扑和健康数据

        Args:
            topology_data: 拓扑映射器导出的数据
            health_data: 链路健康检查数据（可选）
        """
        self.nodes = []
        self.links = []

        # 加载节点
        for node_data in topology_data.get("nodes", []):
            node = NodeVisual(
                node_id=node_data["guid"],
                label=node_data.get("name", node_data["guid"][:16]),
                node_type=node_data.get("type", "unknown"),
                tier=node_data.get("tier", ""),
            )
            self.nodes.append(node)

        # 加载链路
        for link_data in topology_data.get("links", []):
            link = LinkVisual(
                source_id=link_data["source"],
                dest_id=link_data["dest"],
                source_port=link_data.get("source_port", 0),
                dest_port=link_data.get("dest_port", 0),
                bandwidth=link_data.get("speed", ""),
            )
            self.links.append(link)

        # 应用健康状态
        if health_data:
            self._apply_health_data(health_data)

        logger.info(f"拓扑可视化数据加载完成: {len(self.nodes)} 节点, {len(self.links)} 链路")

    def _apply_health_data(self, health_data: dict) -> None:
        """将健康检查结果映射到拓扑节点和链路"""
        results = health_data.get("results", [])

        # 构建节点健康状态映射
        node_health: Dict[str, str] = {}
        for r in results:
            hostname = r.get("node_hostname", "")
            status = r.get("status", "unknown")
            # 取最严重的状态
            current = node_health.get(hostname, "healthy")
            severity_order = ["healthy", "warning", "degraded", "critical", "down"]
            if severity_order.index(status) > severity_order.index(current):
                node_health[hostname] = status

        # 应用到节点
        for node in self.nodes:
            if node.label in node_health:
                node.health = node_health[node.label]

    def _sanitize_id(self, raw_id: str) -> str:
        """将原始ID转换为DOT安全的ID"""
        return raw_id.replace("-", "_").replace(".", "_").replace(":", "_")

    def _get_node_color(self, node: NodeVisual) -> str:
        """根据节点类型和健康状态获取颜色"""
        if node.health in ("critical", "down"):
            return self.config.error_link_color

        if node.node_type == "host":
            return self.config.host_color
        elif node.node_type == "switch":
            tier_colors = {
                "leaf": self.config.leaf_switch_color,
                "spine": self.config.spine_switch_color,
                "core": self.config.core_switch_color,
            }
            return tier_colors.get(node.tier, "#95a5a6")

        return "#95a5a6"

    def _get_link_color(self, link: LinkVisual) -> str:
        """根据链路健康状态获取颜色"""
        color_map = {
            "healthy": self.config.healthy_link_color,
            "warning": self.config.warning_link_color,
            "critical": self.config.error_link_color,
            "down": self.config.down_link_color,
        }
        return color_map.get(link.health, "#2c3e50")

    def render_dot(self) -> str:
        """
        渲染为DOT格式

        Returns:
            DOT格式字符串
        """
        lines = [
            "digraph cluster_topology {",
            f"    rankdir={self.config.graph_direction};",
            "    concentrate=true;",
            f'    node [fontsize={self.config.node_font_size}];',
            f'    edge [fontsize={self.config.edge_font_size}];',
            "",
        ]

        # 按层级分组（使用subgraph控制布局）
        tier_groups: Dict[str, List[NodeVisual]] = defaultdict(list)
        host_nodes: List[NodeVisual] = []

        for node in self.nodes:
            if node.node_type == "switch":
                tier_groups[node.tier or "unknown"].append(node)
            else:
                host_nodes.append(node)

        # Core交换机子图
        if tier_groups.get("core"):
            lines.append("    subgraph cluster_core {")
            lines.append('        label="Core Layer";')
            lines.append('        style=dashed; color="#e74c3c";')
            for node in tier_groups["core"]:
                lines.append(self._dot_node(node))
            lines.append("    }")
            lines.append("")

        # Spine交换机子图
        if tier_groups.get("spine"):
            lines.append("    subgraph cluster_spine {")
            lines.append('        label="Spine Layer";')
            lines.append('        style=dashed; color="#f39c12";')
            for node in tier_groups["spine"]:
                lines.append(self._dot_node(node))
            lines.append("    }")
            lines.append("")

        # Leaf交换机子图
        if tier_groups.get("leaf"):
            lines.append("    subgraph cluster_leaf {")
            lines.append('        label="Leaf Layer";')
            lines.append('        style=dashed; color="#3498db";')
            for node in tier_groups["leaf"]:
                lines.append(self._dot_node(node))
            lines.append("    }")
            lines.append("")

        # 未分类交换机
        for tier_name in tier_groups:
            if tier_name not in ("core", "spine", "leaf"):
                for node in tier_groups[tier_name]:
                    lines.append(self._dot_node(node))

        # 主机节点
        if host_nodes:
            lines.append("    subgraph cluster_hosts {")
            lines.append('        label="Compute Nodes";')
            lines.append('        style=dashed; color="#27ae60";')
            for node in host_nodes:
                lines.append(self._dot_node(node))
            lines.append("    }")
            lines.append("")

        # 链路
        seen_links = set()
        for link in self.links:
            pair = tuple(sorted([link.source_id, link.dest_id]))
            if pair in seen_links:
                continue
            seen_links.add(pair)
            lines.append(self._dot_edge(link))

        lines.append("}")
        return "\n".join(lines)

    def _dot_node(self, node: NodeVisual) -> str:
        """生成单个DOT节点定义"""
        safe_id = self._sanitize_id(node.node_id)
        color = self._get_node_color(node)
        shape = "diamond" if node.node_type == "switch" else "box"
        style = "filled"
        if node.health in ("critical", "down"):
            style = "filled,bold"

        label = node.label
        if node.tier:
            label += f"\\n[{node.tier}]"

        return (
            f'        "{safe_id}" ['
            f'label="{label}", shape={shape}, style="{style}", '
            f'fillcolor="{color}", fontcolor="white"'
            f'];'
        )

    def _dot_edge(self, link: LinkVisual) -> str:
        """生成单个DOT边定义"""
        src_id = self._sanitize_id(link.source_id)
        dst_id = self._sanitize_id(link.dest_id)
        color = self._get_link_color(link)

        label_parts = []
        if self.config.show_port_numbers and (link.source_port or link.dest_port):
            label_parts.append(f"p{link.source_port}-p{link.dest_port}")
        if self.config.show_bandwidth and link.bandwidth:
            label_parts.append(link.bandwidth)
        label = "\\n".join(label_parts)

        penwidth = "1.5"
        if link.health == "critical":
            penwidth = "3.0"
        elif link.health == "down":
            penwidth = "1.0"

        return (
            f'    "{src_id}" -> "{dst_id}" ['
            f'label="{label}", color="{color}", penwidth={penwidth}, '
            f'dir=none'
            f'];'
        )

    def render_ascii(self, max_width: int = 120) -> str:
        """
        渲染为ASCII文本拓扑图

        简化的文本表示，适合终端输出和日志记录

        Args:
            max_width: 最大宽度

        Returns:
            ASCII拓扑图字符串
        """
        lines = []
        separator = "=" * min(max_width, 80)

        lines.append(separator)
        lines.append("  集群网络拓扑图")
        lines.append(separator)

        # 按层级分组
        core_switches = [n for n in self.nodes if n.node_type == "switch" and n.tier == "core"]
        spine_switches = [n for n in self.nodes if n.node_type == "switch" and n.tier == "spine"]
        leaf_switches = [n for n in self.nodes if n.node_type == "switch" and n.tier == "leaf"]
        unknown_switches = [n for n in self.nodes if n.node_type == "switch" and n.tier not in ("core", "spine", "leaf")]
        hosts = [n for n in self.nodes if n.node_type == "host"]

        # 绘制各层
        if core_switches:
            lines.append("")
            lines.append("  [Core Layer]")
            lines.append(self._ascii_node_row(core_switches, max_width))
            lines.append(self._ascii_connector_line(len(core_switches), max_width))

        if spine_switches:
            lines.append("")
            lines.append("  [Spine Layer]")
            lines.append(self._ascii_node_row(spine_switches, max_width))
            lines.append(self._ascii_connector_line(len(spine_switches), max_width))

        if leaf_switches:
            lines.append("")
            lines.append("  [Leaf Layer]")
            lines.append(self._ascii_node_row(leaf_switches, max_width))
            lines.append(self._ascii_connector_line(len(leaf_switches), max_width))

        if unknown_switches:
            lines.append("")
            lines.append("  [Switches]")
            lines.append(self._ascii_node_row(unknown_switches, max_width))
            lines.append(self._ascii_connector_line(len(unknown_switches), max_width))

        if hosts:
            lines.append("")
            lines.append("  [Compute Nodes]")
            # 主机可能很多，每行显示一部分
            hosts_per_row = max(1, max_width // 18)
            for i in range(0, len(hosts), hosts_per_row):
                batch = hosts[i:i + hosts_per_row]
                lines.append(self._ascii_node_row(batch, max_width))

        lines.append("")
        lines.append(separator)

        # 统计摘要
        health_counts = defaultdict(int)
        for node in self.nodes:
            health_counts[node.health] += 1

        lines.append(f"  总计: {len(self.nodes)} 节点, {len(self.links)} 链路")
        lines.append(
            f"  健康: {health_counts.get('healthy', 0)} | "
            f"告警: {health_counts.get('warning', 0)} | "
            f"严重: {health_counts.get('critical', 0)} | "
            f"断开: {health_counts.get('down', 0)}"
        )
        lines.append(separator)

        return "\n".join(lines)

    def _ascii_node_row(self, nodes: List[NodeVisual], max_width: int) -> str:
        """生成一行ASCII节点"""
        if not nodes:
            return ""

        node_strs = []
        for node in nodes:
            # 健康状态符号
            health_symbol = {
                "healthy": "+",
                "warning": "!",
                "critical": "X",
                "down": "-",
            }.get(node.health, "?")

            name = node.label[:12]
            node_str = f"[{health_symbol}{name}]"
            node_strs.append(node_str)

        # 均匀分布
        total_len = sum(len(s) for s in node_strs)
        if total_len > max_width:
            return "  " + " ".join(node_strs)

        spacing = max(2, (max_width - total_len) // (len(node_strs) + 1))
        padded = (" " * spacing).join(node_strs)
        return "  " + padded

    def _ascii_connector_line(self, node_count: int, max_width: int) -> str:
        """生成层间连接线"""
        if node_count <= 1:
            return "       |"
        connector_width = min(max_width - 4, node_count * 15)
        mid = connector_width // 2
        line = " " * 4
        for i in range(connector_width):
            if i == 0 or i == connector_width - 1:
                line += "|"
            elif i == mid:
                line += "|"
            elif i % (connector_width // max(node_count - 1, 1)) == 0:
                line += "|"
            else:
                line += "-"
        return line

    def render_summary_table(self) -> str:
        """
        生成节点状态摘要表

        Returns:
            格式化的文本表格
        """
        lines = []
        header = f"{'节点名称':<20} {'类型':<10} {'层级':<8} {'健康状态':<10}"
        lines.append(header)
        lines.append("-" * len(header))

        for node in sorted(self.nodes, key=lambda n: (n.node_type, n.tier, n.label)):
            status_icon = {
                "healthy": "[OK]",
                "warning": "[!!]",
                "critical": "[XX]",
                "down": "[--]",
            }.get(node.health, "[??]")

            lines.append(
                f"{node.label:<20} {node.node_type:<10} "
                f"{node.tier or '-':<8} {status_icon:<10}"
            )

        return "\n".join(lines)

    def save_dot(self, output_path: str) -> None:
        """保存DOT文件"""
        dot_content = self.render_dot()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(dot_content)
        logger.info(f"DOT拓扑图已保存: {output_path}")

    def save_ascii(self, output_path: str) -> None:
        """保存ASCII拓扑图"""
        ascii_content = self.render_ascii()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(ascii_content)
        logger.info(f"ASCII拓扑图已保存: {output_path}")
