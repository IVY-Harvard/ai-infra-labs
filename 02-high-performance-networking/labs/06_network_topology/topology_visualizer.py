#!/usr/bin/env python3
"""GPU Topology Visualizer - 生成 GPU 集群拓扑可视化图"""

import subprocess
import argparse
import json
import re
from typing import Dict, List, Tuple, Optional


class TopologyVisualizer:
    """GPU 拓扑可视化器"""

    def __init__(self):
        self.gpus: List[Dict] = []
        self.nics: List[Dict] = []
        self.connections: List[Dict] = []
        self.numa_nodes: Dict[int, List[str]] = {}

    def collect_gpu_info(self):
        """收集 GPU 信息"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,pci.bus_id,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True
            )
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    self.gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "pci_bus": parts[2],
                        "memory": parts[3],
                    })
        except Exception as e:
            print(f"警告: 无法获取 GPU 信息: {e}")

    def collect_nic_info(self):
        """收集网络接口信息"""
        try:
            result = subprocess.run(["ibstat", "-l"], capture_output=True, text=True)
            for dev in result.stdout.strip().split("\n"):
                dev = dev.strip()
                if dev:
                    self.nics.append({"name": dev, "type": "InfiniBand"})
        except Exception:
            pass

        # 尝试获取 RDMA 设备的 PCIe 地址
        try:
            result = subprocess.run(
                ["ls", "-la", "/sys/class/infiniband/"],
                capture_output=True, text=True
            )
            for line in result.stdout.split("\n"):
                match = re.search(r"(mlx\w+)\s+->\s+.*?/(\w{4}:\w{2}:\w{2}\.\w)/", line)
                if match:
                    for nic in self.nics:
                        if nic["name"] == match.group(1):
                            nic["pci_bus"] = match.group(2)
        except Exception:
            pass

    def collect_numa_info(self):
        """收集 NUMA 拓扑信息"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "topo", "-m"], capture_output=True, text=True
            )
            # 解析 NUMA 亲和性
            for line in result.stdout.split("\n"):
                if line.startswith("GPU"):
                    parts = line.split()
                    gpu_id = parts[0]
                    # NUMA 信息通常在最后一列
                    if parts and parts[-1].isdigit():
                        numa = int(parts[-1])
                        self.numa_nodes.setdefault(numa, []).append(gpu_id)
        except Exception:
            pass

    def parse_topo_matrix(self) -> Dict[str, Dict[str, str]]:
        """解析拓扑连接矩阵"""
        topo = {}
        try:
            result = subprocess.run(
                ["nvidia-smi", "topo", "-m"], capture_output=True, text=True
            )
            lines = result.stdout.strip().split("\n")
            headers = []
            for line in lines:
                if re.match(r"\s*(GPU\d|mlx)", line):
                    if not headers:
                        headers = line.split()
                    else:
                        parts = line.split()
                        src = parts[0]
                        topo[src] = {}
                        for j, hdr in enumerate(headers):
                            if j + 1 < len(parts):
                                topo[src][hdr] = parts[j + 1]
        except Exception:
            pass
        return topo

    def generate_dot(self) -> str:
        """生成 Graphviz DOT 格式"""
        lines = ['digraph gpu_topology {', '  rankdir=TB;',
                 '  node [shape=box, style=filled];', '']

        # NUMA 节点子图
        for numa_id, devices in self.numa_nodes.items():
            lines.append(f'  subgraph cluster_numa{numa_id} {{')
            lines.append(f'    label="NUMA Node {numa_id}";')
            lines.append(f'    style=dashed; color=blue;')
            for dev in devices:
                lines.append(f'    {dev} [fillcolor=lightgreen];')
            lines.append('  }')
            lines.append('')

        # GPU 节点
        for gpu in self.gpus:
            label = f"GPU{gpu['index']}\\n{gpu['name']}\\n{gpu['memory']}"
            lines.append(f'  GPU{gpu["index"]} [label="{label}", fillcolor=lightgreen];')

        # NIC 节点
        for nic in self.nics:
            lines.append(f'  {nic["name"]} [label="{nic["name"]}\\n{nic["type"]}", fillcolor=lightyellow];')

        lines.append('')

        # 连接边
        topo = self.parse_topo_matrix()
        added_edges = set()
        for src, dsts in topo.items():
            for dst, link in dsts.items():
                if link in ("X", ""):
                    continue
                edge_key = tuple(sorted([src, dst]))
                if edge_key not in added_edges:
                    color = "red" if link.startswith("NV") else "black"
                    penwidth = "2.0" if link.startswith("NV") else "1.0"
                    lines.append(
                        f'  {src} -> {dst} [label="{link}", '
                        f'dir=both, color={color}, penwidth={penwidth}];'
                    )
                    added_edges.add(edge_key)

        lines.append('}')
        return '\n'.join(lines)

    def generate_ascii(self) -> str:
        """生成 ASCII 拓扑图"""
        output = []
        output.append("=" * 60)
        output.append("GPU 集群拓扑")
        output.append("=" * 60)

        for numa_id, devices in self.numa_nodes.items():
            output.append(f"\n┌─── NUMA Node {numa_id} {'─' * 40}┐")
            for dev in devices:
                gpu_info = next((g for g in self.gpus if f"GPU{g['index']}" == dev), None)
                if gpu_info:
                    output.append(f"│  [{dev}] {gpu_info['name']} ({gpu_info['memory']})")
                else:
                    output.append(f"│  [{dev}]")
            output.append(f"└{'─' * 55}┘")

        if not self.numa_nodes:
            for gpu in self.gpus:
                output.append(f"  [GPU{gpu['index']}] {gpu['name']} ({gpu['memory']})")

        if self.nics:
            output.append(f"\n网络设备:")
            for nic in self.nics:
                pci = nic.get('pci_bus', 'unknown')
                output.append(f"  [{nic['name']}] {nic['type']} @ {pci}")

        return '\n'.join(output)

    def visualize(self, fmt: str = "ascii") -> str:
        """生成拓扑可视化"""
        self.collect_gpu_info()
        self.collect_nic_info()
        self.collect_numa_info()

        if fmt == "dot":
            return self.generate_dot()
        return self.generate_ascii()


def main():
    parser = argparse.ArgumentParser(description="GPU 拓扑可视化工具")
    parser.add_argument("--format", choices=["ascii", "dot"], default="ascii",
                        help="输出格式 (默认: ascii)")
    parser.add_argument("--output", "-o", help="输出文件路径")
    parser.add_argument("--render", action="store_true",
                        help="使用 graphviz 渲染 (需要 dot 格式)")
    args = parser.parse_args()

    visualizer = TopologyVisualizer()
    result = visualizer.visualize(args.format)

    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
        print(f"[+] 拓扑图已保存: {args.output}")
        if args.render and args.format == "dot":
            png_path = args.output.replace(".dot", ".png")
            subprocess.run(["dot", "-Tpng", args.output, "-o", png_path])
            print(f"[+] 已渲染: {png_path}")
    else:
        print(result)


if __name__ == "__main__":
    main()
