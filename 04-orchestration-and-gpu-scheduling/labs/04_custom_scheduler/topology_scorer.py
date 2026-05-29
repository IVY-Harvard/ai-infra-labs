"""
GPU 拓扑感知调度打分器

根据节点的 GPU NVLink 拓扑和空闲 GPU 的连接关系，
为 Pod 的调度决策提供拓扑感知的打分。

目标：让请求多 GPU 的 Pod 尽量获得 NVLink 互连的 GPU 组合。

使用方式：
    python topology_scorer.py
    # 启动 HTTP server，供 K8s Scheduler Extender 调用
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)


@dataclass
class GPUDevice:
    """单个 GPU 设备信息"""
    index: int
    uuid: str
    allocated: bool = False
    nvlink_peers: list[int] = field(default_factory=list)  # NVLink 连接的对端 GPU index
    pcie_bus_id: str = ""
    numa_node: int = 0


@dataclass
class NodeGPUTopology:
    """节点的 GPU 拓扑信息"""
    node_name: str
    gpus: list[GPUDevice]
    nvlink_groups: list[list[int]]   # NVLink 互连组，如 [[0,1,2,3],[4,5,6,7]]
    pcie_switches: list[list[int]]   # PCIe Switch 分组
    numa_nodes: dict[int, list[int]] # NUMA 节点到 GPU 的映射

    @property
    def total_gpus(self) -> int:
        return len(self.gpus)

    @property
    def available_gpus(self) -> int:
        return sum(1 for g in self.gpus if not g.allocated)

    def get_available_indices(self) -> list[int]:
        return [g.index for g in self.gpus if not g.allocated]


# --- 模拟 8x H20 的 NVLink 拓扑 ---

def build_h20_topology(node_name: str, allocated_indices: list[int] = None) -> NodeGPUTopology:
    """
    构建 H20 的 GPU 拓扑。
    H20 的 NVLink 拓扑（简化模型）：
      - GPU 0-3 通过 NVLink 互连（NVSwitch 域 A）
      - GPU 4-7 通过 NVLink 互连（NVSwitch 域 B）
      - 两组之间通过 PCIe / Host Bridge 连接
    """
    allocated = set(allocated_indices or [])

    gpus = []
    for i in range(8):
        # NVLink peers：同组内的其他 GPU
        if i < 4:
            peers = [j for j in range(4) if j != i]
        else:
            peers = [j for j in range(4, 8) if j != i]

        gpus.append(GPUDevice(
            index=i,
            uuid=f"GPU-{node_name}-{i:04d}",
            allocated=(i in allocated),
            nvlink_peers=peers,
            pcie_bus_id=f"0000:{0x3b + i * 0x10:02x}:00.0",
            numa_node=0 if i < 4 else 1,
        ))

    return NodeGPUTopology(
        node_name=node_name,
        gpus=gpus,
        nvlink_groups=[[0, 1, 2, 3], [4, 5, 6, 7]],
        pcie_switches=[[0, 1], [2, 3], [4, 5], [6, 7]],
        numa_nodes={0: [0, 1, 2, 3], 1: [4, 5, 6, 7]},
    )


def find_best_gpu_combination(
    topology: NodeGPUTopology,
    num_gpus: int,
) -> tuple[list[int], int]:
    """
    在节点中找到最佳的 GPU 组合。

    优先级：
      1. 全部在同一 NVLink 组内 → 分数 100
      2. 全部在同一 NUMA 节点 → 分数 80
      3. 同一 PCIe Switch 下 → 分数 60
      4. 跨 NVLink 组 → 分数 30

    返回：(选中的 GPU indices, 拓扑分数)
    """
    available = topology.get_available_indices()
    if len(available) < num_gpus:
        return [], 0

    # 策略 1：尝试在同一 NVLink 组内找
    for group in topology.nvlink_groups:
        available_in_group = [i for i in group if i in available]
        if len(available_in_group) >= num_gpus:
            return available_in_group[:num_gpus], 100

    # 策略 2：尝试在同一 NUMA 节点内找
    for numa_id, gpu_indices in topology.numa_nodes.items():
        available_in_numa = [i for i in gpu_indices if i in available]
        if len(available_in_numa) >= num_gpus:
            return available_in_numa[:num_gpus], 80

    # 策略 3：尝试在同一 PCIe Switch 下找
    for switch_group in topology.pcie_switches:
        available_in_switch = [i for i in switch_group if i in available]
        if len(available_in_switch) >= num_gpus:
            return available_in_switch[:num_gpus], 60

    # 策略 4：跨组分配
    if len(available) >= num_gpus:
        return available[:num_gpus], 30

    return [], 0


def score_node(topology: NodeGPUTopology, requested_gpus: int) -> int:
    """
    为节点的 GPU 拓扑打分（0-100）。
    """
    if topology.available_gpus < requested_gpus:
        return 0

    _, topo_score = find_best_gpu_combination(topology, requested_gpus)
    return topo_score


# --- K8s Scheduler Extender HTTP API ---

# 模拟集群拓扑数据（实际应从 DCGM/Node Annotation 获取）
CLUSTER_TOPOLOGIES: dict[str, NodeGPUTopology] = {
    "gpu-node-0": build_h20_topology("gpu-node-0", allocated_indices=[0, 1]),
    "gpu-node-1": build_h20_topology("gpu-node-1", allocated_indices=[0, 1, 2, 3, 4]),
}


@app.route("/prioritize", methods=["POST"])
def prioritize():
    """
    Scheduler Extender 的 prioritize 接口。
    输入：ExtenderArgs（Pod + 候选节点列表）
    输出：HostPriorityList（每个节点的分数）
    """
    data = request.get_json()
    pod = data.get("pod", {})
    nodes = data.get("nodes", {}).get("items", [])

    # 获取 Pod 请求的 GPU 数量
    requested_gpus = _get_pod_gpu_request(pod)
    if requested_gpus == 0:
        # 非 GPU Pod，所有节点同分
        result = [{"host": n["metadata"]["name"], "score": 50} for n in nodes]
        return jsonify(result)

    logger.info(f"为 Pod {pod['metadata']['name']} 调度 {requested_gpus} GPU")

    result = []
    for node in nodes:
        node_name = node["metadata"]["name"]
        topology = CLUSTER_TOPOLOGIES.get(node_name)

        if topology is None:
            score = 0
        else:
            score = score_node(topology, requested_gpus)
            gpus, _ = find_best_gpu_combination(topology, requested_gpus)
            logger.info(f"  节点 {node_name}: 分数={score}, 推荐 GPU={gpus}")

        result.append({"host": node_name, "score": score})

    return jsonify(result)


@app.route("/filter", methods=["POST"])
def filter_nodes():
    """
    Scheduler Extender 的 filter 接口。
    过滤掉 GPU 拓扑不满足要求的节点。
    """
    data = request.get_json()
    pod = data.get("pod", {})
    nodes = data.get("nodes", {}).get("items", [])

    requested_gpus = _get_pod_gpu_request(pod)
    if requested_gpus == 0:
        return jsonify({"nodes": {"items": nodes}})

    filtered = []
    failed = {}
    for node in nodes:
        node_name = node["metadata"]["name"]
        topology = CLUSTER_TOPOLOGIES.get(node_name)

        if topology and topology.available_gpus >= requested_gpus:
            filtered.append(node)
        else:
            reason = f"可用 GPU 不足 (需要 {requested_gpus})"
            failed[node_name] = reason

    return jsonify({
        "nodes": {"items": filtered},
        "failedNodes": failed,
    })


def _get_pod_gpu_request(pod: dict) -> int:
    """从 Pod spec 中提取 GPU 请求数量"""
    total = 0
    for container in pod.get("spec", {}).get("containers", []):
        limits = container.get("resources", {}).get("limits", {})
        gpu_str = limits.get("nvidia.com/gpu", "0")
        total += int(gpu_str)
    return total


if __name__ == "__main__":
    logger.info("启动 GPU 拓扑感知调度器 Extender")
    logger.info(f"集群拓扑：{list(CLUSTER_TOPOLOGIES.keys())}")

    for name, topo in CLUSTER_TOPOLOGIES.items():
        logger.info(f"  {name}: {topo.total_gpus} GPU, {topo.available_gpus} 可用")

    app.run(host="0.0.0.0", port=8888, debug=False)
