"""
GPU 设备发现与拓扑采集

从 K8s 节点和 DCGM 采集 GPU 设备信息和拓扑结构。
"""

import logging
import subprocess
import json
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GPUDevice:
    """GPU 设备信息"""
    index: int
    uuid: str
    name: str
    memory_total_mb: int
    pcie_bus_id: str
    numa_node: int
    nvlink_peers: list[int] = field(default_factory=list)
    healthy: bool = True
    ecc_errors: int = 0
    temperature: float = 0.0
    power_usage_watts: float = 0.0


@dataclass
class NodeGPUs:
    """节点上的所有 GPU"""
    node_name: str
    gpus: list[GPUDevice]
    driver_version: str = ""
    cuda_version: str = ""

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def healthy_count(self) -> int:
        return sum(1 for g in self.gpus if g.healthy)

    @property
    def gpu_type(self) -> str:
        return self.gpus[0].name if self.gpus else "unknown"

    @property
    def gpu_memory_gb(self) -> int:
        if not self.gpus:
            return 0
        return self.gpus[0].memory_total_mb // 1024


class GPUDiscovery:
    """GPU 发现服务"""

    def __init__(self):
        self._node_gpus: dict[str, NodeGPUs] = {}

    def discover_local(self) -> Optional[NodeGPUs]:
        """发现本地节点的 GPU"""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,memory.total,pci.bus_id,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=10,
            )

            if result.returncode != 0:
                logger.error(f"nvidia-smi 失败: {result.stderr}")
                return None

            gpus = []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpus.append(GPUDevice(
                        index=int(parts[0]),
                        uuid=parts[1],
                        name=parts[2],
                        memory_total_mb=int(parts[3]),
                        pcie_bus_id=parts[4],
                        numa_node=self._get_numa_node(parts[4]),
                        temperature=float(parts[5]) if len(parts) > 5 else 0,
                        power_usage_watts=float(parts[6]) if len(parts) > 6 else 0,
                    ))

            # 发现 NVLink 拓扑
            self._discover_nvlink_topology(gpus)

            return NodeGPUs(
                node_name="localhost",
                gpus=gpus,
                driver_version=self._get_driver_version(),
            )

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"GPU 发现失败: {e}")
            return None

    def discover_from_k8s(self, node_name: str, annotations: dict) -> Optional[NodeGPUs]:
        """从 K8s 节点 Annotations 发现 GPU 信息"""
        gpu_count_str = annotations.get("gpu.topology/total-gpus", "0")
        gpu_count = int(gpu_count_str)

        if gpu_count == 0:
            return None

        gpu_type = annotations.get("nvidia.com/gpu.product", "unknown")
        gpu_memory = int(annotations.get("nvidia.com/gpu.memory", "0"))

        # 构建 GPU 列表
        gpus = []
        for i in range(gpu_count):
            gpus.append(GPUDevice(
                index=i,
                uuid=f"GPU-{node_name}-{i}",
                name=gpu_type,
                memory_total_mb=gpu_memory,
                pcie_bus_id=f"0000:{i:02x}:00.0",
                numa_node=0 if i < gpu_count // 2 else 1,
            ))

        # 解析 NVLink 拓扑 annotation
        nvlink_str = annotations.get("gpu.topology/nvlink-groups", "")
        if nvlink_str:
            try:
                nvlink_groups = json.loads(nvlink_str)
                for group in nvlink_groups:
                    for gpu_idx in group:
                        peers = [p for p in group if p != gpu_idx]
                        if gpu_idx < len(gpus):
                            gpus[gpu_idx].nvlink_peers = peers
            except json.JSONDecodeError:
                pass

        node_gpus = NodeGPUs(node_name=node_name, gpus=gpus)
        self._node_gpus[node_name] = node_gpus
        return node_gpus

    def _discover_nvlink_topology(self, gpus: list[GPUDevice]):
        """通过 nvidia-smi topo 发现 NVLink 连接"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return

            # 解析拓扑矩阵
            lines = result.stdout.strip().split("\n")
            for line in lines:
                # 查找 NV 连接（NV1, NV2, ... NV12 表示 NVLink 数量）
                if line.startswith("GPU"):
                    parts = line.split()
                    gpu_idx = int(parts[0].replace("GPU", ""))
                    for col_idx, val in enumerate(parts[1:]):
                        if val.startswith("NV"):
                            if gpu_idx < len(gpus):
                                gpus[gpu_idx].nvlink_peers.append(col_idx)
        except Exception:
            pass

    def _get_numa_node(self, pcie_bus_id: str) -> int:
        """根据 PCIe Bus ID 获取 NUMA 节点"""
        try:
            # Linux: /sys/bus/pci/devices/<bus_id>/numa_node
            path = f"/sys/bus/pci/devices/{pcie_bus_id}/numa_node"
            with open(path) as f:
                return int(f.read().strip())
        except (FileNotFoundError, ValueError):
            return 0

    def _get_driver_version(self) -> str:
        """获取 NVIDIA 驱动版本"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip().split("\n")[0]
        except Exception:
            return "unknown"

    def get_node_gpus(self, node_name: str) -> Optional[NodeGPUs]:
        """获取节点 GPU 信息"""
        return self._node_gpus.get(node_name)
