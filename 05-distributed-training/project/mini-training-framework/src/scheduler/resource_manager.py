"""
GPU 资源管理器
===============
管理 GPU 资源分配、显存监控和负载均衡。
"""

import torch
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class GPUInfo:
    """单个 GPU 的信息"""
    index: int
    name: str
    total_memory_gb: float
    used_memory_gb: float = 0.0
    utilization_pct: float = 0.0


class ResourceManager:
    """
    GPU 资源管理器。
    职责:
      - 监控各 GPU 显存使用
      - 根据显存预算分配 batch size
      - 提供显存报警
    """

    def __init__(self, memory_limit_pct: float = 90.0):
        self.memory_limit_pct = memory_limit_pct
        self.num_gpus = torch.cuda.device_count()
        self.gpus: List[GPUInfo] = []
        self._init_gpu_info()

    def _init_gpu_info(self):
        """获取 GPU 基本信息"""
        for i in range(self.num_gpus):
            props = torch.cuda.get_device_properties(i)
            self.gpus.append(GPUInfo(
                index=i,
                name=props.name,
                total_memory_gb=props.total_mem / 1e9,
            ))

    def update_memory_stats(self):
        """更新所有 GPU 的显存使用情况"""
        for gpu in self.gpus:
            gpu.used_memory_gb = torch.cuda.memory_allocated(gpu.index) / 1e9

    def get_available_memory_gb(self, gpu_index: int = 0) -> float:
        """获取指定 GPU 的可用显存"""
        gpu = self.gpus[gpu_index]
        limit_gb = gpu.total_memory_gb * self.memory_limit_pct / 100
        used_gb = torch.cuda.memory_allocated(gpu_index) / 1e9
        return max(0, limit_gb - used_gb)

    def estimate_max_batch_size(
        self,
        model_memory_gb: float,
        per_sample_memory_gb: float,
        gpu_index: int = 0,
    ) -> int:
        """估算最大 batch size"""
        available = self.get_available_memory_gb(gpu_index)
        remaining = available - model_memory_gb
        if remaining <= 0:
            return 0
        return int(remaining / per_sample_memory_gb)

    def check_memory_pressure(self, gpu_index: int = 0) -> bool:
        """检查是否存在显存压力"""
        used_pct = (torch.cuda.memory_allocated(gpu_index) /
                    torch.cuda.get_device_properties(gpu_index).total_mem * 100)
        return used_pct > self.memory_limit_pct

    def summary(self) -> str:
        """生成资源摘要"""
        self.update_memory_stats()
        lines = ["GPU 资源状态:"]
        for gpu in self.gpus:
            used_pct = gpu.used_memory_gb / gpu.total_memory_gb * 100
            lines.append(
                f"  GPU {gpu.index}: {gpu.name} | "
                f"{gpu.used_memory_gb:.1f}/{gpu.total_memory_gb:.0f} GB ({used_pct:.0f}%)"
            )
        return "\n".join(lines)
