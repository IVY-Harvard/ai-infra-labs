"""Worker 进程管理"""
import torch
from typing import Optional


class Worker:
    """GPU Worker — 管理一个 GPU 上的推理"""

    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    def init_device(self):
        if torch.cuda.is_available():
            torch.cuda.set_device(self.gpu_id)

    def get_gpu_memory(self) -> dict:
        if not torch.cuda.is_available():
            return {"total": 0, "used": 0, "free": 0}
        total = torch.cuda.get_device_properties(self.gpu_id).total_mem
        used = torch.cuda.memory_allocated(self.gpu_id)
        return {"total": total, "used": used, "free": total - used}
