"""
GPU 性能分析器 — NVML 封装

提供统一的 GPU 监控接口，支持：
- 实时状态查询（利用率、温度、功耗、频率）
- 进程监控
- 持续采集和记录
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False


@dataclass
class GPUStatus:
    """GPU 实时状态"""
    timestamp: str
    device_id: int
    name: str
    gpu_utilization: int  # %
    memory_utilization: int  # %
    memory_used_gb: float
    memory_total_gb: float
    temperature: int  # Celsius
    power_draw_w: float
    power_limit_w: float
    sm_clock_mhz: int
    memory_clock_mhz: int
    num_processes: int


@dataclass
class GPUProfile:
    """GPU 性能 Profile（一段时间的统计）"""
    device_id: int
    duration_s: float
    samples: int
    avg_gpu_util: float
    max_gpu_util: float
    avg_mem_util: float
    max_mem_used_gb: float
    avg_temperature: float
    max_temperature: int
    avg_power_w: float
    max_power_w: float


class GPUProfiler:
    """GPU 性能分析器"""

    def __init__(self):
        if not HAS_NVML:
            raise ImportError("pynvml 未安装: pip install pynvml")
        pynvml.nvmlInit()
        self.num_gpus = pynvml.nvmlDeviceGetCount()
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.num_gpus)]
        self._history: Dict[int, List[GPUStatus]] = {i: [] for i in range(self.num_gpus)}

    def __del__(self):
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    def get_status(self, device_id: int) -> GPUStatus:
        """获取单个 GPU 的当前状态"""
        handle = self.handles[device_id]

        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode('utf-8')

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu
            mem_util = util.memory
        except Exception:
            gpu_util = mem_util = -1

        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            temp = -1

        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
        except Exception:
            power = power_limit = -1

        try:
            sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        except Exception:
            sm_clock = mem_clock = -1

        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            num_procs = len(procs)
        except Exception:
            num_procs = -1

        return GPUStatus(
            timestamp=datetime.now().isoformat(),
            device_id=device_id,
            name=name,
            gpu_utilization=gpu_util,
            memory_utilization=mem_util,
            memory_used_gb=mem.used / (1024**3),
            memory_total_gb=mem.total / (1024**3),
            temperature=temp,
            power_draw_w=power,
            power_limit_w=power_limit,
            sm_clock_mhz=sm_clock,
            memory_clock_mhz=mem_clock,
            num_processes=num_procs,
        )

    def get_all_status(self) -> List[GPUStatus]:
        """获取所有 GPU 的状态"""
        return [self.get_status(i) for i in range(self.num_gpus)]

    def collect(self, duration_s: float, interval_s: float = 0.5) -> Dict[int, List[GPUStatus]]:
        """采集一段时间的 GPU 状态"""
        history = {i: [] for i in range(self.num_gpus)}
        start = time.time()

        while time.time() - start < duration_s:
            for i in range(self.num_gpus):
                status = self.get_status(i)
                history[i].append(status)
            time.sleep(interval_s)

        return history

    def profile(self, duration_s: float, interval_s: float = 0.5) -> List[GPUProfile]:
        """采集一段时间并生成统计 Profile"""
        history = self.collect(duration_s, interval_s)
        profiles = []

        for device_id, statuses in history.items():
            if not statuses:
                continue

            n = len(statuses)
            profiles.append(GPUProfile(
                device_id=device_id,
                duration_s=duration_s,
                samples=n,
                avg_gpu_util=sum(s.gpu_utilization for s in statuses) / n,
                max_gpu_util=max(s.gpu_utilization for s in statuses),
                avg_mem_util=sum(s.memory_utilization for s in statuses) / n,
                max_mem_used_gb=max(s.memory_used_gb for s in statuses),
                avg_temperature=sum(s.temperature for s in statuses) / n,
                max_temperature=max(s.temperature for s in statuses),
                avg_power_w=sum(s.power_draw_w for s in statuses) / n,
                max_power_w=max(s.power_draw_w for s in statuses),
            ))

        return profiles

    def to_dict(self, status: GPUStatus) -> dict:
        """将 GPUStatus 转为 dict（用于 JSON 序列化）"""
        return {
            'timestamp': status.timestamp,
            'device_id': status.device_id,
            'name': status.name,
            'gpu_utilization': status.gpu_utilization,
            'memory_utilization': status.memory_utilization,
            'memory_used_gb': round(status.memory_used_gb, 2),
            'memory_total_gb': round(status.memory_total_gb, 2),
            'temperature': status.temperature,
            'power_draw_w': round(status.power_draw_w, 1),
            'power_limit_w': round(status.power_limit_w, 1),
            'sm_clock_mhz': status.sm_clock_mhz,
            'memory_clock_mhz': status.memory_clock_mhz,
            'num_processes': status.num_processes,
        }


if __name__ == "__main__":
    profiler = GPUProfiler()

    print("当前 GPU 状态:")
    for status in profiler.get_all_status():
        print(f"  GPU {status.device_id}: {status.name}")
        print(f"    利用率: {status.gpu_utilization}%")
        print(f"    显存: {status.memory_used_gb:.1f}/{status.memory_total_gb:.0f} GB")
        print(f"    温度: {status.temperature}°C")
        print(f"    功耗: {status.power_draw_w:.0f}/{status.power_limit_w:.0f} W")
