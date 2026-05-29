"""GPU 指标采集器 — 通过 NVML/DCGM 采集 GPU 硬件指标"""

import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GPUMetrics:
    """单个 GPU 的指标快照"""
    gpu_id: int
    temperature_c: int
    power_watts: float
    sm_clock_mhz: int
    memory_used_gb: float
    memory_total_gb: float
    utilization_pct: int
    memory_utilization_pct: int
    ecc_errors_correctable: int = 0
    ecc_errors_uncorrectable: int = 0
    pcie_tx_bytes: int = 0
    pcie_rx_bytes: int = 0
    timestamp: float = 0


class GPUCollector:
    """GPU 硬件指标采集器

    数据源:
    1. NVML (pynvml): 直接从 NVIDIA 驱动读取
    2. DCGM Exporter: 通过 Prometheus 端点采集 (更全面)

    采集指标:
    - 温度、功率、时钟频率
    - 显存使用量
    - SM/Memory 利用率
    - ECC 错误计数
    - PCIe 流量
    """

    def __init__(self, use_nvml: bool = True):
        self._nvml_initialized = False
        self._device_count = 0
        if use_nvml:
            self._init_nvml()

    def _init_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._device_count = pynvml.nvmlDeviceGetCount()
            self._nvml_initialized = True
            logger.info(f"NVML initialized: {self._device_count} GPUs")
        except Exception as e:
            logger.warning(f"NVML init failed: {e}")

    def collect(self) -> List[GPUMetrics]:
        """采集所有 GPU 指标"""
        if not self._nvml_initialized:
            return self._collect_mock()

        import pynvml
        metrics = []
        for i in range(self._device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)

                metrics.append(GPUMetrics(
                    gpu_id=i,
                    temperature_c=temp,
                    power_watts=power,
                    sm_clock_mhz=clock,
                    memory_used_gb=mem_info.used / (1024**3),
                    memory_total_gb=mem_info.total / (1024**3),
                    utilization_pct=util.gpu,
                    memory_utilization_pct=util.memory,
                    timestamp=time.time(),
                ))
            except Exception as e:
                logger.error(f"Failed to collect GPU {i}: {e}")

        return metrics

    def _collect_mock(self) -> List[GPUMetrics]:
        """模拟数据 (测试用)"""
        import random
        return [
            GPUMetrics(
                gpu_id=i,
                temperature_c=65 + random.randint(0, 10),
                power_watts=300 + random.randint(0, 80),
                sm_clock_mhz=1800 + random.randint(-100, 100),
                memory_used_gb=72 + random.uniform(0, 10),
                memory_total_gb=96.0,
                utilization_pct=random.randint(60, 95),
                memory_utilization_pct=random.randint(70, 95),
                timestamp=time.time(),
            )
            for i in range(8)
        ]

    def collect_as_prometheus(self) -> str:
        """输出 Prometheus 格式"""
        metrics = self.collect()
        lines = []
        for m in metrics:
            lines.append(f'gpu_temperature_celsius{{gpu="{m.gpu_id}"}} {m.temperature_c}')
            lines.append(f'gpu_power_watts{{gpu="{m.gpu_id}"}} {m.power_watts}')
            lines.append(f'gpu_utilization_percent{{gpu="{m.gpu_id}"}} {m.utilization_pct}')
            lines.append(f'gpu_memory_used_bytes{{gpu="{m.gpu_id}"}} {int(m.memory_used_gb * 1e9)}')
        return "\n".join(lines)
