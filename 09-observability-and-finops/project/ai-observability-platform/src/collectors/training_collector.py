"""训练任务指标采集器"""

import time
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    job_name: str
    gpu_utilization_avg: float = 0
    samples_per_second: float = 0
    loss: float = 0
    epoch: int = 0
    eta_hours: float = 0
    gpu_memory_allocated_gb: float = 0
    timestamp: float = 0


class TrainingCollector:
    """训练任务指标采集 (PyTorch/DeepSpeed/Megatron)"""

    def __init__(self, prometheus_url: str = "http://prometheus:9090"):
        self.prometheus_url = prometheus_url

    def collect_mock(self) -> TrainingMetrics:
        import random
        return TrainingMetrics(
            job_name="pretrain-qwen-72b",
            gpu_utilization_avg=random.uniform(0.85, 0.98),
            samples_per_second=random.uniform(10, 20),
            loss=random.uniform(1.5, 2.5),
            epoch=3,
            eta_hours=random.uniform(10, 50),
            gpu_memory_allocated_gb=random.uniform(80, 90),
            timestamp=time.time(),
        )
