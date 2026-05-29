"""网络指标采集器 — NVLink/PCIe/InfiniBand"""

import time
import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class NetworkMetrics:
    node: str
    nvlink_tx_bytes_per_s: float = 0
    nvlink_rx_bytes_per_s: float = 0
    pcie_tx_bytes_per_s: float = 0
    pcie_rx_bytes_per_s: float = 0
    nvlink_crc_errors: int = 0
    ib_port_xmit_data: int = 0
    ib_port_rcv_data: int = 0
    timestamp: float = 0


class NetworkCollector:
    """GPU 互联网络指标采集"""

    def __init__(self, prometheus_url: str = "http://prometheus:9090"):
        self.prometheus_url = prometheus_url

    def collect_mock(self) -> List[NetworkMetrics]:
        import random
        return [
            NetworkMetrics(
                node=f"gpu-node-{i}",
                nvlink_tx_bytes_per_s=random.uniform(100e9, 400e9),
                nvlink_rx_bytes_per_s=random.uniform(100e9, 400e9),
                pcie_tx_bytes_per_s=random.uniform(5e9, 30e9),
                pcie_rx_bytes_per_s=random.uniform(5e9, 30e9),
                nvlink_crc_errors=random.randint(0, 2),
                timestamp=time.time(),
            )
            for i in range(4)
        ]
