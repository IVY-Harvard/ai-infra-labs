"""
吞吐量追踪器
==============
追踪训练吞吐量 (tokens/sec, samples/sec, MFU)。
"""

import time
from collections import deque
from typing import Optional

import torch
import torch.distributed as dist


class ThroughputTracker:
    """
    训练吞吐量追踪器。
    提供:
      - 实时 tokens/sec
      - 滑动窗口平均
      - MFU (Model FLOPs Utilization) 估算
    """

    def __init__(
        self,
        num_params: int,
        window_size: int = 50,
        gpu_tflops: float = 148.0,  # H20 BF16 TFLOPS
        num_gpus: int = 8,
    ):
        self.num_params = num_params
        self.window_size = window_size
        self.gpu_tflops = gpu_tflops
        self.num_gpus = num_gpus

        self._step_times = deque(maxlen=window_size)
        self._step_tokens = deque(maxlen=window_size)
        self._last_time = time.perf_counter()
        self._total_tokens = 0
        self._total_steps = 0
        self._start_time = time.perf_counter()

    def step(self, tokens: int):
        """记录一步的 token 数"""
        now = time.perf_counter()
        dt = now - self._last_time
        self._step_times.append(dt)
        self._step_tokens.append(tokens)
        self._last_time = now
        self._total_tokens += tokens
        self._total_steps += 1

    @property
    def tokens_per_sec(self) -> float:
        """当前 tokens/sec (滑动窗口平均)"""
        if not self._step_times:
            return 0.0
        total_time = sum(self._step_times)
        total_tokens = sum(self._step_tokens)
        if total_time == 0:
            return 0.0
        return total_tokens / total_time

    @property
    def samples_per_sec(self) -> float:
        """当前 samples/sec"""
        if not self._step_times:
            return 0.0
        total_time = sum(self._step_times)
        return len(self._step_times) / total_time

    @property
    def mfu(self) -> float:
        """
        Model FLOPs Utilization 估算。
        MFU = actual_flops / peak_flops

        actual_flops = 6 * num_params * tokens_per_sec (forward + backward)
        peak_flops = gpu_tflops * num_gpus * 1e12
        """
        tps = self.tokens_per_sec
        if tps == 0:
            return 0.0
        actual_flops = 6 * self.num_params * tps
        peak_flops = self.gpu_tflops * self.num_gpus * 1e12
        return actual_flops / peak_flops

    @property
    def avg_step_time_ms(self) -> float:
        """平均每步时间 (ms)"""
        if not self._step_times:
            return 0.0
        return sum(self._step_times) / len(self._step_times) * 1000

    def summary(self) -> str:
        """生成摘要"""
        elapsed = time.perf_counter() - self._start_time
        return (
            f"Throughput: {self.tokens_per_sec:.0f} tok/s | "
            f"MFU: {self.mfu*100:.1f}% | "
            f"Step: {self.avg_step_time_ms:.0f}ms | "
            f"Total: {self._total_tokens/1e6:.1f}M tokens in {elapsed:.0f}s"
        )
