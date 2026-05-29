"""
Continuous Batching 模拟

模拟连续批处理方式:
- 每步都可以加入新请求/移出完成的请求
- 不需要等待最慢的请求
- GPU 始终保持满载

与 Static Batching 对比，展示吞吐提升。
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional, Deque
from collections import deque


@dataclass
class Request:
    """请求"""
    id: int
    prompt_len: int
    target_output_len: int
    arrival_time: float

    start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    generated_tokens: int = 0
    is_prefilling: bool = False


@dataclass
class ContinuousBatchScheduler:
    """
    连续批处理调度器

    核心改变: Iteration-Level Scheduling
    - 每步 decode 后检查: 谁完成了？有新请求可以加入吗？
    - 完成的请求立即移出 → 释放 slot
    - 新请求立即加入 → 填满 slot
    """
    max_batch_size: int
    prefill_time_per_token_ms: float = 0.1
    decode_step_time_ms: float = 40.0
    # Chunked Prefill: 每步最多 Prefill 多少 tokens
    max_prefill_tokens_per_step: int = 512

    # 内部状态
    waiting_queue: Deque[Request] = field(default_factory=deque)
    running: List[Request] = field(default_factory=list)
    completed: List[Request] = field(default_factory=list)

    # 统计
    total_steps: int = 0
    total_active_tokens: int = 0  # 每步实际处理的 token 数

    def add_request(self, request: Request):
        self.waiting_queue.append(request)

    def _schedule_step(self, current_time: float) -> float:
        """执行一步调度"""
        # 1. 移出完成的请求
        still_running = []
        for req in self.running:
            if req.generated_tokens >= req.target_output_len:
                req.finish_time = current_time
                self.completed.append(req)
            else:
                still_running.append(req)
        self.running = still_running

        # 2. 加入新请求 (如果有空闲 slot)
        while self.waiting_queue and len(self.running) < self.max_batch_size:
            new_req = self.waiting_queue.popleft()
            new_req.start_time = current_time
            new_req.is_prefilling = True
            self.running.append(new_req)

        if not self.running:
            return current_time

        # 3. 执行一步
        step_time = self.decode_step_time_ms
        active_count = 0

        for req in self.running:
            if req.is_prefilling:
                # Prefill (简化: 假设一步完成)
                prefill_time = req.prompt_len * self.prefill_time_per_token_ms
                # 实际会用 Chunked Prefill, 这里简化
                req.first_token_time = current_time + max(step_time, prefill_time)
                req.is_prefilling = False
                req.generated_tokens += 1
                active_count += 1
            else:
                # Decode
                req.generated_tokens += 1
                active_count += 1

        self.total_steps += 1
        self.total_active_tokens += active_count
        current_time += step_time

        return current_time

    def run(self, requests: List[Request]) -> dict:
        """运行所有请求"""
        self.waiting_queue = deque()
        self.running = []
        self.completed = []
        self.total_steps = 0
        self.total_active_tokens = 0

        for req in requests:
            self.add_request(req)

        current_time = 0.0

        while self.running or self.waiting_queue:
            current_time = self._schedule_step(current_time)

        return self.get_stats()

    def get_stats(self) -> dict:
        if not self.completed:
            return {}

        total_time = max(r.finish_time for r in self.completed)

        ttfts = [(r.first_token_time - r.arrival_time) for r in self.completed]
        latencies = [(r.finish_time - r.arrival_time) for r in self.completed]
        tpots = [
            (r.finish_time - r.first_token_time) / max(r.generated_tokens - 1, 1)
            for r in self.completed
        ]

        total_output_tokens = sum(r.generated_tokens for r in self.completed)
        max_possible_tokens = self.total_steps * self.max_batch_size

        return {
            "total_time_ms": total_time,
            "throughput_req_s": len(self.completed) / (total_time / 1000),
            "throughput_tok_s": total_output_tokens / (total_time / 1000),
            "avg_ttft_ms": sum(ttfts) / len(ttfts),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "avg_tpot_ms": sum(tpots) / len(tpots),
            "gpu_utilization": self.total_active_tokens / max_possible_tokens
                if max_possible_tokens > 0 else 0,
            "total_steps": self.total_steps,
            "total_output_tokens": total_output_tokens,
            "num_requests": len(self.completed),
            "avg_batch_occupancy": self.total_active_tokens / self.total_steps
                if self.total_steps > 0 else 0,
        }


def generate_workload(
    num_requests: int,
    prompt_len_range: tuple = (50, 500),
    output_len_range: tuple = (50, 500),
    seed: int = 42,
) -> List[Request]:
    """生成模拟工作负载"""
    random.seed(seed)
    requests = []
    for i in range(num_requests):
        requests.append(Request(
            id=i,
            prompt_len=random.randint(*prompt_len_range),
            target_output_len=random.randint(*output_len_range),
            arrival_time=0.0,
        ))
    return requests


def run_demo():
    """运行 Continuous Batching 演示"""
    print("\n" + "=" * 70)
    print("  Continuous Batching Simulation")
    print("=" * 70)

    workload = generate_workload(100, output_len_range=(50, 500))

    avg_output = sum(r.target_output_len for r in workload) / len(workload)
    max_output = max(r.target_output_len for r in workload)
    print(f"\n  Workload: {len(workload)} requests")
    print(f"  Output lengths: avg={avg_output:.0f}, max={max_output}")

    for batch_size in [8, 16, 32, 64]:
        scheduler = ContinuousBatchScheduler(max_batch_size=batch_size)
        wl = [Request(r.id, r.prompt_len, r.target_output_len, r.arrival_time)
              for r in workload]
        stats = scheduler.run(wl)

        print(f"\n  Max Batch Size = {batch_size}:")
        print(f"    Throughput: {stats['throughput_tok_s']:.1f} tok/s, {stats['throughput_req_s']:.2f} req/s")
        print(f"    GPU Utilization: {stats['gpu_utilization']*100:.1f}%")
        print(f"    Avg Latency: {stats['avg_latency_ms']:.0f} ms")
        print(f"    Avg Batch Occupancy: {stats['avg_batch_occupancy']:.1f}")

    # 极端情况演示
    print(f"\n\n  {'='*60}")
    print(f"  Mixed Length Demo (same as static_batching.py)")
    print(f"  {'='*60}")

    extreme = [
        Request(0, 100, 500, 0.0),
        *[Request(i, 100, 50, 0.0) for i in range(1, 8)],
    ]

    scheduler = ContinuousBatchScheduler(max_batch_size=8)
    stats = scheduler.run(extreme)
    print(f"\n  1 long (500 tok) + 7 short (50 tok):")
    print(f"  GPU Utilization: {stats['gpu_utilization']*100:.1f}%")
    print(f"  Short requests finish early, slots freed immediately!")
    print(f"  (Compare with static: short requests must wait for the long one)")


if __name__ == "__main__":
    run_demo()
