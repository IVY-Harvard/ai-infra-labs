"""
Static Batching 模拟

模拟传统的静态批处理方式:
- 收集一批请求
- 所有请求一起处理，等最慢的完成
- 然后处理下一批

演示"短板效应"导致的 GPU 浪费。
"""

import random
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Request:
    """请求"""
    id: int
    prompt_len: int
    target_output_len: int  # 目标生成长度
    arrival_time: float

    # 运行时状态
    start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    generated_tokens: int = 0


@dataclass
class StaticBatchScheduler:
    """
    静态批处理调度器

    行为:
    1. 等待收集 batch_size 个请求 (或超时)
    2. 所有请求一起 Prefill
    3. 所有请求一起 Decode，直到最慢的完成
    4. 全部返回后，处理下一批
    """
    batch_size: int
    prefill_time_per_token_ms: float = 0.1   # Prefill 每 token 耗时 (ms)
    decode_step_time_ms: float = 40.0        # 每步 Decode 耗时 (ms)
    max_wait_time_ms: float = 100.0          # 最大等待时间 (ms)

    # 统计
    completed_requests: List[Request] = field(default_factory=list)
    total_idle_steps: int = 0
    total_active_steps: int = 0

    def process_batch(self, batch: List[Request], current_time: float) -> float:
        """处理一个 batch"""
        if not batch:
            return current_time

        # Prefill: 取最长 prompt 的时间 (所有请求 padding 到最长)
        max_prompt_len = max(r.prompt_len for r in batch)
        prefill_time = max_prompt_len * self.prefill_time_per_token_ms

        for req in batch:
            req.start_time = current_time
            req.first_token_time = current_time + prefill_time

        current_time += prefill_time

        # Decode: 每步所有请求一起，直到最长的完成
        max_output_len = max(r.target_output_len for r in batch)

        for step in range(max_output_len):
            current_time += self.decode_step_time_ms

            active_in_step = 0
            for req in batch:
                if req.generated_tokens < req.target_output_len:
                    req.generated_tokens += 1
                    active_in_step += 1

                    if req.generated_tokens >= req.target_output_len:
                        req.finish_time = current_time

            # 统计 GPU 利用率
            self.total_active_steps += active_in_step
            self.total_idle_steps += (len(batch) - active_in_step)

        self.completed_requests.extend(batch)
        return current_time

    def run(self, requests: List[Request]) -> dict:
        """运行所有请求"""
        self.completed_requests = []
        self.total_idle_steps = 0
        self.total_active_steps = 0

        current_time = 0.0
        request_queue = list(requests)

        while request_queue:
            # 收集一个 batch
            batch = request_queue[:self.batch_size]
            request_queue = request_queue[self.batch_size:]

            current_time = self.process_batch(batch, current_time)

        return self.get_stats()

    def get_stats(self) -> dict:
        if not self.completed_requests:
            return {}

        total_time = max(r.finish_time for r in self.completed_requests)

        ttfts = [(r.first_token_time - r.arrival_time) for r in self.completed_requests]
        latencies = [(r.finish_time - r.arrival_time) for r in self.completed_requests]
        tpots = [
            (r.finish_time - r.first_token_time) / r.generated_tokens
            if r.generated_tokens > 0 else 0
            for r in self.completed_requests
        ]

        total_output_tokens = sum(r.generated_tokens for r in self.completed_requests)
        total_steps = self.total_active_steps + self.total_idle_steps

        return {
            "total_time_ms": total_time,
            "throughput_req_s": len(self.completed_requests) / (total_time / 1000),
            "throughput_tok_s": total_output_tokens / (total_time / 1000),
            "avg_ttft_ms": sum(ttfts) / len(ttfts),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "avg_tpot_ms": sum(tpots) / len(tpots),
            "gpu_utilization": self.total_active_steps / total_steps if total_steps > 0 else 0,
            "idle_slots": self.total_idle_steps,
            "total_slots": total_steps,
            "num_requests": len(self.completed_requests),
            "total_output_tokens": total_output_tokens,
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
            arrival_time=0.0,  # 假设所有请求同时到达
        ))
    return requests


def run_demo():
    """运行 Static Batching 演示"""
    print("\n" + "=" * 70)
    print("  Static Batching Simulation")
    print("=" * 70)

    workload = generate_workload(100, output_len_range=(50, 500))

    # 分析工作负载
    avg_output = sum(r.target_output_len for r in workload) / len(workload)
    max_output = max(r.target_output_len for r in workload)
    min_output = min(r.target_output_len for r in workload)
    print(f"\n  Workload: {len(workload)} requests")
    print(f"  Output lengths: avg={avg_output:.0f}, min={min_output}, max={max_output}")

    for batch_size in [8, 16, 32, 64]:
        scheduler = StaticBatchScheduler(batch_size=batch_size)
        # 深拷贝 workload
        wl = [Request(r.id, r.prompt_len, r.target_output_len, r.arrival_time)
              for r in workload]
        stats = scheduler.run(wl)

        print(f"\n  Batch Size = {batch_size}:")
        print(f"    Throughput: {stats['throughput_tok_s']:.1f} tok/s, {stats['throughput_req_s']:.2f} req/s")
        print(f"    GPU Utilization: {stats['gpu_utilization']*100:.1f}%")
        print(f"    Avg Latency: {stats['avg_latency_ms']:.0f} ms")
        print(f"    Idle slots: {stats['idle_slots']:,} / {stats['total_slots']:,}")

    # 演示短板效应
    print(f"\n\n  {'='*60}")
    print(f"  Short-tail Effect Demo")
    print(f"  {'='*60}")

    # 极端情况: 1 个长请求 + 7 个短请求
    extreme = [
        Request(0, 100, 500, 0.0),  # 长请求
        *[Request(i, 100, 50, 0.0) for i in range(1, 8)],  # 短请求
    ]

    scheduler = StaticBatchScheduler(batch_size=8)
    stats = scheduler.run(extreme)
    print(f"\n  1 long request (500 tokens) + 7 short requests (50 tokens)")
    print(f"  Batch Size = 8:")
    print(f"  GPU Utilization: {stats['gpu_utilization']*100:.1f}%")
    print(f"  Most GPUs idle after step 50, but must wait until step 500!")
    print(f"  Waste: {stats['idle_slots']:,} idle slots = {(1-stats['gpu_utilization'])*100:.1f}% wasted")


if __name__ == "__main__":
    run_demo()
