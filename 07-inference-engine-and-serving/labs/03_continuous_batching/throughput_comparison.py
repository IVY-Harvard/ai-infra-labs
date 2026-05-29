"""
吞吐对比: Static vs Continuous Batching

直观展示两种方案在不同工作负载下的性能差异。
"""

import random
from typing import List
from dataclasses import dataclass

from static_batching import StaticBatchScheduler, Request as StaticRequest
from continuous_batching import ContinuousBatchScheduler, Request as ContRequest


def generate_requests(num: int, output_range: tuple, seed: int = 42):
    """生成请求 (返回两种格式)"""
    random.seed(seed)
    static_reqs = []
    cont_reqs = []
    for i in range(num):
        prompt_len = random.randint(50, 300)
        output_len = random.randint(*output_range)
        static_reqs.append(StaticRequest(i, prompt_len, output_len, 0.0))
        cont_reqs.append(ContRequest(i, prompt_len, output_len, 0.0))
    return static_reqs, cont_reqs


def run_comparison():
    """运行对比实验"""
    print("\n" + "=" * 70)
    print("  Throughput Comparison: Static vs Continuous Batching")
    print("=" * 70)

    batch_size = 32
    num_requests = 200

    # 测试不同的输出长度分布
    scenarios = [
        ("Uniform [50-100]", (50, 100)),      # 长度均匀 (差异小)
        ("Uniform [50-500]", (50, 500)),       # 中等差异
        ("Uniform [50-2000]", (50, 2000)),     # 大差异
        ("Skewed [10-100] + outliers", None),  # 大部分短 + 少数长
    ]

    print(f"\n  Config: batch_size={batch_size}, num_requests={num_requests}")
    print(f"\n  {'Scenario':<30} {'Static (tok/s)':<16} {'Contin (tok/s)':<16} {'Speedup':<10} {'Why'}")
    print(f"  {'-'*85}")

    for name, output_range in scenarios:
        if output_range is not None:
            static_reqs, cont_reqs = generate_requests(num_requests, output_range)
        else:
            # 特殊场景: 90% 短请求 + 10% 长请求
            random.seed(42)
            static_reqs = []
            cont_reqs = []
            for i in range(num_requests):
                prompt_len = random.randint(50, 200)
                if random.random() < 0.9:
                    output_len = random.randint(10, 100)
                else:
                    output_len = random.randint(500, 2000)
                static_reqs.append(StaticRequest(i, prompt_len, output_len, 0.0))
                cont_reqs.append(ContRequest(i, prompt_len, output_len, 0.0))

        # Run Static
        static = StaticBatchScheduler(batch_size=batch_size)
        s_stats = static.run(static_reqs)

        # Run Continuous
        contin = ContinuousBatchScheduler(max_batch_size=batch_size)
        c_stats = contin.run(cont_reqs)

        speedup = c_stats['throughput_tok_s'] / s_stats['throughput_tok_s'] if s_stats['throughput_tok_s'] > 0 else 0

        # 分析原因
        output_lens = [r.target_output_len for r in static_reqs]
        avg_len = sum(output_lens) / len(output_lens)
        max_len = max(output_lens)
        theoretical = max_len / avg_len

        reason = f"max/avg = {max_len}/{avg_len:.0f} = {theoretical:.1f}x"

        print(f"  {name:<30} {s_stats['throughput_tok_s']:<16.1f} {c_stats['throughput_tok_s']:<16.1f} {speedup:<10.2f}x {reason}")

    # 详细对比 (典型场景)
    print(f"\n\n  {'='*70}")
    print(f"  Detailed Comparison: Output Length [50-500]")
    print(f"  {'='*70}")

    static_reqs, cont_reqs = generate_requests(num_requests, (50, 500))

    static = StaticBatchScheduler(batch_size=32)
    s_stats = static.run(static_reqs)

    contin = ContinuousBatchScheduler(max_batch_size=32)
    c_stats = contin.run(cont_reqs)

    print(f"\n  {'Metric':<30} {'Static':<20} {'Continuous':<20}")
    print(f"  {'-'*70}")
    print(f"  {'Throughput (tok/s)':<30} {s_stats['throughput_tok_s']:<20.1f} {c_stats['throughput_tok_s']:<20.1f}")
    print(f"  {'Throughput (req/s)':<30} {s_stats['throughput_req_s']:<20.2f} {c_stats['throughput_req_s']:<20.2f}")
    print(f"  {'Avg Latency (ms)':<30} {s_stats['avg_latency_ms']:<20.0f} {c_stats['avg_latency_ms']:<20.0f}")
    print(f"  {'Avg TTFT (ms)':<30} {s_stats['avg_ttft_ms']:<20.0f} {c_stats['avg_ttft_ms']:<20.0f}")
    print(f"  {'GPU Utilization':<30} {s_stats['gpu_utilization']*100:<20.1f}% {c_stats['gpu_utilization']*100:<20.1f}%")
    print(f"  {'Total Time (ms)':<30} {s_stats['total_time_ms']:<20.0f} {c_stats['total_time_ms']:<20.0f}")

    # Batch size 影响
    print(f"\n\n  {'='*70}")
    print(f"  Batch Size Impact")
    print(f"  {'='*70}")
    print(f"\n  {'Batch Size':<12} {'Static tok/s':<15} {'Contin tok/s':<15} {'Speedup':<10}")
    print(f"  {'-'*52}")

    for bs in [4, 8, 16, 32, 64, 128]:
        static_reqs, cont_reqs = generate_requests(200, (50, 500))

        static = StaticBatchScheduler(batch_size=bs)
        s_stats = static.run(static_reqs)

        contin = ContinuousBatchScheduler(max_batch_size=bs)
        c_stats = contin.run(cont_reqs)

        speedup = c_stats['throughput_tok_s'] / s_stats['throughput_tok_s']
        print(f"  {bs:<12} {s_stats['throughput_tok_s']:<15.1f} {c_stats['throughput_tok_s']:<15.1f} {speedup:.2f}x")


if __name__ == "__main__":
    run_comparison()

    print("\n" + "=" * 70)
    print("  Key Findings:")
    print("  1. Speedup ≈ max_gen_len / avg_gen_len")
    print("  2. More variance in output length → more speedup")
    print("  3. Continuous Batching keeps GPU fully utilized")
    print("  4. Static Batching wastes GPU time waiting for slowest request")
    print("  5. Both scale with batch size, but Continuous scales better")
    print("=" * 70)
