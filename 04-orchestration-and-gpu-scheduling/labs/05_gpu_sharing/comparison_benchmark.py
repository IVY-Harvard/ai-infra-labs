"""
GPU 共享方式性能基准测试

对比 3 种 GPU 共享方式的性能特征：
1. 独占模式（Baseline）
2. MPS 共享
3. Time-Slicing 共享

测试指标：
  - 推理延迟（P50/P95/P99）
  - 吞吐量（requests/sec）
  - GPU 利用率
  - 显存使用效率

使用方式：
    pip install torch numpy
    python comparison_benchmark.py
"""

import time
import statistics
import subprocess
import threading
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

# 注意：实际运行需要 PyTorch + CUDA 环境
# 这里用模拟数据演示基准测试框架

@dataclass
class BenchmarkResult:
    """基准测试结果"""
    mode: str
    num_clients: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_rps: float
    gpu_util_percent: float
    memory_used_gb: float
    notes: str = ""


@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    model_name: str = "resnet50"
    batch_size: int = 32
    num_requests: int = 1000
    num_clients: int = 4           # 共享客户端数量
    warmup_requests: int = 50
    input_shape: tuple = (3, 224, 224)


def simulate_inference_latency(mode: str, num_clients: int) -> list[float]:
    """
    模拟不同共享模式下的推理延迟（毫秒）。

    真实测试应替换为实际的模型推理调用。
    这里用经验数据模拟：
      - 独占：~5ms baseline
      - MPS (4 clients)：~7ms (+40% overhead)
      - Time-Slicing (4 clients)：~15ms (+200% overhead)
    """
    import random

    base_latency = 5.0  # ms

    if mode == "exclusive":
        # 独占模式：最低延迟
        latencies = [base_latency + random.gauss(0, 0.5) for _ in range(1000)]
    elif mode == "mps":
        # MPS：中等开销，随客户端数增加
        overhead = 1.0 + 0.1 * num_clients
        latencies = [base_latency * overhead + random.gauss(0, 1.0) for _ in range(1000)]
    elif mode == "time_slicing":
        # Time-Slicing：高开销，上下文切换
        overhead = 1.0 + 0.5 * num_clients
        jitter = 2.0 * num_clients  # 时间片抖动更大
        latencies = [base_latency * overhead + abs(random.gauss(0, jitter)) for _ in range(1000)]
    else:
        latencies = [base_latency for _ in range(1000)]

    return [max(0.1, l) for l in latencies]


def run_benchmark(mode: str, config: BenchmarkConfig) -> BenchmarkResult:
    """运行单个模式的基准测试"""
    print(f"\n{'='*60}")
    print(f"测试模式: {mode} ({config.num_clients} clients)")
    print(f"{'='*60}")

    # 模拟推理
    latencies = simulate_inference_latency(mode, config.num_clients)

    # 计算统计量
    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[int(len(latencies_sorted) * 0.50)]
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
    p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]

    # 吞吐量 = 请求数 / 总时间
    total_time_s = sum(latencies) / 1000.0 / config.num_clients  # 并行
    throughput = len(latencies) / total_time_s

    # GPU 利用率和显存（模拟）
    if mode == "exclusive":
        gpu_util = 30.0  # 推理时 GPU 利用率通常不高
        mem_used = 10.0
    elif mode == "mps":
        gpu_util = 30.0 * config.num_clients * 0.8  # MPS 可以并行
        mem_used = 10.0 * config.num_clients
    else:  # time_slicing
        gpu_util = 30.0 * min(config.num_clients, 2)  # 时间片不能真并行
        mem_used = 10.0 * config.num_clients  # 所有客户端的模型都在显存

    result = BenchmarkResult(
        mode=mode,
        num_clients=config.num_clients,
        latency_p50_ms=round(p50, 2),
        latency_p95_ms=round(p95, 2),
        latency_p99_ms=round(p99, 2),
        throughput_rps=round(throughput, 1),
        gpu_util_percent=min(100.0, round(gpu_util, 1)),
        memory_used_gb=round(mem_used, 1),
    )

    print(f"  延迟 P50: {result.latency_p50_ms} ms")
    print(f"  延迟 P95: {result.latency_p95_ms} ms")
    print(f"  延迟 P99: {result.latency_p99_ms} ms")
    print(f"  吞吐量:   {result.throughput_rps} req/s")
    print(f"  GPU 利用率: {result.gpu_util_percent}%")
    print(f"  显存使用:  {result.memory_used_gb} GB")

    return result


def print_comparison_table(results: list[BenchmarkResult]):
    """打印对比表格"""
    print("\n")
    print("=" * 80)
    print("GPU 共享方式性能对比 (H20, ResNet50, batch=32)")
    print("=" * 80)
    print(f"{'模式':<15} {'客户端':<8} {'P50(ms)':<10} {'P95(ms)':<10} "
          f"{'P99(ms)':<10} {'吞吐(r/s)':<12} {'GPU%':<8} {'显存(GB)':<10}")
    print("-" * 80)

    for r in results:
        print(f"{r.mode:<15} {r.num_clients:<8} {r.latency_p50_ms:<10} "
              f"{r.latency_p95_ms:<10} {r.latency_p99_ms:<10} "
              f"{r.throughput_rps:<12} {r.gpu_util_percent:<8} {r.memory_used_gb:<10}")

    print("-" * 80)
    print("\n分析：")
    print("  - 独占模式延迟最低，但 GPU 利用率低（30%），显存浪费大")
    print("  - MPS 模式延迟增加有限（~40%），但可以真正并行利用 SM")
    print("  - Time-Slicing 延迟增加明显（~200%），但配置最简单")
    print("\n建议：")
    print("  - 对延迟敏感的在线推理 → MPS")
    print("  - 开发环境/离线批处理 → Time-Slicing")
    print("  - 训练/大模型推理 → 独占")


def main():
    config = BenchmarkConfig(num_clients=4)
    results = []

    # 测试 1：独占模式（Baseline）
    config_exclusive = BenchmarkConfig(num_clients=1)
    results.append(run_benchmark("exclusive", config_exclusive))

    # 测试 2：MPS 共享 (4 clients)
    results.append(run_benchmark("mps", config))

    # 测试 3：Time-Slicing (4 clients)
    results.append(run_benchmark("time_slicing", config))

    # 对比表格
    print_comparison_table(results)

    print("\n" + "=" * 80)
    print("实际测试命令（需要真实 GPU 环境）：")
    print("=" * 80)
    print("""
# 1. 独占模式测试
python -c "
import torch, time
model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True).cuda().eval()
x = torch.randn(32, 3, 224, 224).cuda()
# warmup
for _ in range(50): model(x)
# benchmark
torch.cuda.synchronize()
start = time.time()
for _ in range(1000): model(x)
torch.cuda.synchronize()
print(f'Avg latency: {(time.time()-start)/1000*1000:.2f} ms')
"

# 2. MPS 模式测试
# 先启动 MPS server，然后并行运行多个推理进程
nvidia-cuda-mps-control -d
for i in $(seq 1 4); do
  python benchmark_inference.py --client-id=$i &
done
wait
echo quit | nvidia-cuda-mps-control

# 3. Time-Slicing 测试
# 配置 Device Plugin replicas=4，然后提交 4 个 Pod
kubectl apply -f time_slicing_benchmark_pods.yaml
""")


if __name__ == "__main__":
    main()
