"""吞吐压测"""
import time
import argparse
import sys
sys.path.insert(0, "../..")

from src.core.engine import LLMEngine, EngineConfig
from src.core.sequence import SamplingParams


def benchmark_throughput(
    model: str = "gpt2",
    num_requests: int = 100,
    prompt_len: int = 128,
    max_tokens: int = 128,
):
    """吞吐压测"""
    print("=" * 60)
    print("  Throughput Benchmark")
    print("=" * 60)

    config = EngineConfig(model_name=model, max_model_len=512)
    engine = LLMEngine(config)
    engine.load_model()

    # 构造请求
    prompt = "Hello " * (prompt_len // 2)
    params = SamplingParams(max_tokens=max_tokens, temperature=0.7)

    # 提交所有请求
    print(f"\n  Submitting {num_requests} requests...")
    start = time.time()

    for i in range(num_requests):
        engine.add_request(f"req-{i}", prompt, params)

    # 运行
    completed = 0
    total_output_tokens = 0

    while engine.has_unfinished_requests():
        results = engine.step()
        for r in results:
            completed += 1
            total_output_tokens += r["output_len"]

    elapsed = time.time() - start

    print(f"\n  Results:")
    print(f"  Completed: {completed} requests")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Throughput: {completed/elapsed:.2f} req/s")
    print(f"  Throughput: {total_output_tokens/elapsed:.1f} tok/s")
    print(f"  Avg tokens/request: {total_output_tokens/max(completed,1):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--num-requests", type=int, default=50)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    benchmark_throughput(args.model, args.num_requests, args.prompt_len, args.max_tokens)
