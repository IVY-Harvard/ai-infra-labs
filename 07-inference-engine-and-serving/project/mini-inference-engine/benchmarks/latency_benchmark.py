"""延迟压测"""
import time
import argparse
import sys
sys.path.insert(0, "../..")

from src.core.engine import LLMEngine, EngineConfig
from src.core.sequence import SamplingParams


def benchmark_latency(model: str = "gpt2", num_requests: int = 20, max_tokens: int = 64):
    """延迟压测 — 逐个请求测 TTFT 和 TPOT"""
    print("=" * 60)
    print("  Latency Benchmark")
    print("=" * 60)

    config = EngineConfig(model_name=model, max_model_len=512)
    engine = LLMEngine(config)
    engine.load_model()

    ttfts = []
    e2e_latencies = []

    for i in range(num_requests):
        prompt = f"Tell me about topic number {i}"
        params = SamplingParams(max_tokens=max_tokens, temperature=0.7)

        start = time.time()
        engine.add_request(f"req-{i}", prompt, params)

        first_step = True
        while engine.has_unfinished_requests():
            results = engine.step()
            if first_step:
                ttft = time.time() - start
                ttfts.append(ttft)
                first_step = False
            for r in results:
                if r["request_id"] == f"req-{i}":
                    e2e = time.time() - start
                    e2e_latencies.append(e2e)

    ttfts.sort()
    e2e_latencies.sort()

    print(f"\n  Results ({num_requests} requests, max_tokens={max_tokens}):")
    print(f"  TTFT  avg: {sum(ttfts)/len(ttfts)*1000:.1f} ms")
    print(f"  TTFT  p50: {ttfts[len(ttfts)//2]*1000:.1f} ms")
    print(f"  E2E   avg: {sum(e2e_latencies)/len(e2e_latencies)*1000:.1f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    benchmark_latency(args.model, args.num_requests, args.max_tokens)
