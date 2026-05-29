"""
vLLM vs TensorRT-LLM 性能对比

对比两个引擎在相同条件下的:
- 吞吐 (tokens/s)
- TTFT (首 token 延迟)
- TPOT (每 token 延迟)
"""

import time
import json
import subprocess
import argparse
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
import requests


def benchmark_vllm(
    base_url: str = "http://localhost:8000",
    num_requests: int = 100,
    prompt_len: int = 512,
    max_tokens: int = 256,
    concurrency: int = 32,
) -> Dict:
    """压测 vLLM"""
    print(f"\n  Benchmarking vLLM at {base_url}...")

    prompt = "Explain the concept of " + "artificial intelligence " * (prompt_len // 4)

    results = []

    def send_request(req_id: int):
        start = time.time()
        first_token_time = None
        tokens_received = 0

        try:
            response = requests.post(
                f"{base_url}/v1/completions",
                json={
                    "model": "llama",
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "stream": True,
                },
                stream=True,
                timeout=120,
            )

            for line in response.iter_lines():
                if line:
                    if first_token_time is None:
                        first_token_time = time.time()
                    tokens_received += 1

            end = time.time()
            return {
                "ttft": (first_token_time - start) if first_token_time else None,
                "total_time": end - start,
                "tokens": tokens_received,
                "success": True,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request, i) for i in range(num_requests)]
        results = [f.result() for f in futures]

    return _analyze_results(results, "vLLM")


def benchmark_trtllm(
    base_url: str = "http://localhost:8001",
    num_requests: int = 100,
    prompt_len: int = 512,
    max_tokens: int = 256,
    concurrency: int = 32,
) -> Dict:
    """压测 TensorRT-LLM (via Triton)"""
    print(f"\n  Benchmarking TRT-LLM at {base_url}...")

    # TRT-LLM via Triton 使用不同的 API 格式
    # 这里用通用的 HTTP 接口
    prompt = "Explain the concept of " + "artificial intelligence " * (prompt_len // 4)

    results = []

    def send_request(req_id: int):
        start = time.time()
        try:
            response = requests.post(
                f"{base_url}/v2/models/ensemble/generate",
                json={
                    "text_input": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                },
                timeout=120,
            )
            end = time.time()

            if response.status_code == 200:
                data = response.json()
                output_tokens = len(data.get("text_output", "").split())
                return {
                    "total_time": end - start,
                    "tokens": output_tokens,
                    "success": True,
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request, i) for i in range(num_requests)]
        results = [f.result() for f in futures]

    return _analyze_results(results, "TRT-LLM")


def _analyze_results(results: List[Dict], engine_name: str) -> Dict:
    """分析压测结果"""
    successful = [r for r in results if r.get("success")]
    failed = len(results) - len(successful)

    if not successful:
        return {"engine": engine_name, "error": "All requests failed"}

    total_tokens = sum(r.get("tokens", 0) for r in successful)
    total_time = max(r.get("total_time", 0) for r in successful)
    ttfts = [r["ttft"] for r in successful if r.get("ttft")]

    stats = {
        "engine": engine_name,
        "total_requests": len(results),
        "successful": len(successful),
        "failed": failed,
        "total_tokens": total_tokens,
        "throughput_tok_s": total_tokens / total_time if total_time > 0 else 0,
        "throughput_req_s": len(successful) / total_time if total_time > 0 else 0,
        "avg_latency_s": sum(r["total_time"] for r in successful) / len(successful),
        "p50_latency_s": sorted([r["total_time"] for r in successful])[len(successful) // 2],
        "p99_latency_s": sorted([r["total_time"] for r in successful])[int(len(successful) * 0.99)],
    }

    if ttfts:
        stats["avg_ttft_ms"] = sum(ttfts) / len(ttfts) * 1000
        stats["p50_ttft_ms"] = sorted(ttfts)[len(ttfts) // 2] * 1000

    return stats


def print_comparison(vllm_stats: Dict, trtllm_stats: Dict):
    """打印对比结果"""
    print("\n" + "=" * 70)
    print("  Performance Comparison: vLLM vs TensorRT-LLM")
    print("=" * 70)

    metrics = [
        ("Throughput (tok/s)", "throughput_tok_s", ".1f"),
        ("Throughput (req/s)", "throughput_req_s", ".2f"),
        ("Avg Latency (s)", "avg_latency_s", ".3f"),
        ("P50 Latency (s)", "p50_latency_s", ".3f"),
        ("P99 Latency (s)", "p99_latency_s", ".3f"),
        ("Avg TTFT (ms)", "avg_ttft_ms", ".1f"),
    ]

    print(f"\n  {'Metric':<25} {'vLLM':<15} {'TRT-LLM':<15} {'Winner':<10}")
    print(f"  {'-'*65}")

    for name, key, fmt in metrics:
        v = vllm_stats.get(key, 0)
        t = trtllm_stats.get(key, 0)

        if v and t:
            # 对于延迟，越小越好; 对于吞吐，越大越好
            if "latency" in key.lower() or "ttft" in key.lower():
                winner = "vLLM" if v < t else "TRT-LLM"
                ratio = v / t if t > 0 else 0
            else:
                winner = "vLLM" if v > t else "TRT-LLM"
                ratio = t / v if v > 0 else 0

            print(f"  {name:<25} {v:<15{fmt}} {t:<15{fmt}} {winner}")

    print(f"\n  Note: TRT-LLM typically 10-30% faster due to compile-time optimization")
    print(f"  Note: vLLM has better flexibility and faster iteration")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--trtllm-url", default="http://localhost:8001")
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    print("=" * 70)
    print("  vLLM vs TensorRT-LLM Benchmark")
    print("=" * 70)
    print(f"\n  Config: {args.num_requests} requests, concurrency={args.concurrency}")
    print(f"  Prompt: ~{args.prompt_len} tokens, Max output: {args.max_tokens} tokens")
    print(f"\n  Make sure both engines are running:")
    print(f"    vLLM:    {args.vllm_url}")
    print(f"    TRT-LLM: {args.trtllm_url}")

    # Run benchmarks
    vllm_stats = benchmark_vllm(
        args.vllm_url, args.num_requests, args.prompt_len,
        args.max_tokens, args.concurrency
    )
    trtllm_stats = benchmark_trtllm(
        args.trtllm_url, args.num_requests, args.prompt_len,
        args.max_tokens, args.concurrency
    )

    print_comparison(vllm_stats, trtllm_stats)
