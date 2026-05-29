"""
SGLang 部署与使用

SGLang 的核心优势: Radix Attention (自动前缀缓存)
特别适合多轮对话和共享 System Prompt 的场景。
"""

import subprocess
import requests
import time
import argparse


def start_sglang_server(
    model: str = "meta-llama/Llama-2-7b-chat-hf",
    tp_size: int = 1,
    port: int = 30000,
):
    """启动 SGLang 服务"""
    print("=" * 70)
    print("  Starting SGLang Server")
    print("=" * 70)

    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model,
        "--tp-size", str(tp_size),
        "--port", str(port),
        "--host", "0.0.0.0",
        # SGLang 自动启用 Radix Attention
    ]

    print(f"\n  Command: {' '.join(cmd)}")
    print(f"\n  Server will be available at http://localhost:{port}")
    print(f"  Radix Attention is enabled by default!")

    # 实际运行:
    # subprocess.run(cmd)

    return f"http://localhost:{port}"


def test_sglang_api(base_url: str = "http://localhost:30000"):
    """测试 SGLang API"""
    print("\n" + "=" * 70)
    print("  Testing SGLang API")
    print("=" * 70)

    # 基本生成
    print("\n  [1] Basic Generation:")
    response = requests.post(
        f"{base_url}/generate",
        json={
            "text": "What is machine learning?",
            "sampling_params": {
                "temperature": 0.7,
                "max_new_tokens": 100,
            }
        }
    )
    if response.status_code == 200:
        print(f"  Response: {response.json()['text'][:200]}...")

    # 多轮对话 (Radix Attention 自动缓存 System Prompt)
    print("\n  [2] Multi-turn Chat (Radix Attention caches shared prefix):")
    system_prompt = "You are a helpful AI assistant. " * 50  # 长 System Prompt

    for i, user_msg in enumerate(["Hello!", "What's 2+2?", "Tell me a joke"]):
        start = time.time()
        response = requests.post(
            f"{base_url}/generate",
            json={
                "text": f"{system_prompt}\nUser: {user_msg}\nAssistant:",
                "sampling_params": {"temperature": 0.7, "max_new_tokens": 100},
            }
        )
        elapsed = time.time() - start

        if response.status_code == 200:
            print(f"  Turn {i+1} ({user_msg}): {elapsed:.3f}s")
            if i == 0:
                print(f"    First turn: Prefill entire prompt (including system prompt)")
            else:
                print(f"    Subsequent: System prompt KV cached by Radix Attention!")


def benchmark_prefix_caching(base_url: str = "http://localhost:30000"):
    """测试 Radix Attention 的前缀缓存效果"""
    print("\n" + "=" * 70)
    print("  Benchmark: Radix Attention Prefix Caching")
    print("=" * 70)

    system_prompt = "You are an expert programmer. " * 100  # ~500 tokens

    # 第一个请求: 完整 Prefill
    start = time.time()
    requests.post(f"{base_url}/generate", json={
        "text": f"{system_prompt}\nUser: Write Hello World\nAssistant:",
        "sampling_params": {"max_new_tokens": 50},
    })
    first_time = time.time() - start

    # 后续请求: 共享前缀 → 跳过 System Prompt 的 Prefill
    times = []
    for i in range(10):
        start = time.time()
        requests.post(f"{base_url}/generate", json={
            "text": f"{system_prompt}\nUser: Question {i}\nAssistant:",
            "sampling_params": {"max_new_tokens": 50},
        })
        times.append(time.time() - start)

    avg_cached = sum(times) / len(times)
    print(f"\n  First request (full prefill):  {first_time:.3f}s")
    print(f"  Avg cached request:            {avg_cached:.3f}s")
    print(f"  Speedup from caching:          {first_time/avg_cached:.2f}x")
    print(f"\n  Radix Attention automatically cached the system prompt KV!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--mode", choices=["serve", "test", "bench"], default="test")
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}"

    if args.mode == "serve":
        start_sglang_server(args.model, args.tp, args.port)
    elif args.mode == "test":
        test_sglang_api(base_url)
    elif args.mode == "bench":
        benchmark_prefix_caching(base_url)
