"""速度对比: 不同量化方案的推理性能"""
import argparse

def benchmark_speed():
    """对比不同量化方案的推理速度"""
    print("=" * 70)
    print("  Speed Benchmark: Quantization Performance")
    print("=" * 70)
    print(f"\n  Expected results (LLaMA-2-70B, 8×H20, batch=32):")
    print(f"\n  {'Method':<15} {'Model Size':<12} {'TPOT(ms)':<10} {'Throughput':<14} {'Max Batch'}")
    print(f"  {'-'*61}")
    print(f"  {'FP16':<15} {'140 GB':<12} {'48':<10} {'667 tok/s':<14} {'60'}")
    print(f"  {'FP8':<15} {'70 GB':<12} {'30':<10} {'1067 tok/s':<14} {'120'}")
    print(f"  {'AWQ-W4':<15} {'37 GB':<12} {'25':<10} {'1280 tok/s':<14} {'200+'}")
    print(f"  {'GPTQ-W4':<15} {'37 GB':<12} {'25':<10} {'1280 tok/s':<14} {'200+'}")
    print(f"\n  Key insight: FP8 is the sweet spot for H20")
    print(f"  - Almost no quality loss")
    print(f"  - 1.6x speedup")
    print(f"  - 2x batch size")
    print(f"  - Native hardware support")
    print(f"\n  Run actual benchmark:")
    print(f"  # Terminal 1: start server")
    print(f"  vllm serve model --quantization fp8 --kv-cache-dtype fp8 --tp 8")
    print(f"  # Terminal 2: run benchmark")
    print(f"  python -m vllm.entrypoints.openai.api_server --benchmark")

if __name__ == "__main__":
    benchmark_speed()
