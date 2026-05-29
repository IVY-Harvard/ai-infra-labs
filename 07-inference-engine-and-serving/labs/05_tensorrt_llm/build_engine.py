"""
构建 TensorRT-LLM Engine

将 HuggingFace 模型转换为 TensorRT-LLM 优化 engine。
编译时优化: 算子融合、精度选择、内存规划。
"""

import os
import subprocess
import argparse
from pathlib import Path


def build_trtllm_engine(
    model_path: str,
    output_dir: str,
    tp_size: int = 8,
    dtype: str = "float16",
    max_batch_size: int = 64,
    max_input_len: int = 4096,
    max_output_len: int = 2048,
    quantization: str = None,
):
    """
    构建 TensorRT-LLM engine

    步骤:
    1. 转换 HF 模型权重为 TRT-LLM 格式
    2. 编译优化 engine (算子融合、精度优化等)
    3. 保存序列化的 engine 文件
    """

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Convert checkpoint
    print(f"[Step 1] Converting checkpoint from {model_path}...")
    convert_cmd = [
        "python", "-m", "tensorrt_llm.commands.convert_checkpoint",
        f"--model_dir={model_path}",
        f"--output_dir={output_dir}/checkpoint",
        f"--tp_size={tp_size}",
        f"--dtype={dtype}",
    ]
    if quantization == "fp8":
        convert_cmd.extend(["--use_fp8", "--fp8_kv_cache"])
    elif quantization == "int8":
        convert_cmd.extend(["--use_smooth_quant"])

    print(f"  Command: {' '.join(convert_cmd)}")
    # subprocess.run(convert_cmd, check=True)  # 取消注释以实际运行

    # Step 2: Build engine
    print(f"\n[Step 2] Building TRT-LLM engine...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={output_dir}/checkpoint",
        f"--output_dir={output_dir}/engine",
        f"--max_batch_size={max_batch_size}",
        f"--max_input_len={max_input_len}",
        f"--max_seq_len={max_input_len + max_output_len}",
        "--gemm_plugin=float16",
        "--gpt_attention_plugin=float16",
        "--remove_input_padding=enable",
        "--paged_kv_cache=enable",
        "--context_fmha=enable",
        "--use_custom_all_reduce=enable",
    ]
    if quantization == "fp8":
        build_cmd.append("--strongly_typed")

    print(f"  Command: {' '.join(build_cmd)}")
    # subprocess.run(build_cmd, check=True)  # 取消注释以实际运行

    # Step 3: Verify
    print(f"\n[Step 3] Engine build complete!")
    print(f"  Output: {output_dir}/engine/")
    print(f"  Config: TP={tp_size}, dtype={dtype}, quant={quantization}")
    print(f"  Max batch: {max_batch_size}, Max input: {max_input_len}")

    return f"{output_dir}/engine"


def run_trtllm_benchmark(engine_dir: str, tp_size: int = 8):
    """运行 TRT-LLM 基准测试"""
    print(f"\n[Benchmark] Running TRT-LLM benchmark...")

    bench_cmd = [
        "python", "-m", "tensorrt_llm.commands.bench",
        f"--engine_dir={engine_dir}",
        "--dataset=ShareGPT",
        "--num_requests=1000",
        f"--tp_size={tp_size}",
    ]

    print(f"  Command: {' '.join(bench_cmd)}")
    # result = subprocess.run(bench_cmd, capture_output=True, text=True)
    # print(result.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TensorRT-LLM Engine")
    parser.add_argument("--model", default="meta-llama/Llama-2-70b-hf")
    parser.add_argument("--output", default="./trt_engines/llama2-70b")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--quant", choices=["fp8", "int8", None], default=None)
    parser.add_argument("--max-batch", type=int, default=64)
    args = parser.parse_args()

    print("=" * 70)
    print("  TensorRT-LLM Engine Builder")
    print("=" * 70)
    print(f"\n  Model: {args.model}")
    print(f"  TP Size: {args.tp}")
    print(f"  Dtype: {args.dtype}")
    print(f"  Quantization: {args.quant}")
    print(f"  Max Batch: {args.max_batch}")

    engine_dir = build_trtllm_engine(
        model_path=args.model,
        output_dir=args.output,
        tp_size=args.tp,
        dtype=args.dtype,
        max_batch_size=args.max_batch,
        quantization=args.quant,
    )

    print("\n" + "=" * 70)
    print("  Next Steps:")
    print("  1. Run benchmark: python benchmark_vs_vllm.py")
    print("  2. Deploy with Triton: tritonserver --model-repository=...")
    print("=" * 70)
