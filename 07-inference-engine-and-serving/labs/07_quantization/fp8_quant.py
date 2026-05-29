"""FP8 量化 — H20 原生支持，推荐方案"""
import argparse

def quantize_fp8(model_name: str, output_dir: str):
    """FP8 量化 (H20 原生 Tensor Core 支持)"""
    print("=" * 70)
    print("  FP8 Quantization (Recommended for H20)")
    print("=" * 70)
    print(f"\n  FP8 on H20:")
    print(f"  - H20 supports FP8 (E4M3) natively on Tensor Core")
    print(f"  - No calibration data needed (dynamic quantization)")
    print(f"  - Almost lossless (<0.1% PPL increase)")
    print(f"  - 2x memory reduction → 2x batch size → 2x throughput")
    print(f"\n  Usage with vLLM (no separate quantization step!):")
    print(f"  vllm serve {model_name} \\")
    print(f"    --quantization fp8 \\")
    print(f"    --kv-cache-dtype fp8 \\")
    print(f"    --tensor-parallel-size 8")
    print(f"\n  That's it! vLLM handles FP8 conversion on-the-fly.")
    print(f"\n  For pre-quantized models (slightly faster startup):")
    print(f"  pip install llm-compressor")
    print(f"  python -c \"")
    print(f"  from llmcompressor.modifiers.quantization import QuantizationModifier")
    print(f"  from llmcompressor import oneshot")
    print(f"  recipe = QuantizationModifier(targets='Linear', scheme='FP8')")
    print(f"  oneshot(model='{model_name}', recipe=recipe, output_dir='{output_dir}')\"")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-2-70b-hf")
    parser.add_argument("--output", default="./quantized/fp8")
    args = parser.parse_args()
    quantize_fp8(args.model, args.output)
