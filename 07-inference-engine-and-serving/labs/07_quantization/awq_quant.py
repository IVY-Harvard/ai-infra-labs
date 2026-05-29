"""
AWQ 量化

AWQ: Activation-Aware Weight Quantization
核心思想: 保护激活值大的通道对应的权重。
"""

import argparse
import time


def quantize_awq(model_name: str, output_dir: str, bits: int = 4, group_size: int = 128):
    """使用 AWQ 量化模型"""
    print("=" * 70)
    print("  AWQ Quantization")
    print("=" * 70)

    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        print("  Please install: pip install autoawq")
        _show_awq_steps(model_name, bits, group_size)
        return

    print(f"\n  Model: {model_name}")
    print(f"  Bits: {bits}, Group Size: {group_size}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(model_name)

    quant_config = {
        "zero_point": True,
        "q_group_size": group_size,
        "w_bit": bits,
        "version": "GEMM",  # GEMM or GEMV kernel
    }

    print(f"\n  Quantizing...")
    start = time.time()
    model.quantize(tokenizer, quant_config=quant_config)
    print(f"  Time: {time.time() - start:.1f}s")

    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n  Saved to {output_dir}")
    print(f"  Use: vllm serve {output_dir} --quantization awq")


def _show_awq_steps(model_name: str, bits: int, group_size: int):
    """展示 AWQ 量化步骤"""
    print(f"\n  AWQ Quantization Steps:")
    print(f"  ─────────────────────────")
    print(f"  1. Load model and run calibration data")
    print(f"  2. For each linear layer:")
    print(f"     a. Measure activation magnitudes per channel: s_j = max(|X_j|)")
    print(f"     b. Identify 'salient' channels (top 1% by activation)")
    print(f"     c. Compute scaling factors: scale_j = s_j^alpha / max(|W_j|)^(1-alpha)")
    print(f"     d. Scale up important weights: W_j *= scale_j")
    print(f"     e. Quantize all weights to {bits}-bit (per group of {group_size})")
    print(f"     f. Scale down activations at runtime: X_j /= scale_j")
    print(f"  3. Result: important weights have lower quantization error")
    print(f"\n  AWQ vs GPTQ:")
    print(f"  - AWQ: faster quantization (no Hessian computation)")
    print(f"  - AWQ: slightly better quality at extreme quantization (W3)")
    print(f"  - GPTQ: more established, wider tool support")
    print(f"  - Both: similar speed/quality at W4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--output", default="./quantized/awq-4bit")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()

    quantize_awq(args.model, args.output, args.bits, args.group_size)
