"""
GPTQ 量化

GPTQ: 基于 Hessian 信息的 Weight-Only INT4 量化。
逐列量化权重，用剩余列补偿量化误差。
"""

import argparse
import time
import torch


def quantize_gptq(model_name: str, output_dir: str, bits: int = 4, group_size: int = 128):
    """
    使用 GPTQ 量化模型

    需要安装: pip install auto-gptq
    """
    print("=" * 70)
    print("  GPTQ Quantization")
    print("=" * 70)

    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from transformers import AutoTokenizer
    except ImportError:
        print("  Please install: pip install auto-gptq")
        print("  Showing quantization steps instead...")
        _show_gptq_steps(model_name, bits, group_size)
        return

    print(f"\n  Model: {model_name}")
    print(f"  Bits: {bits}")
    print(f"  Group Size: {group_size}")

    # 量化配置
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=False,  # 激活降序 (更精确但更慢)
    )

    # 加载模型
    print(f"\n  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoGPTQForCausalLM.from_pretrained(
        model_name, quantize_config=quantize_config
    )

    # 准备校准数据
    print(f"  Preparing calibration data...")
    calibration_texts = [
        "The meaning of life is",
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language",
    ] * 43  # ~128 samples

    calibration_data = [
        tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        for text in calibration_texts
    ]

    # 量化
    print(f"  Quantizing (this may take hours for large models)...")
    start = time.time()
    model.quantize(calibration_data)
    quant_time = time.time() - start
    print(f"  Quantization time: {quant_time:.1f}s")

    # 保存
    print(f"  Saving to {output_dir}...")
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n  Done! Quantized model saved to {output_dir}")
    print(f"  Use with vLLM: vllm serve {output_dir} --quantization gptq")


def _show_gptq_steps(model_name: str, bits: int, group_size: int):
    """展示 GPTQ 量化步骤 (无需实际运行)"""
    print(f"\n  GPTQ Quantization Steps:")
    print(f"  ─────────────────────────")
    print(f"  1. Load model: {model_name}")
    print(f"  2. Prepare calibration data (128 samples)")
    print(f"  3. For each layer:")
    print(f"     a. Run calibration data, collect Hessian H = X^T X")
    print(f"     b. For each column j of weight W:")
    print(f"        - Quantize column j to {bits}-bit")
    print(f"        - Compute quantization error e_j")
    print(f"        - Update remaining columns to compensate: W[:,j+1:] -= e_j * H[j,j+1:] / H[j,j]")
    print(f"     c. Group size {group_size}: every {group_size} elements share scale/zero")
    print(f"  4. Save quantized weights + scales + zeros")
    print(f"\n  Expected results:")
    print(f"  - Model size: ~50% of FP16 (W4 + scales overhead)")
    print(f"  - PPL increase: < 0.5 (almost lossless)")
    print(f"  - Decode speedup: ~1.5-2x (less data to read from HBM)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--output", default="./quantized/gptq-4bit")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()

    quantize_gptq(args.model, args.output, args.bits, args.group_size)
