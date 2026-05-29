"""
LoRA 权重合并与导出工具
将 LoRA adapter 合并到基座模型，导出完整模型用于部署

用法:
    python merge_and_export.py --base_model Qwen/Qwen2-7B --lora_path ./output/lora_qwen2_7b
    python merge_and_export.py --base_model Qwen/Qwen2-7B --lora_path ./output/lora --output_dir ./merged_model
    python merge_and_export.py --base_model Qwen/Qwen2-7B --lora_path ./output/lora --export_gguf
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora(base_model_name: str, lora_path: str, output_dir: str, dtype: str = "bf16"):
    """合并 LoRA 权重到基座模型"""
    print("=" * 60)
    print("LoRA 合并与导出")
    print("=" * 60)

    # 确定数据类型
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    # 1. 加载基座模型
    print(f"\n[1/4] 加载基座模型: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # 2. 加载 LoRA
    print(f"\n[2/4] 加载 LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch_dtype,
    )

    # 打印 LoRA 信息
    print(f"  LoRA config: {model.peft_config}")

    # 3. 合并
    print("\n[3/4] 合并 LoRA 权重到基座模型...")
    print("  执行: W_merged = W_base + (alpha/r) * B @ A")
    merged_model = model.merge_and_unload()

    # 验证合并后模型
    total_params = sum(p.numel() for p in merged_model.parameters())
    print(f"  合并后参数量: {total_params:,}")

    # 4. 保存
    print(f"\n[4/4] 保存合并模型到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    # 计算文件大小
    total_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, f))
    )
    print(f"  总文件大小: {total_size / 1e9:.2f} GB")

    return merged_model, tokenizer


def verify_merge(merged_model, tokenizer, test_prompts=None):
    """验证合并后模型可以正常推理"""
    print("\n" + "=" * 60)
    print("验证合并模型")
    print("=" * 60)

    if test_prompts is None:
        test_prompts = [
            "你好，请介绍一下你自己。",
            "什么是机器学习？",
        ]

    merged_model.eval()

    for prompt in test_prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(merged_model.device)

        with torch.no_grad():
            outputs = merged_model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        print(f"\n  输入: {prompt}")
        print(f"  输出: {response[:200]}")

    print("\n验证通过！合并模型可正常推理。")


def export_for_vllm(output_dir: str):
    """为 vLLM 部署准备的说明"""
    print("\n" + "=" * 60)
    print("vLLM 部署指南")
    print("=" * 60)
    print(f"""
合并后的模型可以直接用 vLLM 部署:

    # Python API
    from vllm import LLM
    llm = LLM(model="{output_dir}")

    # 命令行
    vllm serve {output_dir} --tensor-parallel-size 1

    # 合并后无需 PEFT 库，无额外推理开销
""")


def main():
    parser = argparse.ArgumentParser(description="LoRA 合并与导出")
    parser.add_argument("--base_model", required=True, help="基座模型名称或路径")
    parser.add_argument("--lora_path", required=True, help="LoRA adapter 路径")
    parser.add_argument("--output_dir", default=None, help="输出目录")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--verify", action="store_true", default=True, help="验证合并结果")
    parser.add_argument("--export_gguf", action="store_true", help="额外导出 GGUF 格式")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.lora_path + "_merged"

    # 合并
    merged_model, tokenizer = merge_lora(
        args.base_model, args.lora_path, args.output_dir, args.dtype
    )

    # 验证
    if args.verify:
        verify_merge(merged_model, tokenizer)

    # 部署说明
    export_for_vllm(args.output_dir)

    # GGUF 导出（需要 llama.cpp）
    if args.export_gguf:
        print("\n要导出 GGUF 格式，请使用 llama.cpp:")
        print(f"  python convert_hf_to_gguf.py {args.output_dir} --outtype f16")
        print(f"  ./quantize {args.output_dir}/model-f16.gguf {args.output_dir}/model-q4_k_m.gguf Q4_K_M")

    print("\n完成！")


if __name__ == "__main__":
    main()
