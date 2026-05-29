"""
速度对比实验：Unsloth vs 标准 HuggingFace 训练
对比相同配置下的训练速度和显存使用

用法:
    python speed_comparison.py
    python speed_comparison.py --model_name Qwen/Qwen2-1.5B --steps 50
"""

import argparse
import time
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

SEED = 42


def create_benchmark_data(tokenizer, num_samples=500, seq_len=1024):
    """创建 benchmark 数据"""
    import random
    random.seed(SEED)

    text = "这是一段用于性能测试的文本。" * 50
    examples = []
    for _ in range(num_samples):
        examples.append({
            "messages": [
                {"role": "user", "content": text[:200]},
                {"role": "assistant", "content": text[:500]},
            ]
        })

    dataset = Dataset.from_list(examples)
    dataset = dataset.map(
        lambda x: {"text": tokenizer.apply_chat_template(x["messages"], tokenize=False)},
        remove_columns=dataset.column_names,
    )
    return dataset


def benchmark_standard_hf(model_name, dataset, tokenizer, max_steps=100):
    """标准 HuggingFace 训练速度"""
    print("\n--- 标准 HuggingFace (QLoRA) ---")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=64, lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="./tmp/bench_hf",
        max_steps=max_steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        gradient_checkpointing=True,
        report_to="none",
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=1024,
        dataset_text_field="text",
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    result = {
        "method": "Standard HuggingFace",
        "time_sec": elapsed,
        "steps": max_steps,
        "sec_per_step": elapsed / max_steps,
        "peak_memory_gb": peak_mem,
    }

    print(f"  时间: {elapsed:.1f}s ({elapsed/max_steps:.2f}s/step)")
    print(f"  显存: {peak_mem:.2f} GB")

    del model, trainer
    torch.cuda.empty_cache()

    return result


def benchmark_unsloth(model_name, dataset, tokenizer, max_steps=100):
    """Unsloth 训练速度"""
    print("\n--- Unsloth (QLoRA) ---")

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("  Unsloth 未安装，跳过")
        return None

    model, tokenizer_us = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model, r=64, lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth",
    )

    training_args = TrainingArguments(
        output_dir="./tmp/bench_unsloth",
        max_steps=max_steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer_us,
        max_seq_length=1024,
        dataset_text_field="text",
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    result = {
        "method": "Unsloth",
        "time_sec": elapsed,
        "steps": max_steps,
        "sec_per_step": elapsed / max_steps,
        "peak_memory_gb": peak_mem,
    }

    print(f"  时间: {elapsed:.1f}s ({elapsed/max_steps:.2f}s/step)")
    print(f"  显存: {peak_mem:.2f} GB")

    del model, trainer
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2-7B")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    print("=" * 60)
    print("训练速度对比: Unsloth vs Standard HuggingFace")
    print(f"模型: {args.model_name}")
    print(f"步数: {args.steps}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = create_benchmark_data(tokenizer)

    # 运行 benchmark
    hf_result = benchmark_standard_hf(args.model_name, dataset, tokenizer, args.steps)
    unsloth_result = benchmark_unsloth(args.model_name, dataset, tokenizer, args.steps)

    # 对比
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    print(f"{'方法':<25} {'总时间':<12} {'每步时间':<12} {'显存':<10}")
    print("-" * 60)
    print(f"{'Standard HuggingFace':<25} {hf_result['time_sec']:<12.1f} "
          f"{hf_result['sec_per_step']:<12.3f} {hf_result['peak_memory_gb']:<10.2f}")

    if unsloth_result:
        print(f"{'Unsloth':<25} {unsloth_result['time_sec']:<12.1f} "
              f"{unsloth_result['sec_per_step']:<12.3f} {unsloth_result['peak_memory_gb']:<10.2f}")

        speedup = hf_result['time_sec'] / unsloth_result['time_sec']
        mem_saving = hf_result['peak_memory_gb'] - unsloth_result['peak_memory_gb']
        print(f"\nUnsloth 加速比: {speedup:.2f}x")
        print(f"显存节省: {mem_saving:.2f} GB")

    # 保存结果
    results = {"standard_hf": hf_result, "unsloth": unsloth_result}
    with open("speed_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
