"""
Rank 对比实验：测试不同 LoRA rank 对微调效果的影响
用于确定特定任务的最优 rank 设置

用法:
    python rank_experiment.py --ranks 4 8 16 32 64 128
    python rank_experiment.py --model_name Qwen/Qwen2-1.5B --quick
"""

import argparse
import json
import time
import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

SEED = 42


def create_experiment_data(tokenizer, num_samples=800):
    """创建统一的实验数据"""
    import random
    random.seed(SEED)

    tasks = [
        ("将以下句子改写为正式语气：今天天气不错啊", "今日天气状况良好。"),
        ("用一句话总结：深度学习使用多层神经网络来学习数据中的复杂模式和表示。", "深度学习通过多层神经网络学习数据的复杂模式。"),
        ("回答：什么是Git？", "Git是一个分布式版本控制系统，用于跟踪文件变化和协同开发。"),
    ]

    examples = []
    for _ in range(num_samples):
        q, a = random.choice(tasks)
        examples.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        })

    dataset = Dataset.from_list(examples)
    dataset = dataset.map(
        lambda x: {"text": tokenizer.apply_chat_template(x["messages"], tokenize=False)},
        remove_columns=dataset.column_names,
    )
    return dataset.train_test_split(test_size=0.1, seed=SEED)


def run_single_experiment(model_name, rank, alpha, dataset, tokenizer, output_base, quick=False):
    """运行单个 rank 的训练实验"""
    print(f"\n{'='*50}")
    print(f"实验: rank={rank}, alpha={alpha}")
    print(f"{'='*50}")

    output_dir = os.path.join(output_base, f"rank_{rank}")

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA 配置
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # 参数统计
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    # 训练
    epochs = 1 if quick else 3
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="no",
        gradient_checkpointing=True,
        report_to="none",
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        max_seq_length=512,
        dataset_text_field="text",
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time

    # 评估
    eval_result = trainer.evaluate()

    peak_memory = 0
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1e9

    result = {
        "rank": rank,
        "alpha": alpha,
        "trainable_params": trainable,
        "trainable_pct": 100 * trainable / total,
        "train_loss": train_result.training_loss,
        "eval_loss": eval_result["eval_loss"],
        "training_time_sec": training_time,
        "peak_memory_gb": peak_memory,
        "tokens_per_sec": train_result.metrics.get("train_samples_per_second", 0) * 512,
    }

    print(f"  训练参数: {trainable:,} ({result['trainable_pct']:.2f}%)")
    print(f"  训练 Loss: {result['train_loss']:.4f}")
    print(f"  验证 Loss: {result['eval_loss']:.4f}")
    print(f"  训练时间: {training_time:.1f}s")
    print(f"  峰值显存: {peak_memory:.2f} GB")

    # 清理
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2-1.5B")
    parser.add_argument("--ranks", nargs="+", type=int, default=[4, 8, 16, 32, 64, 128])
    parser.add_argument("--output_dir", default="./output/rank_experiment")
    parser.add_argument("--quick", action="store_true", help="快速实验（1 epoch）")
    args = parser.parse_args()

    print("=" * 60)
    print("LoRA Rank 对比实验")
    print(f"模型: {args.model_name}")
    print(f"测试 Ranks: {args.ranks}")
    print("=" * 60)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 准备数据（所有实验用同一份数据）
    dataset = create_experiment_data(tokenizer)

    # 运行实验
    all_results = []
    for rank in args.ranks:
        alpha = 2 * rank  # alpha = 2r 是常见做法
        result = run_single_experiment(
            args.model_name, rank, alpha, dataset, tokenizer,
            args.output_dir, args.quick
        )
        all_results.append(result)

    # 汇总结果
    print("\n" + "=" * 80)
    print("实验结果汇总")
    print("=" * 80)
    print(f"{'Rank':<6} {'Alpha':<7} {'Params':<12} {'Train Loss':<12} "
          f"{'Eval Loss':<12} {'Time(s)':<10} {'Memory(GB)':<12}")
    print("-" * 80)

    for r in all_results:
        print(f"{r['rank']:<6} {r['alpha']:<7} {r['trainable_params']:>10,} "
              f"{r['train_loss']:<12.4f} {r['eval_loss']:<12.4f} "
              f"{r['training_time_sec']:<10.1f} {r['peak_memory_gb']:<12.2f}")

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n详细结果已保存到: {results_file}")

    # 推荐
    best = min(all_results, key=lambda x: x["eval_loss"])
    print(f"\n推荐 Rank: {best['rank']} (最低验证 Loss: {best['eval_loss']:.4f})")


if __name__ == "__main__":
    main()
