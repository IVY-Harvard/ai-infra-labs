"""
LoRA 微调 Qwen2-7B
硬件: 1 × H20 (96GB)，实际使用约 30GB
"""

import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# ============================================================
# 配置
# ============================================================
MODEL_NAME = "Qwen/Qwen2-7B"
OUTPUT_DIR = "./output/lora_qwen2_7b"
MAX_SEQ_LENGTH = 2048
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4  # effective batch = 16
LEARNING_RATE = 2e-4  # LoRA 可以用更大的学习率
SEED = 42

# LoRA 配置
LORA_R = 64
LORA_ALPHA = 128  # alpha = 2 * r
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",       # MLP
]


# ============================================================
# 数据准备
# ============================================================
def load_training_data():
    """
    加载训练数据
    实际使用时替换为你自己的数据集
    """
    # 示例：使用 Alpaca 中文数据
    # dataset = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train")

    # Demo: 创建模拟数据
    examples = []
    tasks = [
        ("请解释什么是量子计算", "量子计算是利用量子力学原理进行计算的技术。与经典计算机使用比特（0或1）不同，量子计算机使用量子比特（qubit），可以同时处于0和1的叠加态，从而实现并行计算能力的指数级提升。"),
        ("写一首关于春天的诗", "春风拂面暖如丝，\n万物复苏展新姿。\n桃花含笑迎朝阳，\n柳叶轻摇舞清池。"),
        ("如何学习编程", "学习编程建议分以下步骤：\n1. 选择一门语言（推荐Python入门）\n2. 学习基础语法和数据结构\n3. 通过项目实践巩固\n4. 阅读他人代码学习最佳实践\n5. 持续练习和解决实际问题"),
    ]

    import random
    random.seed(SEED)

    for _ in range(1000):
        q, a = random.choice(tasks)
        examples.append({
            "messages": [
                {"role": "system", "content": "你是一个知识渊博、乐于助人的AI助手。"},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        })

    return Dataset.from_list(examples)


def format_dataset(example, tokenizer):
    """应用 chat template"""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("LoRA 微调: Qwen2-7B")
    print(f"LoRA Config: r={LORA_R}, alpha={LORA_ALPHA}")
    print("=" * 60)

    # 1. 加载模型
    print("\n[1/5] 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 2. 配置 LoRA
    print("\n[2/5] 配置 LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. 准备数据
    print("\n[3/5] 准备数据...")
    dataset = load_training_data()
    dataset = dataset.map(
        lambda x: format_dataset(x, tokenizer),
        remove_columns=dataset.column_names,
    )
    split = dataset.train_test_split(test_size=0.05, seed=SEED)

    # 4. 训练配置
    print("\n[4/5] 配置训练...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        gradient_checkpointing=True,
        report_to="none",
        seed=SEED,
        optim="adamw_torch",  # 或 "paged_adamw_8bit" 进一步省显存
    )

    # 5. 训练
    print("\n[5/5] 开始训练...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_result = trainer.train()

    # 结果
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"训练 loss: {train_result.training_loss:.4f}")
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"峰值显存: {peak_mem:.2f} GB")

    # 保存 LoRA 权重
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nLoRA 权重已保存到: {OUTPUT_DIR}")

    # 打印 LoRA 文件大小
    lora_size = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f))
        for f in os.listdir(OUTPUT_DIR)
        if f.endswith('.safetensors') or f.endswith('.bin')
    ) / 1e6
    print(f"LoRA 文件大小: {lora_size:.1f} MB (vs 原模型 ~14GB)")


if __name__ == "__main__":
    main()
