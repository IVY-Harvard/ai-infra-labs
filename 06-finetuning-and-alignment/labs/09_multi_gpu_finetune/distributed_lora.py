"""
分布式 LoRA 微调：使用 Accelerate 进行多卡 LoRA 训练

启动方式:
    accelerate launch --num_processes 8 distributed_lora.py
    # 或
    torchrun --nproc_per_node 8 distributed_lora.py

硬件: 4-8 × H20
适用: 大模型 LoRA（如 70B）或大 batch 训练
"""

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

# ============================================================
# 配置
# ============================================================
MODEL_NAME = "Qwen/Qwen2-7B"
OUTPUT_DIR = "./output/distributed_lora"
MAX_SEQ_LENGTH = 2048
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
SEED = 42

# 多卡配置
# 总 effective batch = per_device_batch * gradient_accum * num_gpus
PER_DEVICE_BATCH = 4
GRADIENT_ACCUMULATION = 2
# 8 GPU: effective batch = 4 * 2 * 8 = 64

# LoRA 配置
LORA_R = 64
LORA_ALPHA = 128


def create_training_data(num_samples=5000):
    """创建较大规模的训练数据（多卡训练需要更多数据）"""
    import random
    random.seed(SEED)

    templates = [
        ("请用简洁的语言解释{topic}", "{explanation}"),
        ("写一段关于{topic}的介绍", "{explanation}"),
        ("{topic}的主要特点是什么？", "{explanation}"),
    ]

    topics = [
        ("Python编程", "Python是一种高级编程语言，以其简洁的语法和丰富的库生态系统而著称。"),
        ("深度学习", "深度学习是机器学习的一个分支，使用多层神经网络来学习数据的复杂表示。"),
        ("云计算", "云计算通过互联网提供按需的计算资源，包括服务器、存储、数据库等。"),
        ("区块链", "区块链是一种分布式账本技术，通过密码学保证数据的不可篡改性。"),
        ("量子计算", "量子计算利用量子力学原理进行计算，理论上可以解决经典计算机难以处理的问题。"),
    ]

    examples = []
    for _ in range(num_samples):
        template = random.choice(templates)
        topic, explanation = random.choice(topics)
        examples.append({
            "messages": [
                {"role": "user", "content": template[0].format(topic=topic)},
                {"role": "assistant", "content": template[1].format(explanation=explanation)},
            ]
        })

    return Dataset.from_list(examples)


def main():
    # 获取分布式信息
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if local_rank == 0:
        print("=" * 60)
        print("分布式 LoRA 微调")
        print(f"模型: {MODEL_NAME}")
        print(f"GPU 数量: {world_size}")
        print(f"Effective batch size: {PER_DEVICE_BATCH * GRADIENT_ACCUMULATION * world_size}")
        print("=" * 60)

    # 1. 加载模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # 注意: 多卡训练不使用 device_map="auto"
        # Trainer/accelerate 会自动处理设备分配
    )

    # 2. LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    if local_rank == 0:
        model.print_trainable_parameters()

    # 3. 数据
    dataset = create_training_data(5000)
    dataset = dataset.map(
        lambda x: {"text": tokenizer.apply_chat_template(x["messages"], tokenize=False)},
        remove_columns=dataset.column_names,
    )
    split = dataset.train_test_split(test_size=0.05, seed=SEED)

    # 4. 训练参数（多卡）
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
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
        # 分布式相关
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        # 日志
        report_to="none",
        seed=SEED,
    )

    # 5. 训练
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
    )

    trainer.train()

    # 保存（只在 rank 0 保存）
    if local_rank == 0:
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"\n模型已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
