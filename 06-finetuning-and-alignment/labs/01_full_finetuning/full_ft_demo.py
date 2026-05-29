"""
全量微调 Demo：使用 Qwen2-1.5B 进行 SFT 全量微调
硬件要求：1 × H20 (96GB) — 1.5B 模型全量微调约需 15-20GB
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from trl import SFTTrainer

# ============================================================
# 配置
# ============================================================
MODEL_NAME = "Qwen/Qwen2-1.5B"  # 使用小模型作为 baseline
OUTPUT_DIR = "./output/full_ft_qwen2_1.5b"
MAX_SEQ_LENGTH = 1024
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-5  # 全量微调用较小学习率
SEED = 42


# ============================================================
# 准备演示数据
# ============================================================
def create_demo_dataset(num_samples=500):
    """创建演示用的指令数据集"""
    examples = []

    # 模拟多种任务的训练数据
    task_templates = [
        {
            "instruction": "请将以下文本翻译成英文：{text}",
            "texts": ["你好世界", "机器学习是一种人工智能技术", "今天天气很好"],
            "outputs": ["Hello World", "Machine learning is an AI technology", "The weather is nice today"],
        },
        {
            "instruction": "请对以下文本进行摘要：{text}",
            "texts": [
                "人工智能（AI）是计算机科学的一个分支，它致力于创建能够模拟人类智能行为的系统。"
                "这包括学习、推理、自我修正和理解自然语言等能力。",
            ],
            "outputs": ["AI是计算机科学分支，致力于模拟人类智能行为，包括学习、推理等能力。"],
        },
        {
            "instruction": "请回答以下问题：{text}",
            "texts": ["什么是深度学习？", "Python的主要特点是什么？"],
            "outputs": [
                "深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的层次化表示。",
                "Python的主要特点包括：简洁易读的语法、丰富的标准库、跨平台支持、动态类型和自动内存管理。",
            ],
        },
    ]

    import random
    random.seed(SEED)

    for _ in range(num_samples):
        task = random.choice(task_templates)
        idx = random.randint(0, len(task["texts"]) - 1)
        text = task["texts"][idx]
        output = task["outputs"][idx]

        examples.append({
            "messages": [
                {"role": "system", "content": "你是一个有帮助的AI助手。"},
                {"role": "user", "content": task["instruction"].format(text=text)},
                {"role": "assistant", "content": output},
            ]
        })

    return Dataset.from_list(examples)


def format_messages(example, tokenizer):
    """将 messages 格式转换为模型输入"""
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
    print("全量微调 Demo: Qwen2-1.5B")
    print("=" * 60)

    # 1. 加载 tokenizer 和模型
    print("\n[1/5] 加载模型和 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # 2. 准备数据
    print("\n[2/5] 准备训练数据...")
    dataset = create_demo_dataset(500)
    dataset = dataset.map(
        lambda x: format_messages(x, tokenizer),
        remove_columns=dataset.column_names,
    )

    # 划分训练/验证
    split = dataset.train_test_split(test_size=0.1, seed=SEED)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(eval_dataset)} 样本")

    # 3. 训练参数
    print("\n[3/5] 配置训练参数...")
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
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        gradient_checkpointing=True,  # 节省显存
        report_to="none",  # 不使用 wandb（demo 模式）
        seed=SEED,
        dataloader_num_workers=4,
    )

    # 4. 创建 Trainer
    print("\n[4/5] 创建 Trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
    )

    # 5. 开始训练
    print("\n[5/5] 开始训练...")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")

    # 显存使用情况
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_result = trainer.train()

    # 打印训练结果
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"训练 loss: {train_result.training_loss:.4f}")
    print(f"训练时间: {train_result.metrics['train_runtime']:.1f} 秒")
    print(f"训练样本/秒: {train_result.metrics['train_samples_per_second']:.2f}")

    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"峰值显存: {peak_memory:.2f} GB")

    # 保存模型
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n模型已保存到: {OUTPUT_DIR}")

    # 6. 简单推理测试
    print("\n[测试] 推理验证...")
    model.eval()
    test_messages = [
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
        {"role": "user", "content": "请将以下文本翻译成英文：你好世界"},
    ]

    inputs = tokenizer.apply_chat_template(
        test_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    print(f"输入: 请将以下文本翻译成英文：你好世界")
    print(f"输出: {response}")


if __name__ == "__main__":
    main()
