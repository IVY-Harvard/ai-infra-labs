"""
Unsloth 加速微调
使用 Unsloth 实现 2-4x 训练速度提升

安装:
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

硬件: 1 × H20 (96GB)
"""

import os
import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

SEED = 42
OUTPUT_DIR = "./output/unsloth_qwen2_7b"


def create_training_data():
    """创建训练数据"""
    import random
    random.seed(SEED)

    examples = []
    tasks = [
        ("请用简单的语言解释什么是区块链", "区块链是一种分布式数据库技术。想象一个公共账本，每一页（区块）记录了一些交易，页与页之间通过密码学链接起来（链）。所有参与者都有这个账本的副本，任何人都无法篡改已记录的内容。"),
        ("给我写一段Python代码来排序列表", "```python\n# 方法1：内置排序\nnumbers = [3, 1, 4, 1, 5, 9, 2, 6]\nsorted_numbers = sorted(numbers)\nprint(sorted_numbers)  # [1, 1, 2, 3, 4, 5, 6, 9]\n\n# 方法2：原地排序\nnumbers.sort()\nprint(numbers)  # [1, 1, 2, 3, 4, 5, 6, 9]\n```"),
        ("中国有哪些著名的世界文化遗产", "中国著名的世界文化遗产包括：\n1. 长城 - 世界最长的防御工事\n2. 故宫 - 明清两代皇宫\n3. 秦始皇陵兵马俑 - 古代军事杰作\n4. 敦煌莫高窟 - 佛教艺术宝库\n5. 苏州古典园林 - 中国园林艺术典范"),
    ]

    for _ in range(2000):
        q, a = random.choice(tasks)
        examples.append({
            "messages": [
                {"role": "system", "content": "你是一个知识渊博的AI助手。"},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        })

    return examples


def main():
    print("=" * 60)
    print("Unsloth 加速微调: Qwen2-7B")
    print("=" * 60)

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("请先安装 Unsloth:")
        print('  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
        return

    # 1. 加载模型 (Unsloth 方式)
    print("\n[1/4] 使用 Unsloth 加载模型...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2-7B",
        max_seq_length=2048,
        dtype=None,  # 自动检测
        load_in_4bit=True,  # QLoRA 模式
    )

    # 2. 配置 LoRA (Unsloth 方式)
    print("\n[2/4] 配置 Unsloth LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=128,
        lora_dropout=0,  # Unsloth 推荐 dropout=0（已有正则化效果）
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth 优化的 GC
        random_state=SEED,
    )

    # 3. 准备数据
    print("\n[3/4] 准备数据...")
    raw_data = create_training_data()

    # 格式化为文本
    formatted_data = []
    for item in raw_data:
        text = tokenizer.apply_chat_template(
            item["messages"], tokenize=False, add_generation_prompt=False
        )
        formatted_data.append({"text": text})

    dataset = Dataset.from_list(formatted_data)
    split = dataset.train_test_split(test_size=0.05, seed=SEED)

    # 4. 训练
    print("\n[4/4] 开始 Unsloth 加速训练...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,  # Unsloth 更省显存，可用更大 batch
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        max_seq_length=2048,
        dataset_text_field="text",
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    import time
    start = time.time()
    train_result = trainer.train()
    training_time = time.time() - start

    # 结果
    print("\n" + "=" * 60)
    print("Unsloth 训练完成！")
    print("=" * 60)
    print(f"训练 Loss: {train_result.training_loss:.4f}")
    print(f"训练时间: {training_time:.1f}s")

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"峰值显存: {peak_mem:.2f} GB")

    tokens_per_sec = (len(split["train"]) * 2048 * 3) / training_time
    print(f"吞吐量: {tokens_per_sec:.0f} tokens/sec")

    # 保存
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Unsloth 也支持直接导出合并模型
    print("\n导出合并模型...")
    model.save_pretrained_merged(
        OUTPUT_DIR + "_merged",
        tokenizer,
        save_method="merged_16bit",  # 合并为 FP16
    )
    print(f"合并模型已保存到: {OUTPUT_DIR}_merged")


if __name__ == "__main__":
    main()
