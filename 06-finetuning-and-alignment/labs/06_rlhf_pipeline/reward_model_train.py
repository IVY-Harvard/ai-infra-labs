"""
Reward Model 训练
使用偏好数据（chosen/rejected pairs）训练 Reward Model

硬件: 1 × H20 (使用小模型作为 RM)
"""

import os
import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    PreTrainedModel,
)
from typing import Dict, List

# ============================================================
# 配置
# ============================================================
MODEL_NAME = "Qwen/Qwen2-1.5B"  # RM 通常比 policy 小
OUTPUT_DIR = "./output/reward_model"
MAX_LENGTH = 2048
NUM_EPOCHS = 1  # RM 通常只训练 1 epoch
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
SEED = 42


# ============================================================
# 数据准备
# ============================================================
def create_preference_data(num_samples=500):
    """创建偏好数据（demo）"""
    import random
    random.seed(SEED)

    preference_pairs = [
        {
            "prompt": "什么是Python？",
            "chosen": "Python是一种高级编程语言，由Guido van Rossum于1991年创建。它以简洁优雅的语法著称，支持多种编程范式（面向对象、函数式、过程式）。Python拥有丰富的标准库和第三方生态系统，广泛应用于Web开发、数据科学、人工智能、自动化运维等领域。",
            "rejected": "Python就是一种编程语言。",
        },
        {
            "prompt": "如何提高编程能力？",
            "chosen": "提高编程能力可以从以下几个方面着手：\n1. 坚持每天写代码，养成编程习惯\n2. 阅读优秀的开源项目代码\n3. 参与实际项目开发\n4. 学习算法和数据结构\n5. 写技术博客总结所学\n6. 参加代码审查，学习他人经验",
            "rejected": "多写代码就行了，没什么技巧。",
        },
        {
            "prompt": "解释深度学习",
            "chosen": "深度学习是机器学习的一个子领域，它使用多层神经网络来学习数据中的层次化表示。与传统机器学习需要手动设计特征不同，深度学习能够自动从原始数据中学习有用的特征。核心组件包括神经元、层、激活函数、损失函数和优化器。常见架构有CNN（图像）、RNN/Transformer（序列）等。",
            "rejected": "深度学习就是很深的学习，用很多层的网络。不太好解释，你去看论文吧。",
        },
    ]

    data = []
    for _ in range(num_samples):
        pair = random.choice(preference_pairs)
        data.append(pair)

    return Dataset.from_list(data)


class RewardDataCollator:
    """Reward Model 数据整理器"""

    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict]) -> Dict:
        # 分别编码 chosen 和 rejected
        chosen_texts = []
        rejected_texts = []

        for f in features:
            prompt = f["prompt"]
            chosen_texts.append(f"{prompt}\n{f['chosen']}")
            rejected_texts.append(f"{prompt}\n{f['rejected']}")

        chosen_encodings = self.tokenizer(
            chosen_texts, max_length=self.max_length,
            truncation=True, padding=True, return_tensors="pt"
        )
        rejected_encodings = self.tokenizer(
            rejected_texts, max_length=self.max_length,
            truncation=True, padding=True, return_tensors="pt"
        )

        return {
            "chosen_input_ids": chosen_encodings["input_ids"],
            "chosen_attention_mask": chosen_encodings["attention_mask"],
            "rejected_input_ids": rejected_encodings["input_ids"],
            "rejected_attention_mask": rejected_encodings["attention_mask"],
        }


class RewardTrainer(Trainer):
    """自定义 Reward Model Trainer"""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 计算 chosen 的 reward
        chosen_outputs = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
        )
        chosen_rewards = chosen_outputs.logits[:, 0]  # shape: (batch,)

        # 计算 rejected 的 reward
        rejected_outputs = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
        )
        rejected_rewards = rejected_outputs.logits[:, 0]

        # Bradley-Terry loss
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()

        if return_outputs:
            return loss, {"chosen_rewards": chosen_rewards, "rejected_rewards": rejected_rewards}
        return loss


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("Reward Model 训练")
    print(f"基座模型: {MODEL_NAME}")
    print("=" * 60)

    # 1. 加载模型
    print("\n[1/4] 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 使用 SequenceClassification 头作为 reward head
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,  # 输出单个 reward 值
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params/1e9:.2f}B")

    # 2. 准备数据
    print("\n[2/4] 准备偏好数据...")
    dataset = create_preference_data(500)
    split = dataset.train_test_split(test_size=0.1, seed=SEED)
    print(f"  训练集: {len(split['train'])} 对")
    print(f"  验证集: {len(split['test'])} 对")

    # 3. 训练
    print("\n[3/4] 配置训练...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        gradient_checkpointing=True,
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
    )

    data_collator = RewardDataCollator(tokenizer, MAX_LENGTH)

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 4. 训练
    print("\n[4/4] 开始训练...")
    train_result = trainer.train()

    print("\n" + "=" * 60)
    print("Reward Model 训练完成！")
    print(f"训练 Loss: {train_result.training_loss:.4f}")
    print("=" * 60)

    # 保存
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nRM 已保存到: {OUTPUT_DIR}")

    # 验证：计算准确率
    print("\n验证 RM 准确率...")
    model.eval()
    correct = 0
    total = 0

    for item in split["test"]:
        prompt = item["prompt"]
        chosen_text = f"{prompt}\n{item['chosen']}"
        rejected_text = f"{prompt}\n{item['rejected']}"

        chosen_enc = tokenizer(chosen_text, return_tensors="pt", max_length=512, truncation=True)
        rejected_enc = tokenizer(rejected_text, return_tensors="pt", max_length=512, truncation=True)

        with torch.no_grad():
            chosen_reward = model(**chosen_enc.to(model.device)).logits[0, 0].item()
            rejected_reward = model(**rejected_enc.to(model.device)).logits[0, 0].item()

        if chosen_reward > rejected_reward:
            correct += 1
        total += 1

    accuracy = correct / total
    print(f"RM 准确率: {accuracy:.2%} (目标 > 70%)")


if __name__ == "__main__":
    main()
