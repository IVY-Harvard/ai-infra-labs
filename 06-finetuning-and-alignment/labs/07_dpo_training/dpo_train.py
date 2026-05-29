"""
DPO 训练：直接偏好优化
使用 TRL 的 DPOTrainer 实现

硬件: 1 × H20 (需要 policy + ref model)
"""

import os
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer, DPOConfig

# ============================================================
# 配置
# ============================================================
MODEL_NAME = "Qwen/Qwen2-1.5B"  # 使用小模型 demo，实际替换为 SFT 模型
OUTPUT_DIR = "./output/dpo_qwen2"
MAX_LENGTH = 2048
MAX_PROMPT_LENGTH = 1024
BETA = 0.1  # DPO 温度参数
LEARNING_RATE = 5e-7  # DPO 需要很小的学习率
NUM_EPOCHS = 1  # 通常 1 epoch 足够
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
SEED = 42


# ============================================================
# 数据准备
# ============================================================
def create_preference_dataset(num_samples=500):
    """创建 DPO 偏好数据"""
    import random
    random.seed(SEED)

    preference_pairs = [
        {
            "prompt": "什么是Python？",
            "chosen": "Python是一种高级编程语言，由Guido van Rossum于1991年首次发布。它以简洁、易读的语法著称，遵循'优美胜于丑陋'的设计哲学。Python支持多种编程范式，拥有丰富的标准库和第三方生态系统，广泛应用于Web开发、数据科学、人工智能等领域。",
            "rejected": "Python就是一种编程语言，用来写代码的。",
        },
        {
            "prompt": "如何学好英语？",
            "chosen": "学好英语可以从以下几个方面入手：\n1. 建立坚实的语法基础\n2. 大量阅读英文原版材料\n3. 通过听播客、看英文电影提高听力\n4. 每天练习口语，可以找语伴或使用AI对话\n5. 用英文写日记或博客\n6. 背单词要结合语境，而非死记硬背\n持之以恒是关键，建议每天至少投入30分钟。",
            "rejected": "多背单词，多看美剧就好了。",
        },
        {
            "prompt": "帮我写一封请假邮件",
            "chosen": "主题：请假申请 - [您的姓名] - [日期]\n\n尊敬的[上级姓名]：\n\n您好！因[个人事务/身体不适]，特申请于[日期]请假[X]天（[起始日期]至[结束日期]）。\n\n在请假期间，我已安排[同事姓名]代为处理紧急事务，并确保不影响工作进度。如有任何问题可通过手机联系我。\n\n感谢您的理解与支持！\n\n此致\n敬礼\n[您的姓名]\n[日期]",
            "rejected": "领导你好，我要请假几天，谢谢。",
        },
    ]

    data = []
    for _ in range(num_samples):
        pair = random.choice(preference_pairs)
        data.append(pair)

    return Dataset.from_list(data)


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("DPO 训练")
    print(f"模型: {MODEL_NAME}")
    print(f"Beta: {BETA}, LR: {LEARNING_RATE}")
    print("=" * 60)

    # 1. 加载模型
    print("\n[1/4] 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Policy Model（要训练的）
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Reference Model（冻结的）
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # 2. 准备数据
    print("\n[2/4] 准备偏好数据...")
    dataset = create_preference_dataset(500)
    split = dataset.train_test_split(test_size=0.1, seed=SEED)
    print(f"  训练集: {len(split['train'])} 对")
    print(f"  验证集: {len(split['test'])} 对")

    # 3. DPO 训练配置
    print("\n[3/4] 配置 DPO 训练...")
    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=BETA,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=NUM_EPOCHS,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        gradient_checkpointing=True,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        loss_type="sigmoid",  # 标准 DPO loss
        report_to="none",
        seed=SEED,
    )

    # 4. 训练
    print("\n[4/4] 开始 DPO 训练...")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_result = trainer.train()

    # 结果
    print("\n" + "=" * 60)
    print("DPO 训练完成！")
    print("=" * 60)
    print(f"训练 Loss: {train_result.training_loss:.4f}")
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"峰值显存: {peak_mem:.2f} GB")

    # 保存
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nDPO 模型已保存到: {OUTPUT_DIR}")

    # 简单验证
    print("\n验证 DPO 效果...")
    model.eval()
    test_prompt = "什么是Python？"
    messages = [{"role": "user", "content": test_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(f"  问题: {test_prompt}")
    print(f"  DPO 回答: {response[:300]}")


if __name__ == "__main__":
    main()
