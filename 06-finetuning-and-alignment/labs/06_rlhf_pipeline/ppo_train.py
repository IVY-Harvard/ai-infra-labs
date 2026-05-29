"""
PPO 训练：使用 Reward Model 对 SFT 模型进行 RLHF 训练

硬件: 2-4 × H20 (需要同时加载 policy + ref + RM)
使用 TRL 库的 PPOTrainer

前置条件:
    - 已完成 SFT 训练（得到 SFT 模型）
    - 已完成 RM 训练（得到 Reward Model）
"""

import argparse
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

# ============================================================
# 配置
# ============================================================
SFT_MODEL = "Qwen/Qwen2-1.5B"  # 替换为实际 SFT 模型路径
REWARD_MODEL = "./output/reward_model"  # RM 模型路径
OUTPUT_DIR = "./output/ppo_model"
SEED = 42


def create_prompt_dataset(num_prompts=200):
    """创建 PPO 训练用的 prompt 数据集"""
    import random
    random.seed(SEED)

    prompts = [
        "什么是量子计算？",
        "请解释一下机器学习和深度学习的区别。",
        "如何成为一名优秀的程序员？",
        "写一首关于大海的诗。",
        "解释什么是区块链技术。",
        "Python中如何处理异常？",
        "什么是RESTful API？",
        "如何进行有效的时间管理？",
        "解释面向对象编程的三大特性。",
        "什么是Docker容器？",
    ]

    data = []
    for _ in range(num_prompts):
        prompt = random.choice(prompts)
        data.append({"query": prompt})

    return Dataset.from_list(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_model", default=SFT_MODEL)
    parser.add_argument("--reward_model", default=REWARD_MODEL)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    print("=" * 60)
    print("PPO 训练 (RLHF)")
    print(f"Policy/Ref Model: {args.sft_model}")
    print(f"Reward Model: {args.reward_model}")
    print("=" * 60)

    # 1. PPO 配置
    print("\n[1/5] 配置 PPO...")
    ppo_config = PPOConfig(
        model_name=args.sft_model,
        learning_rate=1.41e-5,
        batch_size=64,              # 每个 PPO 步骤的 batch
        mini_batch_size=8,          # PPO 内部 mini-batch
        gradient_accumulation_steps=8,
        ppo_epochs=4,               # 每个 batch 的 PPO 更新轮数
        init_kl_coef=0.2,           # KL 惩罚初始系数
        target_kl=6.0,              # 目标 KL 值
        clip_range=0.2,             # PPO clip 范围
        vf_coef=0.1,                # Value function 系数
        seed=SEED,
        log_with=None,              # 不使用 wandb（demo）
    )

    # 2. 加载 tokenizer
    print("\n[2/5] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # PPO 需要左填充

    # 3. 加载模型
    print("\n[3/5] 加载模型...")
    # Policy Model (with value head)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.sft_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Reference Model (冻结的 SFT 模型)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.sft_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Reward Model
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    reward_model.eval()

    # 4. 准备数据
    print("\n[4/5] 准备 prompt 数据集...")
    dataset = create_prompt_dataset(200)

    def tokenize_fn(example):
        """编码 prompt"""
        encoded = tokenizer(
            example["query"],
            truncation=True,
            max_length=256,
            padding=False,
        )
        return {"input_ids": encoded["input_ids"], "query": example["query"]}

    dataset = dataset.map(tokenize_fn, remove_columns=["query"])
    dataset.set_format("torch")

    # 5. PPO 训练
    print("\n[5/5] 开始 PPO 训练...")
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    generation_kwargs = {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
    }

    # 训练循环
    num_steps = 50  # demo 只训练少量步骤
    for step, batch in enumerate(ppo_trainer.dataloader):
        if step >= num_steps:
            break

        query_tensors = batch["input_ids"]

        # 1. 生成回答
        response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, **generation_kwargs
        )

        # 2. 计算 reward
        rewards = []
        for query_t, response_t in zip(query_tensors, response_tensors):
            # 拼接 query + response
            full_ids = torch.cat([query_t, response_t]).unsqueeze(0)
            attention_mask = torch.ones_like(full_ids)

            with torch.no_grad():
                reward_output = reward_model(
                    input_ids=full_ids.to(reward_model.device),
                    attention_mask=attention_mask.to(reward_model.device),
                )
            reward = reward_output.logits[0, 0]
            rewards.append(reward)

        # 3. PPO 更新
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        # 日志
        if (step + 1) % 5 == 0:
            mean_reward = torch.stack(rewards).mean().item()
            kl = stats.get("objective/kl", 0)
            print(f"  Step {step+1}/{num_steps} | "
                  f"Reward: {mean_reward:.3f} | "
                  f"KL: {kl:.3f} | "
                  f"Loss: {stats.get('ppo/loss/total', 0):.4f}")

    # 保存
    print("\n" + "=" * 60)
    print("PPO 训练完成！")
    print("=" * 60)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"模型已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
