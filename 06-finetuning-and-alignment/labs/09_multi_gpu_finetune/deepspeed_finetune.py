"""
DeepSpeed 加速微调：使用 ZeRO 优化器进行内存高效训练

启动方式:
    deepspeed --num_gpus 8 deepspeed_finetune.py
    # 或
    accelerate launch --config_file accelerate_ds_config.yaml deepspeed_finetune.py

适用场景:
    - ZeRO-2: LoRA 微调大模型（70B LoRA 8卡）
    - ZeRO-3: 全量微调中大模型（30B 全量 8卡）
"""

import os
import json
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
# DeepSpeed 配置
# ============================================================
DS_CONFIG_ZeRO2 = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "none"},  # H20 显存足够，无需 offload
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True,
    },
    "gradient_accumulation_steps": 2,
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
}

DS_CONFIG_ZeRO3 = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "none"},
        "offload_param": {"device": "none"},
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
}

# ============================================================
# 配置
# ============================================================
MODEL_NAME = "Qwen/Qwen2-7B"
OUTPUT_DIR = "./output/deepspeed_lora"
ZERO_STAGE = 2  # 2 或 3
SEED = 42


def create_training_data(tokenizer, num_samples=5000):
    """创建训练数据"""
    import random
    random.seed(SEED)

    examples = []
    for i in range(num_samples):
        examples.append({
            "messages": [
                {"role": "user", "content": f"这是第{i}条训练数据的问题，请回答。"},
                {"role": "assistant", "content": f"这是第{i}条训练数据的回答，内容是关于AI技术的。"},
            ]
        })

    dataset = Dataset.from_list(examples)
    dataset = dataset.map(
        lambda x: {"text": tokenizer.apply_chat_template(x["messages"], tokenize=False)},
        remove_columns=dataset.column_names,
    )
    return dataset


def save_ds_config(stage: int, output_path: str):
    """保存 DeepSpeed 配置文件"""
    config = DS_CONFIG_ZeRO2 if stage == 2 else DS_CONFIG_ZeRO3
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    return output_path


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if local_rank == 0:
        print("=" * 60)
        print(f"DeepSpeed ZeRO-{ZERO_STAGE} 微调")
        print(f"模型: {MODEL_NAME}")
        print(f"GPU 数量: {world_size}")
        print("=" * 60)

    # 保存 DS 配置
    ds_config_path = save_ds_config(ZERO_STAGE, "./ds_config.json")

    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # LoRA
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    if local_rank == 0:
        model.print_trainable_parameters()

    # 数据
    dataset = create_training_data(tokenizer, 5000)
    split = dataset.train_test_split(test_size=0.05, seed=SEED)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
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
        save_total_limit=3,
        gradient_checkpointing=True,
        # DeepSpeed
        deepspeed=ds_config_path,
        # 其他
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

    trainer.train()

    if local_rank == 0:
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"\n模型已保存到: {OUTPUT_DIR}")

        # 打印 ZeRO stage 区别
        print("\n" + "=" * 60)
        print("DeepSpeed ZeRO Stages 说明:")
        print("-" * 60)
        print("ZeRO-1: 分片优化器状态")
        print("  适用: 全量微调，节省优化器显存")
        print("ZeRO-2: 分片优化器 + 梯度")
        print("  适用: LoRA大模型，推荐用于 8×H20")
        print("ZeRO-3: 分片优化器 + 梯度 + 参数")
        print("  适用: 全量微调大模型，通信开销大")
        print("=" * 60)


if __name__ == "__main__":
    main()
