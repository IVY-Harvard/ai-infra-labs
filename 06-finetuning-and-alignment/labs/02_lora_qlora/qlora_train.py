"""
QLoRA 微调：4-bit 量化 + LoRA
显存需求极低：7B 模型仅需约 12-15GB

适合场景：
- 显存受限环境
- 70B 模型单卡微调
- 快速实验迭代
"""

import os
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

# ============================================================
# 配置
# ============================================================
MODEL_NAME = "Qwen/Qwen2-7B"
OUTPUT_DIR = "./output/qlora_qwen2_7b"
MAX_SEQ_LENGTH = 2048
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
SEED = 42

# QLoRA 特有配置
QUANT_TYPE = "nf4"              # NormalFloat4，比 INT4 更适合正态分布权重
USE_DOUBLE_QUANT = True         # 双重量化，进一步节省显存
COMPUTE_DTYPE = torch.bfloat16  # 计算时使用的精度

# LoRA 配置
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ============================================================
# 数据准备（同 lora_train.py）
# ============================================================
def load_training_data():
    """创建模拟训练数据"""
    import random
    random.seed(SEED)

    examples = []
    tasks = [
        ("解释机器学习中的过拟合问题", "过拟合是指模型在训练数据上表现很好，但在新数据上表现差的现象。原因包括模型过于复杂、训练数据不足等。解决方法有：正则化、数据增强、早停、Dropout等。"),
        ("Python中列表和元组有什么区别", "列表(list)和元组(tuple)的主要区别：\n1. 可变性：列表可变，元组不可变\n2. 语法：列表用[]，元组用()\n3. 性能：元组略快，占用内存少\n4. 用途：列表用于可变集合，元组用于固定数据"),
        ("什么是RESTful API", "RESTful API是一种基于REST架构风格的Web服务接口设计规范。核心原则包括：\n1. 无状态：每个请求独立\n2. 统一接口：使用HTTP方法(GET/POST/PUT/DELETE)\n3. 资源导向：URL表示资源\n4. 分层系统：客户端无需知道中间层"),
    ]

    for _ in range(1000):
        q, a = random.choice(tasks)
        examples.append({
            "messages": [
                {"role": "system", "content": "你是一个专业的技术导师。"},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        })

    return Dataset.from_list(examples)


def format_dataset(example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("QLoRA 微调: Qwen2-7B (4-bit)")
    print(f"量化: {QUANT_TYPE}, Double Quant: {USE_DOUBLE_QUANT}")
    print(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    print("=" * 60)

    # 1. 量化配置
    print("\n[1/6] 配置 4-bit 量化...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=QUANT_TYPE,
        bnb_4bit_use_double_quant=USE_DOUBLE_QUANT,
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    )

    # 2. 加载量化模型
    print("\n[2/6] 加载 4-bit 量化模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=COMPUTE_DTYPE,
    )

    if torch.cuda.is_available():
        model_mem = torch.cuda.memory_allocated() / 1e9
        print(f"量化模型显存占用: {model_mem:.2f} GB (vs FP16 ~14GB)")

    # 3. 准备量化模型用于训练
    print("\n[3/6] 准备模型用于 k-bit 训练...")
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    # 4. 添加 LoRA
    print("\n[4/6] 配置 LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. 数据
    print("\n[5/6] 准备数据...")
    dataset = load_training_data()
    dataset = dataset.map(
        lambda x: format_dataset(x, tokenizer),
        remove_columns=dataset.column_names,
    )
    split = dataset.train_test_split(test_size=0.05, seed=SEED)

    # 6. 训练
    print("\n[6/6] 开始 QLoRA 训练...")
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
        optim="paged_adamw_8bit",  # QLoRA 推荐: 分页 AdamW 8-bit
        report_to="none",
        seed=SEED,
    )

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
    print("QLoRA 训练完成！")
    print("=" * 60)
    print(f"训练 loss: {train_result.training_loss:.4f}")
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"峰值显存: {peak_mem:.2f} GB")
        print(f"(对比 Full FT 约需 120GB，QLoRA 节省 {120-peak_mem:.0f}GB)")

    # 保存
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nQLoRA 权重已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
