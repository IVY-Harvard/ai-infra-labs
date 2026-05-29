# 01 - 微调全景：方法论与技术选型

## 为什么需要微调

预训练模型（Base Model）通过海量语料学习了通用语言能力，但它们：
- 不会遵循指令格式回答问题
- 无法适应特定领域的专业术语和知识
- 不能按照业务要求的风格输出
- 缺乏安全对齐（可能输出有害内容）

微调（Fine-tuning）通过在特定数据上继续训练，让模型获得这些能力。

## 微调方法全景图

```
                    大模型微调方法
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   全量微调          参数高效微调       对齐训练
  Full FT            PEFT            Alignment
        │               │               │
   所有参数         部分参数/         偏好学习
   都更新          额外参数            │
        │               │         ┌────┼────┐
        │       ┌───┬───┼───┐     │    │    │
        │     LoRA Adapter  │   RLHF  DPO  ORPO
        │           Prefix  │   (PPO)
        │           Tuning  │
        │                   │
        │              QLoRA/DoRA
        │
  需要极大显存           显存友好
```

## 全量微调（Full Fine-tuning）

### 原理

更新模型所有参数，等价于在预训练基础上继续训练：

```python
# 伪代码
for batch in dataloader:
    outputs = model(batch["input_ids"], labels=batch["labels"])
    loss = outputs.loss
    loss.backward()  # 所有参数计算梯度
    optimizer.step()  # 所有参数更新
```

### 显存组成分析

以 7B 模型 FP16 训练为例：

| 组成部分 | 计算方式 | 占用 |
|---------|---------|------|
| 模型参数 (FP16) | 7B × 2 bytes | 14 GB |
| 梯度 (FP16) | 7B × 2 bytes | 14 GB |
| 优化器状态 (AdamW) | 7B × 8 bytes | 56 GB |
| 激活值（取决于 seq_len, batch） | 动态 | 10-30 GB |
| **总计** | | **~100-120 GB** |

> AdamW 需要存储：FP32 参数副本(4B) + 一阶矩(4B) + 二阶矩(4B) = 每参数 12 bytes，
> 但参数本身已经有 FP16 (2B)，所以额外 8B/param 用于优化器。

### 适用场景

- 数据量大（>100K 样本），预算充足
- 需要模型能力发生根本性改变（如学习新语言）
- 可接受训练后模型体积=原模型
- 有足够算力（8×H20 可全量微调 7B-30B）

### 优缺点

| 优点 | 缺点 |
|------|------|
| 效果上限最高 | 显存需求极大 |
| 无信息损失 | 训练速度慢 |
| 实现简单 | 灾难性遗忘风险 |
| 适合大规模能力迁移 | 每个任务一个完整模型 |

## 参数高效微调（PEFT）

### LoRA (Low-Rank Adaptation)

**核心思想：** 权重更新矩阵是低秩的，用两个小矩阵的乘积近似。

```
原始权重 W ∈ R^(d×d)
更新量 ΔW = B × A，其中 B ∈ R^(d×r), A ∈ R^(r×d)
r << d (如 r=8~64, d=4096)

前向传播: h = Wx + BAx
训练参数量: 2 × d × r (远小于 d × d)
```

**显存优势：** 以 7B 模型为例
- 训练参数：约 0.1%~1% 的原模型参数
- 优化器状态随之大幅减少
- 总显存：约 20-35GB（单卡 H20 即可）

### QLoRA

在 LoRA 基础上，将基座模型量化为 4-bit：

```
基座模型：4-bit NF4 量化（7B → ~3.5GB）
LoRA 部分：FP16/BF16 训练
反量化 → 计算 → 只更新 LoRA 参数
```

**显存：** 7B 模型仅需 ~12GB，单张消费级 GPU 即可。

### Adapter Tuning

在 Transformer 层之间插入小型瓶颈网络：

```
原始层输出 → Adapter(Down → ReLU → Up) → 残差连接 → 下一层
Down: d → r
Up:   r → d
```

- 优点：结构清晰，不修改原有权重
- 缺点：增加推理延迟（多了前向计算）

### Prefix Tuning / P-Tuning v2

在输入前添加可学习的虚拟 token：

```
原始输入: [x1, x2, x3, ...]
Prefix:   [p1, p2, ..., pk, x1, x2, x3, ...]
只训练 p1~pk 的 embedding
```

- 优点：参数量极小
- 缺点：效果通常不如 LoRA，占用序列长度

## 方法对比矩阵

| 方法 | 训练参数量 | 7B 显存需求 | 效果 | 推理额外开销 | 多任务支持 |
|------|-----------|-----------|------|------------|-----------|
| Full FT | 100% | ~120GB | ★★★★★ | 无 | 每任务一个模型 |
| LoRA | 0.1-1% | ~30GB | ★★★★ | 可合并消除 | 切换 Adapter |
| QLoRA | 0.1-1% | ~12GB | ★★★½ | 需反量化 | 切换 Adapter |
| Adapter | 1-5% | ~40GB | ★★★½ | 有额外层 | 切换 Adapter |
| Prefix | <0.1% | ~20GB | ★★★ | 占序列长度 | 切换 Prefix |

## 技术选型决策树

```
开始
│
├─ 显存是否充足（>100GB/卡 × 多卡）？
│   ├─ 是 → 数据量是否 > 100K？
│   │       ├─ 是 → 全量微调
│   │       └─ 否 → LoRA（防过拟合）
│   └─ 否 → 进入 PEFT 路线
│
├─ 单卡显存 < 24GB？
│   └─ 是 → QLoRA（唯一选择）
│
├─ 单卡显存 24-96GB？
│   ├─ 追求效果 → LoRA (rank=64-128)
│   └─ 多模型并行 → QLoRA（省显存给推理）
│
├─ 是否需要无推理开销？
│   ├─ 是 → LoRA（可合并）/ Full FT
│   └─ 否 → Adapter 也可以
│
└─ 是否需要在多任务间快速切换？
    ├─ 是 → LoRA（多个小 adapter 文件）
    └─ 否 → 全量微调效果更好
```

## 针对 8×H20 的推荐方案

| 模型规模 | 推荐方案 | 配置要点 |
|---------|---------|---------|
| 7B | LoRA 单卡 | rank=64, batch=8, 1张卡 |
| 7B | Full FT 单卡 | gradient checkpointing, batch=4 |
| 13B | LoRA 单/双卡 | rank=64, 1-2张卡 |
| 30B | LoRA + DeepSpeed | ZeRO-2, 4张卡 |
| 70B | QLoRA 单卡 | 4-bit, rank=64, 1张卡 |
| 70B | LoRA + DeepSpeed | ZeRO-3, 8张卡 |
| 70B | Full FT + FSDP | 无法单机，需多节点 |

## 微调流程总览

```python
# 标准 SFT 微调流程（伪代码）
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig
from trl import SFTTrainer

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")

# 2. 配置 LoRA
lora_config = LoraConfig(
    r=64, lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 3. 训练配置
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
)

# 4. 开始训练
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_seq_length=2048,
)
trainer.train()

# 5. 保存 LoRA 权重
model.save_pretrained("./lora_output")
```

## 关键超参数指南

| 参数 | 全量微调 | LoRA | QLoRA |
|------|---------|------|-------|
| 学习率 | 1e-5 ~ 5e-5 | 1e-4 ~ 3e-4 | 1e-4 ~ 2e-4 |
| Batch Size | 尽可能大 | 4-8/卡 | 4-8/卡 |
| Epochs | 2-3 | 2-5 | 3-5 |
| Warmup Ratio | 0.03-0.1 | 0.03-0.1 | 0.03-0.1 |
| Weight Decay | 0.01 | 0.01 | 0.01 |
| Max Seq Length | 2048-4096 | 2048-4096 | 1024-2048 |
| Gradient Checkpointing | 推荐 | 可选 | 推荐 |

## 常见陷阱

1. **灾难性遗忘：** 全量微调数据太少/学习率太大 → 模型丧失通用能力
2. **过拟合：** 数据量小时用大 rank → 训练 loss 下降但泛化变差
3. **格式不匹配：** 训练时的 prompt 格式和推理时不一致 → 效果大打折扣
4. **tokenizer 不一致：** 微调和推理用了不同 tokenizer → 乱码
5. **BOS/EOS 丢失：** 忘记添加特殊 token → 模型无法正确开始/结束生成

## 下一步

学完本章后：
- → [02_lora_deep_dive.md](02_lora_deep_dive.md) 深入理解 LoRA 原理
- → [labs/01_full_finetuning/](../labs/01_full_finetuning/) 动手跑全量微调 baseline
