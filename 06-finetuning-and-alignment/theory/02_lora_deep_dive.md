# 02 - LoRA 深度解析

## 从直觉到数学

### 核心直觉

大模型微调时，权重的变化量（ΔW）虽然维度很高，但实际上"有效维度"很低。
也就是说，ΔW 可以用一个低秩矩阵很好地近似。

类比：一张 1000×1000 的图片看似有 100 万像素自由度，但如果图片是纯色渐变，
用几个参数就能完美描述它。微调的权重更新也是类似的"结构化"变化。

### 数学表达

对于预训练权重 $W_0 \in \mathbb{R}^{d \times k}$，标准微调更新为：

$$W = W_0 + \Delta W$$

LoRA 的约束：$\Delta W = BA$，其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$

前向传播变为：

$$h = W_0 x + \frac{\alpha}{r} BAx$$

其中 $\alpha$ 是缩放因子。

### 初始化策略

```python
# LoRA 的初始化保证训练开始时 ΔW = 0
A = nn.Linear(k, r, bias=False)  # 随机高斯初始化
B = nn.Linear(r, d, bias=False)  # 零初始化

nn.init.kaiming_uniform_(A.weight, a=math.sqrt(5))
nn.init.zeros_(B.weight)

# 这样 BA = 0，训练开始时模型行为 = 原始模型
```

## Rank（秩）的选择

### Rank 对效果的影响

```
Rank 太小 (r=1~4):
- 表达能力不足，欠拟合
- 适合极简单的格式适配任务

Rank 适中 (r=8~64):
- 大多数 SFT 任务的甜蜜区
- r=16 是最常见的默认值
- r=64 对于复杂任务通常够用

Rank 较大 (r=128~256):
- 接近全量微调的效果
- 显存和计算量显著增加
- 适合需要大量新知识注入的场景
```

### Rank 选择经验法则

| 任务类型 | 推荐 Rank | 原因 |
|---------|----------|------|
| 格式/风格适配 | 8-16 | 改变简单，低秩足矣 |
| 指令遵循 SFT | 16-64 | 中等复杂度 |
| 领域知识注入 | 64-128 | 需要学习新信息 |
| 多语言能力 | 128-256 | 跨语言映射复杂 |
| 代码生成 | 32-64 | 结构化但模式有限 |

### 实验验证 Rank 的影响

```python
# 不同 rank 训练对比实验设计
ranks = [4, 8, 16, 32, 64, 128]
results = {}

for r in ranks:
    config = LoraConfig(r=r, lora_alpha=2*r, ...)
    # 训练...
    results[r] = {
        "train_loss": ...,
        "eval_loss": ...,
        "trainable_params": 2 * r * d * num_layers,
        "training_time": ...,
    }

# 通常你会看到：
# - r=4 → 8：提升显著
# - r=16 → 32：提升明显
# - r=64 → 128：提升边际递减
# - r=128 → 256：几乎无提升（说明已收敛到满秩效果）
```

## Alpha 参数的作用

### 缩放机制

LoRA 的实际权重更新为：

$$\Delta W = \frac{\alpha}{r} \cdot BA$$

`alpha/r` 是一个缩放系数，控制 LoRA 更新的幅度。

### Alpha 与学习率的关系

```python
# 等价关系：
# alpha=16, r=16, lr=2e-4
# ≈ alpha=32, r=16, lr=1e-4  (scale加倍，lr减半)
# ≈ alpha=16, r=8, lr=4e-4   (r减半→scale加倍，lr减半)

# 实践中的简化规则：
# 方案 1: alpha = r（scale = 1），调整 lr
# 方案 2: alpha = 2*r（scale = 2），使用标准 lr
# 方案 3: 固定 alpha=16，调整 r（Hugging Face 的风格）
```

### 推荐做法

```python
# 最简单且有效的规则
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,      # alpha = 2 * r
    # 或者
    # lora_alpha=64,     # alpha = r
)

# 如果 rank 变了，alpha 等比例变化，这样有效学习率不变
```

## Target Modules 选择

### Transformer 中的可选模块

```
一个标准 Transformer 层包含：
├── Self-Attention
│   ├── q_proj (Query 投影)     ← 常用
│   ├── k_proj (Key 投影)       ← 常用
│   ├── v_proj (Value 投影)     ← 常用
│   └── o_proj (Output 投影)    ← 常用
├── MLP (Feed-Forward)
│   ├── gate_proj              ← 推荐加入
│   ├── up_proj                ← 推荐加入
│   └── down_proj              ← 推荐加入
└── LayerNorm（通常不加 LoRA）
```

### 不同 target modules 的效果

```python
# 最小集合（效果一般，速度最快）
target_modules = ["q_proj", "v_proj"]

# 标准集合（推荐起始点）
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# 完整集合（效果最好）
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# 注意：不同模型的模块名不同
# LLaMA/Qwen: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
# GPT-NeoX: query_key_value, dense, dense_h_to_4h, dense_4h_to_h
# Bloom: query_key_value, dense, dense_h_to_4h, dense_4h_to_h
```

### 参数量计算

```python
# 以 Qwen2-7B 为例 (hidden_dim=4096, intermediate=11008, layers=32)
# Attention 模块:
#   q, k, v, o: 4 × (4096 × 4096) = 4 × 16M = 64M params/layer

# MLP 模块:
#   gate, up: 2 × (4096 × 11008) = 2 × 45M = 90M params/layer
#   down: 11008 × 4096 = 45M params/layer

# LoRA 参数量 (r=64, all modules):
#   每个线性层: 2 × 64 × dim
#   Attention: 4 × 2 × 64 × 4096 = 2M / layer
#   MLP: 3 × 2 × 64 × ~7500 (avg) ≈ 2.9M / layer
#   总计: 32 layers × (2M + 2.9M) ≈ 157M ≈ 2.2% of 7B

def count_lora_params(model, lora_config):
    """计算 LoRA 参数量"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    print(f"Total: {total:,}")
```

## LoRA 合并推理

### 合并过程

```python
from peft import PeftModel

# 加载基座 + LoRA
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B")
model = PeftModel.from_pretrained(base_model, "./lora_output")

# 合并权重: W_merged = W_0 + (alpha/r) * BA
merged_model = model.merge_and_unload()

# 保存合并后的完整模型
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")

# 合并后：
# - 模型大小 = 原始模型大小
# - 推理速度 = 原始模型（无额外开销）
# - 不再需要 PEFT 库
# - 可直接用 vLLM/TGI 部署
```

### 多 LoRA 服务

```python
# 不合并的好处：一个基座加载多个 LoRA
# vLLM 支持运行时动态加载 LoRA

# vLLM 多 LoRA 服务
# vllm serve Qwen/Qwen2-7B \
#   --enable-lora \
#   --lora-modules customer_service=./lora_cs medical=./lora_med

# 请求时指定 LoRA:
# {"model": "customer_service", "messages": [...]}
```

## QLoRA 详解

### NF4 量化

QLoRA 使用 NormalFloat 4-bit (NF4) 量化，专为正态分布权重设计：

```
传统 INT4: 均匀分布 16 个量化点
NF4: 量化点按正态分布 CDF 分布（权重是近似正态的）

NF4 比 INT4 信息损失更小，因为量化点分布匹配数据分布
```

### 双重量化 (Double Quantization)

```python
# 普通量化: 每 64 个参数一个 FP32 scale factor
# 内存开销: 32/64 = 0.5 bit/param 额外

# 双重量化: 把 scale factors 再量化一次（用 FP8）
# 内存开销: 8/64 + 32/(64*256) ≈ 0.127 bit/param 额外

# 配置
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NF4 量化
    bnb_4bit_use_double_quant=True,       # 双重量化
    bnb_4bit_compute_dtype=torch.bfloat16 # 计算时用 BF16
)
```

### QLoRA 完整配置

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# 准备模型（处理量化层的梯度）
model = prepare_model_for_kbit_training(model)

# LoRA 配置
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
model.print_trainable_parameters()
# 输出: trainable params: 157,286,400 || all params: 7,615,616,000 || trainable%: 2.065%
```

### QLoRA 的精度链

```
存储: NF4 (4-bit) ──反量化──→ BF16 ──计算──→ BF16 输出
                                         ↑
                                    LoRA: BF16 全程
                                         │
                                  梯度只流过 LoRA 参数
```

## DoRA (Weight-Decomposed Low-Rank Adaptation)

### 原理

DoRA 将权重分解为幅度（magnitude）和方向（direction）两个分量：

$$W = m \cdot \frac{W_0 + BA}{\|W_0 + BA\|_c}$$

其中 $m$ 是可学习的幅度向量，方向部分通过 LoRA 更新。

### 与 LoRA 的区别

```
LoRA:  W = W_0 + BA （直接加法更新）
DoRA:  W = m · normalize(W_0 + BA) （分离幅度和方向）

直觉：
- LoRA 同时改变权重的大小和方向
- DoRA 让方向和大小独立学习
- 类似于 Weight Normalization 的思想

效果：
- 相同 rank 下，DoRA 通常优于 LoRA 1-2%
- 代价：训练速度略慢（多一步归一化）
```

### DoRA 使用

```python
from peft import LoraConfig

# PEFT 库已支持 DoRA
config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    use_dora=True,  # 启用 DoRA
    task_type="CAUSAL_LM",
)
```

## LoRA 变体总结

| 方法 | 核心改进 | 适用场景 |
|------|---------|---------|
| LoRA | 低秩分解 | 通用 |
| QLoRA | 4-bit 基座 + LoRA | 显存受限 |
| DoRA | 幅度-方向分离 | 追求更好效果 |
| LoRA+ | A 和 B 不同学习率 | 提升收敛速度 |
| rsLoRA | 改进缩放为 1/√r | 大 rank 更稳定 |
| PiSSA | SVD 初始化 LoRA | 更好的初始化 |
| GaLore | 梯度低秩投影 | 全参训练减内存 |

## 实践建议总结

```yaml
# 推荐的 LoRA 起始配置
model: "Qwen/Qwen2-7B"
method: "LoRA"

lora:
  r: 64
  alpha: 128                  # alpha = 2 * r
  target_modules: "all"       # q,k,v,o + gate,up,down
  dropout: 0.05
  bias: "none"

training:
  lr: 2e-4
  scheduler: "cosine"
  warmup_ratio: 0.03
  epochs: 3
  batch_size: 4
  gradient_accumulation: 4    # effective batch = 16
  max_seq_length: 2048
  bf16: true
  gradient_checkpointing: true

# 如果效果不够好：
# 1. 先增加 rank (64 → 128)
# 2. 检查数据质量
# 3. 尝试 DoRA
# 4. 考虑全量微调
```
