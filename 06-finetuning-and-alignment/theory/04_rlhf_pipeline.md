# 04 - RLHF 全流程

## 为什么需要 RLHF

### SFT 的局限性

SFT（有监督微调）让模型学会了"怎么回答"，但有几个固有问题：

1. **模仿 vs 理解：** 模型模仿数据中的模式，而非真正理解什么是"好的回答"
2. **分布偏移：** 训练时看到的是人写的回答，生成时自己的分布不一样
3. **无法排序：** SFT 不教模型区分"好回答"和"更好的回答"
4. **安全性不足：** 仅靠正面示例无法有效教会模型拒绝有害请求

### RLHF 解决了什么

RLHF 通过人类偏好信号，让模型学会：
- 在多个可能回答中选择更好的那个
- 理解什么样的回答是人类偏好的
- 平衡有帮助性（helpful）和安全性（harmless）
- 避免产生不被期望的行为模式

## InstructGPT 核心思想

### 三阶段训练流程

```
阶段 1: SFT (Supervised Fine-Tuning)
├── 收集人工编写的高质量示范数据
├── 在 GPT-3 上进行有监督微调
└── 得到 SFT 模型

阶段 2: RM (Reward Model) 训练
├── SFT 模型对同一 prompt 生成多个回答
├── 人工标注员对回答排序
├── 训练 Reward Model 学习人类偏好
└── 得到 RM（输入 prompt+response → 标量分数）

阶段 3: RL (PPO 训练)
├── 用 RM 作为奖励信号
├── PPO 算法优化 SFT 模型
├── KL 约束防止偏离太远
└── 得到最终 RLHF 模型
```

### 关键洞察

```
1. 对齐税（Alignment Tax）：
   RLHF 模型在标准 NLP benchmark 上略低于纯 SFT
   但在人类偏好评估中显著更好
   → "对人有用" 比 "答对题" 更重要

2. 数据效率：
   SFT 需要 ~13K 示范数据
   RM 需要 ~33K 对比数据
   RL 阶段使用 ~31K prompts
   → 总共不到 100K 数据就能显著提升

3. 模型大小 vs 对齐：
   1.3B RLHF 模型 > 175B SFT 模型（人类评估）
   → 对齐比单纯堆参数更重要
```

## Reward Model 训练

### 数据格式

```python
# Reward Model 的训练数据是"对比对"
reward_data = {
    "prompt": "解释什么是黑洞",
    "chosen": "黑洞是一个时空区域，其引力极强...(详细准确的回答)",
    "rejected": "黑洞就是太空中一个黑色的洞...(简略不准确的回答)"
}

# 或者排序数据（更细粒度）
ranking_data = {
    "prompt": "...",
    "responses": ["回答A", "回答B", "回答C", "回答D"],
    "ranking": [2, 0, 3, 1]  # A>C>D>B
}
```

### Reward Model 架构

```python
# Reward Model = LLM backbone + 线性 value head
class RewardModel(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model_name)
        self.value_head = nn.Linear(self.backbone.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        # 取最后一个 token 的 hidden state
        last_hidden = outputs.last_hidden_state[:, -1, :]
        reward = self.value_head(last_hidden)
        return reward.squeeze(-1)
```

### 训练目标

Bradley-Terry 模型：

$$\mathcal{L}_{RM} = -\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

其中 $y_w$ 是偏好回答，$y_l$ 是被拒绝回答。

```python
def reward_model_loss(chosen_rewards, rejected_rewards):
    """Bradley-Terry pairwise loss"""
    return -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
```

### 训练要点

```python
# Reward Model 训练配置
rm_training_config = {
    "model": "与 SFT 模型同架构（通常更小）",
    "lr": 1e-5,                    # 较小学习率
    "epochs": 1,                   # 通常只训练 1 epoch
    "batch_size": 64,              # 较大 batch 稳定训练
    "max_length": 2048,            # 与 SFT 一致
    "loss": "bradley_terry",       # 或 margin loss
    
    # 关键注意事项
    "notes": [
        "RM 通常比 policy model 小（如 policy 7B → RM 3B）",
        "训练数据中 chosen/rejected 差距要明显",
        "需要验证 RM 与人类判断的一致性（>70%）",
        "RM 过拟合后会给出 reward hacking 的信号",
    ]
}
```

## PPO 训练

### PPO 算法在 RLHF 中的应用

```
环境设定：
- State: prompt
- Action: 生成的 token 序列（整个 response）
- Reward: RM 给出的分数 - KL 惩罚

目标: max E[R(x,y)] - β·KL(π_θ || π_ref)
      RM奖励最大化    不要偏离原始模型太远
```

### PPO 训练循环

```python
# 简化的 PPO-RLHF 训练循环
class PPOTrainer:
    def __init__(self, policy_model, ref_model, reward_model, tokenizer):
        self.policy = policy_model      # 当前策略（要优化的模型）
        self.ref = ref_model            # 参考模型（SFT 模型，冻结）
        self.rm = reward_model          # Reward Model（冻结）
        self.tokenizer = tokenizer
    
    def train_step(self, prompts):
        # Step 1: 用当前策略生成回答
        responses = self.policy.generate(prompts)
        
        # Step 2: 计算 Reward
        rewards = self.rm(prompts + responses)
        
        # Step 3: 计算 KL 散度惩罚
        policy_logprobs = self.policy.log_prob(prompts, responses)
        ref_logprobs = self.ref.log_prob(prompts, responses)
        kl_penalty = policy_logprobs - ref_logprobs
        
        # Step 4: 最终奖励 = RM奖励 - β*KL
        final_rewards = rewards - self.beta * kl_penalty
        
        # Step 5: PPO 更新
        advantages = self.compute_advantages(final_rewards)
        self.ppo_update(policy_logprobs, advantages)
    
    def ppo_update(self, old_logprobs, advantages):
        """PPO clip 更新"""
        new_logprobs = self.policy.log_prob(...)
        ratio = torch.exp(new_logprobs - old_logprobs)
        
        # Clipped objective
        clip_adv = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages
        loss = -torch.min(ratio * advantages, clip_adv).mean()
        
        loss.backward()
        self.optimizer.step()
```

### 使用 TRL 库实现

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# 配置
ppo_config = PPOConfig(
    model_name="Qwen/Qwen2-7B-SFT",
    learning_rate=1.41e-5,
    batch_size=128,
    mini_batch_size=16,
    gradient_accumulation_steps=8,
    ppo_epochs=4,              # 每批数据的 PPO 更新次数
    kl_penalty="kl",           # KL 惩罚类型
    init_kl_coef=0.2,          # KL 系数 β
    target_kl=6.0,             # 目标 KL 值
    clip_range=0.2,            # PPO clip 范围
    vf_coef=0.1,               # value function 系数
)

# 加载模型
model = AutoModelForCausalLMWithValueHead.from_pretrained("Qwen/Qwen2-7B-SFT")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("Qwen/Qwen2-7B-SFT")

# 创建 trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# 训练循环
for batch in dataloader:
    # 生成
    query_tensors = batch["input_ids"]
    response_tensors = ppo_trainer.generate(query_tensors, max_new_tokens=256)
    
    # 计算 reward
    rewards = reward_model(query_tensors, response_tensors)
    
    # PPO 更新
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
```

## KL 散度约束

### 为什么需要 KL 约束

```
没有 KL 约束时：
- 模型会 "hack" Reward Model（找到 RM 的漏洞得高分）
- 生成退化（重复、无意义但高 reward 的文本）
- 丧失语言能力（灾难性遗忘）

有 KL 约束时：
- 模型在 SFT 基础上温和改进
- 保持流畅的语言生成能力
- 避免 reward hacking
```

### KL 系数的调节

```python
# 自适应 KL 控制（TRL 默认）
class AdaptiveKLController:
    """根据实际 KL 动态调整系数"""
    def __init__(self, init_kl_coef=0.2, target_kl=6.0):
        self.kl_coef = init_kl_coef
        self.target = target_kl
    
    def update(self, current_kl):
        # 如果 KL 太大 → 增大惩罚
        # 如果 KL 太小 → 减小惩罚
        if current_kl > self.target * 1.5:
            self.kl_coef *= 1.5
        elif current_kl < self.target / 1.5:
            self.kl_coef /= 1.5
```

## 工程实现难点

### 1. 显存压力

```
PPO 需要同时维护 4 个模型：
- Policy Model（要训练的）     ~14GB (7B FP16)
- Reference Model（冻结的）    ~14GB
- Reward Model                 ~6-14GB
- Value Head（包含在 Policy 中）~小

总计: 单 7B 训练需要 ~48-56GB
7B PPO 至少需要 1-2 张 H20

解决方案：
- 把 Ref Model 和 RM 放在不同 GPU
- 使用 model offload（CPU offload）
- 减小 RM 大小（如 policy 7B + RM 1.5B）
```

### 2. 训练不稳定

```python
# 常见问题和解决方案
stability_tips = {
    "reward 坍缩": {
        "现象": "所有样本 reward 趋同",
        "原因": "KL 过大或 reward model 过拟合",
        "解决": "减小 KL 系数，检查 RM 质量"
    },
    "reward hacking": {
        "现象": "reward 持续增高但输出质量下降",
        "原因": "模型找到了 RM 的漏洞",
        "解决": "增大 KL 约束，重新训练 RM"
    },
    "KL 爆炸": {
        "现象": "KL 快速增大到很高的值",
        "原因": "学习率太大或 PPO clip 不够",
        "解决": "减小学习率，减小 clip range"
    },
    "梯度爆炸": {
        "现象": "loss 突然 NaN",
        "原因": "数值不稳定",
        "解决": "gradient clipping, 用 BF16"
    },
}
```

### 3. 数据效率低

```
PPO 的数据利用效率较低：
- 每个 batch 需要先生成（推理），再训练
- 生成速度远慢于训练速度
- 大部分时间在等待生成

优化策略：
1. 异步生成 + 训练（vLLM 生成 + 训练并行）
2. 增大 generation batch size
3. 使用更快的推理引擎生成
4. 减少 PPO epochs（4 → 2）
```

### 4. 超参数敏感

```python
# PPO 关键超参数范围（经验值）
ppo_hyperparams = {
    "learning_rate": (1e-6, 5e-5),      # 比 SFT 小一个量级
    "kl_coef": (0.01, 0.5),             # 太小会 hack，太大学不到
    "clip_range": (0.1, 0.3),           # 标准 PPO 范围
    "ppo_epochs": (1, 4),               # 每批数据更新几轮
    "batch_size": (64, 512),            # 越大越稳定
    "generation_length": (128, 512),     # 生成长度
}
```

## RLHF 的完整流水线总结

```
数据准备 (2-4周)
├── 收集 prompt 数据集
├── 人工编写 SFT 示范数据 (~10K)
├── 人工标注偏好对比数据 (~30K)
└── 质量审核

阶段 1: SFT (1-2天训练)
├── 在示范数据上微调 base model
├── 评估基本对话能力
└── 导出 SFT checkpoint

阶段 2: RM 训练 (0.5-1天)
├── 用 SFT 模型生成多样本
├── 人工排序 → 对比对
├── 训练 Reward Model
├── 验证 RM accuracy (>70%)
└── 导出 RM checkpoint

阶段 3: PPO (2-5天)
├── 加载 SFT (policy + ref) + RM
├── PPO 训练循环
├── 监控 reward, KL, loss
├── 定期人工评估
└── 选择最佳 checkpoint

评估与迭代
├── 人工 A/B 测试 vs SFT
├── 安全性评估
├── 能力保持测试
└── 如果效果不好 → 改进 RM → 重新 PPO
```

## 现代 RLHF 的演进

```
InstructGPT (2022): PPO + RM
    ↓ 简化
Constitutional AI (2022): 自我改进 + RLHF
    ↓ 进一步简化
DPO (2023): 直接从偏好数据学习，无需 RM
    ↓ 变体
ORPO/SimPO/KTO (2024): 更简单的对齐方法
    ↓ 回归
Llama-3/Qwen-2 (2024): 大规模用回 PPO/GRPO

趋势：
- 小规模/快速迭代 → DPO 系列
- 大规模/追求极致效果 → 改进版 PPO (如 GRPO)
```
