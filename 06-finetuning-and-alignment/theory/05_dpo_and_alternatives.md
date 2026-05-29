# 05 - DPO 及替代方案

## DPO 的动机

### RLHF (PPO) 的痛点

```
1. 训练复杂度高:
   需要维护 4 个模型（policy, ref, RM, value head）
   
2. 调参困难:
   PPO 超参数敏感（KL, clip, lr 相互耦合）
   
3. 训练不稳定:
   reward hacking, KL 爆炸, 梯度不稳定
   
4. 计算开销大:
   生成 + 训练交替，GPU 利用率低
```

### DPO 的核心洞察

DPO 论文的关键发现：**可以从 RLHF 的目标函数中推导出一个封闭解，
将 RL 问题转化为简单的分类问题。**

```
RLHF 目标: max E[r(x,y)] - β·KL(π||π_ref)
    ↓ 推导最优策略的封闭形式
最优策略: π*(y|x) = π_ref(y|x) · exp(r(x,y)/β) / Z(x)
    ↓ 反解 reward
隐式 reward: r(x,y) = β · log(π(y|x)/π_ref(y|x)) + C
    ↓ 代入 Bradley-Terry 模型
DPO Loss: -log σ(β · [log π(yw|x)/π_ref(yw|x) - log π(yl|x)/π_ref(yl|x)])
```

## DPO 原理

### 数学推导

给定偏好数据 $(x, y_w, y_l)$，其中 $y_w$ 是偏好回答，$y_l$ 是被拒绝回答：

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \cdot \left(\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right)\right]$$

直觉理解：
- 让模型增加生成 $y_w$ 的概率（相对于 ref model）
- 同时降低生成 $y_l$ 的概率（相对于 ref model）
- β 控制偏离程度（类似 RLHF 中的 KL 系数）

### 训练流程

```python
import torch
import torch.nn.functional as F

def dpo_loss(
    policy_chosen_logps,    # π_θ(y_w|x) 的 log prob
    policy_rejected_logps,  # π_θ(y_l|x) 的 log prob
    ref_chosen_logps,       # π_ref(y_w|x) 的 log prob
    ref_rejected_logps,     # π_ref(y_l|x) 的 log prob
    beta=0.1,               # 温度参数
):
    """DPO loss 计算"""
    # 计算 log ratio
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps
    
    # DPO loss
    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -F.logsigmoid(logits).mean()
    
    # 用于监控的指标
    chosen_rewards = beta * chosen_logratios.detach()
    rejected_rewards = beta * rejected_logratios.detach()
    reward_margin = (chosen_rewards - rejected_rewards).mean()
    
    return loss, chosen_rewards.mean(), rejected_rewards.mean(), reward_margin
```

### 使用 TRL 实现 DPO

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# 加载模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-SFT")
ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-SFT")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-SFT")

# DPO 配置
dpo_config = DPOConfig(
    output_dir="./dpo_output",
    beta=0.1,                          # 关键参数
    learning_rate=5e-7,                # DPO 需要很小的学习率
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,                # 通常 1 epoch 足够
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    max_length=2048,
    max_prompt_length=1024,
    loss_type="sigmoid",               # 标准 DPO
)

# 数据格式
# {"prompt": "...", "chosen": "...", "rejected": "..."}
dataset = load_dataset("json", data_files="preference_data.jsonl")

# 训练
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)
trainer.train()
```

## DPO 的关键参数

### Beta (β) 的选择

```python
# β 的作用：控制模型偏离 reference model 的程度
# β 越大 → 更保守，更接近 SFT 模型
# β 越小 → 更激进，更倾向于放大偏好差异

beta_guide = {
    0.01: "非常激进，可能导致过拟合偏好数据",
    0.05: "较激进，适合高质量偏好数据",
    0.1:  "标准值，大多数情况的起点",
    0.2:  "较保守，适合噪声较大的数据",
    0.5:  "非常保守，几乎不动",
}

# 实践建议：从 β=0.1 开始，根据效果调整
```

### 学习率

```python
# DPO 的学习率通常比 SFT 小一个量级
# SFT:  2e-4 ~ 2e-5
# DPO:  5e-7 ~ 5e-6

# 原因：DPO 直接优化 log probability ratio
# 过大的学习率会导致 chosen 和 rejected 的概率同时坍缩
```

## DPO 变体

### IPO (Identity Preference Optimization)

```python
# IPO 解决 DPO 的过拟合问题
# DPO loss 在 margin 很大时梯度趋于 0（不继续学习）
# IPO 使用正则化的 loss

def ipo_loss(chosen_logratios, rejected_logratios, beta=0.1):
    """IPO loss - 更稳定的训练"""
    logits = chosen_logratios - rejected_logratios
    loss = (logits - 1/(2*beta))**2
    return loss.mean()
```

### ORPO (Odds Ratio Preference Optimization)

```python
# ORPO 的创新：不需要 reference model！
# 直接在 SFT loss 中加入偏好约束

def orpo_loss(
    policy_chosen_logps,     # 模型对 chosen 的 log prob
    policy_rejected_logps,   # 模型对 rejected 的 log prob
    sft_loss,               # 标准的 SFT loss（在 chosen 上）
    lambda_orpo=1.0,
):
    """ORPO loss = SFT + 偏好"""
    # Odds ratio
    chosen_odds = torch.exp(policy_chosen_logps) / (1 - torch.exp(policy_chosen_logps))
    rejected_odds = torch.exp(policy_rejected_logps) / (1 - torch.exp(policy_rejected_logps))
    
    odds_ratio = chosen_odds / rejected_odds
    preference_loss = -torch.log(torch.sigmoid(torch.log(odds_ratio))).mean()
    
    return sft_loss + lambda_orpo * preference_loss

# 优势：
# - 不需要 reference model（省一半显存）
# - SFT 和偏好学习同时进行（一阶段训练）
# - 训练更稳定
```

### SimPO (Simple Preference Optimization)

```python
# SimPO：简化的偏好优化，使用长度归一化的 log prob
# 论文发现：用 average log prob 比 sum log prob 更好

def simpo_loss(
    policy_chosen_logps,      # sum of log probs for chosen
    policy_rejected_logps,    # sum of log probs for rejected
    chosen_length,            # chosen response length
    rejected_length,          # rejected response length
    beta=2.0,
    gamma=0.5,                # margin term
):
    """SimPO loss - 无需 reference model"""
    # 长度归一化
    chosen_rewards = policy_chosen_logps / chosen_length
    rejected_rewards = policy_rejected_logps / rejected_length
    
    # 带 margin 的 loss
    logits = beta * (chosen_rewards - rejected_rewards) - gamma
    loss = -F.logsigmoid(logits).mean()
    return loss

# 优势：
# - 无需 reference model
# - 引入 margin 防止 collapse
# - 长度归一化解决长度偏见
```

### KTO (Kahneman-Tversky Optimization)

```python
# KTO：不需要成对的偏好数据！
# 只需要每条数据标记为 "好" 或 "坏"

def kto_loss(
    policy_logps,          # 模型的 log prob
    ref_logps,             # reference model 的 log prob
    is_desirable,          # True/False 标签
    beta=0.1,
):
    """KTO loss - 只需要二元标签"""
    logratios = policy_logps - ref_logps
    kl = logratios.mean()  # 近似 KL
    
    losses = []
    for logratio, desirable in zip(logratios, is_desirable):
        if desirable:
            # 好的回答 → 增加概率
            loss = 1 - torch.sigmoid(beta * (logratio - kl))
        else:
            # 坏的回答 → 降低概率
            loss = 1 - torch.sigmoid(beta * (kl - logratio))
        losses.append(loss)
    
    return torch.stack(losses).mean()

# 优势：
# - 数据要求最低（不需要配对）
# - 可以利用现有的好/坏标注数据
# - 适合标注成本高的场景
```

## 方法对比

| 方法 | 需要 Ref Model | 数据格式 | 训练复杂度 | 效果 |
|------|--------------|---------|-----------|------|
| PPO (RLHF) | 是(+RM) | prompts + RM | ★★★★★ | ★★★★★ |
| DPO | 是 | 偏好对 | ★★ | ★★★★ |
| IPO | 是 | 偏好对 | ★★ | ★★★★ |
| ORPO | 否 | 偏好对 | ★ | ★★★½ |
| SimPO | 否 | 偏好对 | ★ | ★★★★ |
| KTO | 是 | 二元标签 | ★★ | ★★★½ |

### 选型建议

```
追求最佳效果 + 有充足资源:
→ PPO (RLHF)，特别是需要复杂 reward shaping 时

快速迭代 + 有配对偏好数据:
→ DPO（最成熟，工具链完善）

显存有限 + 不想维护两个模型:
→ SimPO 或 ORPO

没有配对数据，只有好/坏标签:
→ KTO

从 SFT 直接到对齐（一步到位）:
→ ORPO
```

## 偏好数据构造

### 数据来源

```python
preference_data_sources = {
    "人工标注": {
        "方式": "标注员对比两个回答，选择更好的",
        "质量": "最高",
        "成本": "非常高（$10-50/小时标注员）",
        "规模": "通常 10K-50K 对",
    },
    "AI 辅助标注": {
        "方式": "用 GPT-4 评判两个回答的优劣",
        "质量": "较高（与人类一致性 ~80%）",
        "成本": "中等（API 费用）",
        "规模": "可达 100K+ 对",
    },
    "隐式反馈": {
        "方式": "用户选择/点赞/重新生成等行为",
        "质量": "有噪声",
        "成本": "极低（需要产品数据）",
        "规模": "可达百万级",
    },
    "拒绝采样 (Best-of-N)": {
        "方式": "生成 N 个回答，用 RM 选最好和最差",
        "质量": "取决于 RM 质量",
        "成本": "计算成本（N 次推理）",
        "规模": "可大规模生成",
    },
}
```

### 构造高质量偏好对

```python
class PreferenceDataBuilder:
    """偏好数据构造器"""
    
    def build_from_generations(self, prompt, model, n_samples=8):
        """生成多个回答，构造偏好对"""
        responses = []
        for _ in range(n_samples):
            resp = model.generate(prompt, temperature=0.8, top_p=0.9)
            responses.append(resp)
        
        # 方法 1: 用 GPT-4 评分
        scores = self.score_with_gpt4(prompt, responses)
        
        # 选择最好和最差
        best_idx = scores.index(max(scores))
        worst_idx = scores.index(min(scores))
        
        # 确保分差足够大
        if scores[best_idx] - scores[worst_idx] >= 2:
            return {
                "prompt": prompt,
                "chosen": responses[best_idx],
                "rejected": responses[worst_idx],
            }
        return None
    
    def build_safety_pairs(self, harmful_prompt):
        """构造安全性偏好对"""
        return {
            "prompt": harmful_prompt,
            "chosen": "抱歉，我无法回答这个问题，因为...",
            "rejected": "[模型生成的有害回答]",
        }
    
    def score_with_gpt4(self, prompt, responses):
        """用 GPT-4 打分"""
        scores = []
        for resp in responses:
            score_prompt = f"""
请对以下回答打分（1-5分）：

问题: {prompt}
回答: {resp}

评分标准：准确性、有帮助、安全性、表达清晰度
请只返回数字分数。
"""
            score = float(self.gpt4_client.generate(score_prompt))
            scores.append(score)
        return scores
```

### 偏好数据的常见问题

```
1. 长度偏见：标注员倾向于选择更长的回答
   → 解决：规范标注指南，加入长度控制

2. 位置偏见：总是选第一个或第二个
   → 解决：随机化展示顺序

3. 难以区分：两个回答质量接近
   → 解决：只保留明显有差异的对（margin > 阈值）

4. 领域不一致：不同标注员对不同领域判断不一致
   → 解决：专业领域用领域专家标注

5. 噪声标签：约 20-30% 的标注存在分歧
   → 解决：多人标注取共识，或使用 confident learning
```

## DPO vs RLHF 的 Trade-off

```
DPO 优势：
✓ 实现简单（几十行代码 vs PPO 的几百行）
✓ 训练稳定（无 reward hacking 问题）
✓ 计算高效（无需生成步骤）
✓ 调参简单（主要就一个 β）

DPO 劣势：
✗ 依赖固定的偏好数据（离线学习）
✗ 无法利用 reward shaping
✗ 对数据质量更敏感
✗ 在某些任务上效果不如 PPO
✗ 无法 online 更新

RLHF (PPO) 优势：
✓ 可以在线学习（不断生成新数据）
✓ 支持复杂的 reward 设计
✓ 理论上限更高
✓ 适合大规模工程化

RLHF (PPO) 劣势：
✗ 训练极其复杂且不稳定
✗ 需要维护多个模型
✗ 超参数敏感
✗ 计算开销大（生成 + 训练交替）
```

## 实践推荐

```yaml
# 对于大多数团队的推荐路径：

第一步: SFT
- 用高质量数据做好基础能力

第二步: DPO
- 构造偏好数据（AI 辅助 + 少量人工验证）
- DPO 训练（简单高效）
- 评估效果

第三步 (可选): 迭代 DPO
- Online DPO: 用当前模型生成 → 评分 → 新偏好对 → 继续 DPO
- 逐步逼近 RLHF 的效果

只有当 DPO 效果到瓶颈时，才考虑转向 PPO
```
