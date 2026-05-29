# 03 - 微调数据工程

## 数据是微调的灵魂

在 LLM 微调中，数据质量的影响远大于模型大小和训练技巧：

```
数据质量 >> 数据数量 >> 模型大小 >> 训练技巧
```

一个经典案例：LIMA 论文用仅 1000 条高质量数据微调 LLaMA-65B，
效果接近用 52K 数据的 Alpaca 和用 RLHF 的模型。

## 数据格式

### Alpaca 格式（单轮指令）

```json
{
    "instruction": "将以下英文翻译成中文",
    "input": "Hello, how are you?",
    "output": "你好，你好吗？"
}
```

适用场景：简单的指令-回答任务，单轮对话。

### ShareGPT 格式（多轮对话）

```json
{
    "conversations": [
        {"from": "human", "value": "请介绍一下机器学习"},
        {"from": "gpt", "value": "机器学习是人工智能的一个分支..."},
        {"from": "human", "value": "那深度学习和机器学习有什么区别？"},
        {"from": "gpt", "value": "深度学习是机器学习的一个子集..."}
    ]
}
```

适用场景：多轮对话模型，需要上下文理解。

### OpenAI Messages 格式

```json
{
    "messages": [
        {"role": "system", "content": "你是一个helpful的助手"},
        {"role": "user", "content": "什么是量子计算？"},
        {"role": "assistant", "content": "量子计算是利用量子力学原理..."}
    ]
}
```

适用场景：兼容 OpenAI API 格式，最通用。

### 选择格式的建议

```
单轮任务（分类/翻译/摘要） → Alpaca 格式
多轮对话 → ShareGPT 或 Messages 格式
需要 system prompt → Messages 格式
与 LLaMA-Factory 配合 → 都支持，ShareGPT 最方便
```

## Chat Template

### 为什么 Chat Template 至关重要

不同模型使用不同的对话模板，微调时必须与推理时一致：

```python
# Qwen2 的 chat template
<|im_start|>system
你是一个有帮助的助手。<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！有什么可以帮助你的吗？<|im_end|>

# LLaMA-3 的 chat template
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

你是一个有帮助的助手。<|eot_id|><|start_header_id|>user<|end_header_id|>

你好<|eot_id|><|start_header_id|>assistant<|end_header_id|>

你好！有什么可以帮助你的吗？<|eot_id|>
```

### Loss Mask

微调时只在 assistant 的回答部分计算 loss：

```python
# 示例 token 序列
tokens = [system_tokens] + [user_tokens] + [assistant_tokens]
labels = [-100]*len(system_tokens) + [-100]*len(user_tokens) + assistant_tokens

# -100 是 PyTorch CrossEntropyLoss 的 ignore_index
# 只有 assistant 部分参与梯度计算
```

## 数据清洗

### 常见质量问题

```python
quality_issues = {
    "格式错误": "JSON 不合法、字段缺失",
    "空内容": "instruction 或 output 为空",
    "重复数据": "完全相同或高度相似的样本",
    "语言不匹配": "中文任务混入英文回答",
    "长度异常": "回答过短（<10字）或过长（>10000字）",
    "有害内容": "涉及暴力、歧视等不当内容",
    "幻觉数据": "回答包含明显事实错误",
    "格式不规范": "markdown 标记不完整、代码块未闭合",
}
```

### 清洗流水线

```python
import json
import re
from collections import Counter
from typing import List, Dict

class DataCleaner:
    def __init__(self):
        self.stats = Counter()
    
    def clean_pipeline(self, data: List[Dict]) -> List[Dict]:
        """完整清洗流水线"""
        cleaned = []
        for item in data:
            # Step 1: 基础格式检查
            if not self._check_format(item):
                self.stats["format_error"] += 1
                continue
            
            # Step 2: 内容长度过滤
            if not self._check_length(item):
                self.stats["length_error"] += 1
                continue
            
            # Step 3: 文本规范化
            item = self._normalize(item)
            
            # Step 4: 语言检测
            if not self._check_language(item):
                self.stats["language_mismatch"] += 1
                continue
            
            cleaned.append(item)
        
        # Step 5: 去重
        cleaned = self._dedup(cleaned)
        
        return cleaned
    
    def _check_format(self, item):
        """检查必需字段"""
        if "conversations" in item:
            return len(item["conversations"]) >= 2
        elif "instruction" in item:
            return bool(item.get("output"))
        return False
    
    def _check_length(self, item, min_len=10, max_len=8000):
        """回答长度检查"""
        if "conversations" in item:
            for conv in item["conversations"]:
                if conv["from"] == "gpt" and len(conv["value"]) < min_len:
                    return False
        elif "output" in item:
            if len(item["output"]) < min_len:
                return False
        return True
    
    def _normalize(self, item):
        """文本规范化"""
        def normalize_text(text):
            # 统一空白符
            text = re.sub(r'\s+', ' ', text).strip()
            # 统一引号
            text = text.replace('"', '"').replace('"', '"')
            text = text.replace(''', "'").replace(''', "'")
            return text
        
        if "conversations" in item:
            for conv in item["conversations"]:
                conv["value"] = normalize_text(conv["value"])
        return item
    
    def _check_language(self, item):
        """简单的语言检测"""
        # 这里用中文字符比例简单判断
        return True  # 实际项目中使用 langdetect 或 fasttext
    
    def _dedup(self, data):
        """基于内容 hash 去重"""
        seen = set()
        unique = []
        for item in data:
            content = json.dumps(item, ensure_ascii=False, sort_keys=True)
            content_hash = hash(content)
            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(item)
            else:
                self.stats["duplicates"] += 1
        return unique
```

## 数据配比

### 多任务微调的配比策略

```python
# 错误做法：简单拼接所有数据
# 问题：某类数据量大的任务会 dominate 训练

# 正确做法：按比例采样
data_mix = {
    "general_chat": 0.30,      # 通用对话
    "coding": 0.20,            # 代码生成
    "math": 0.15,              # 数学推理
    "writing": 0.15,           # 写作创作
    "knowledge_qa": 0.10,      # 知识问答
    "safety": 0.10,            # 安全相关
}

# 每个 epoch 按比例从各类数据中采样
# 数据量少的类别会被过采样（重复使用）
```

### 配比原则

1. **能力保持：** 保留 10-20% 通用数据防止遗忘
2. **重点突出：** 核心能力数据占比最高
3. **安全数据：** 始终保持 5-10% 安全对齐数据
4. **渐进调整：** 先平均配比，根据评估结果调整

## 数据质量 > 数据数量

### 经验数据

| 数据量 | 质量 | 效果 |
|--------|------|------|
| 1K 高质量 | ★★★★★ | 很好（LIMA 证明） |
| 10K 中等质量 | ★★★ | 好 |
| 100K 低质量 | ★★ | 一般（可能有害） |
| 1M 混合质量 | ★★~★★★ | 取决于过滤 |

### 质量评估维度

```python
quality_dimensions = {
    "准确性": "回答事实是否正确",
    "完整性": "是否充分回答了问题",
    "相关性": "回答是否切题",
    "清晰度": "表达是否清楚流畅",
    "安全性": "是否包含不当内容",
    "格式规范": "格式是否整洁统一",
    "独特性": "是否与其他样本高度重复",
}

# 简单的质量打分（可用 GPT-4 自动评分）
def score_quality(item):
    prompt = f"""
    请对以下问答对的质量打分（1-5分）：
    
    问题: {item['instruction']}
    回答: {item['output']}
    
    评分维度：准确性、完整性、相关性、清晰度
    请返回 JSON: {{"score": X, "reason": "..."}}
    """
    # 调用评分模型...
```

## 合成数据生成

### Self-Instruct

核心思想：用少量种子数据 + LLM 生成更多指令数据。

```python
class SelfInstructGenerator:
    """Self-Instruct 数据生成"""
    
    def __init__(self, model_client, seed_tasks: List[Dict]):
        self.client = model_client
        self.seed_tasks = seed_tasks
        self.generated = []
    
    def generate_instruction(self, num_samples=5):
        """生成新指令"""
        # 从种子中随机选几个作为示例
        examples = random.sample(self.seed_tasks + self.generated, 
                                min(3, len(self.seed_tasks)))
        
        prompt = "以下是一些任务指令的示例：\n\n"
        for i, ex in enumerate(examples):
            prompt += f"{i+1}. {ex['instruction']}\n"
        prompt += f"\n请生成{num_samples}个新的、不同类型的任务指令："
        
        response = self.client.generate(prompt)
        new_instructions = self._parse_instructions(response)
        return new_instructions
    
    def generate_response(self, instruction):
        """为指令生成回答"""
        prompt = f"请为以下指令提供高质量的回答：\n\n指令：{instruction}\n\n回答："
        response = self.client.generate(prompt)
        return response
    
    def filter_quality(self, item):
        """质量过滤"""
        # 长度过滤
        if len(item["output"]) < 20:
            return False
        # 与已有数据去重（ROUGE-L 相似度）
        for existing in self.seed_tasks + self.generated:
            if rouge_l(item["instruction"], existing["instruction"]) > 0.7:
                return False
        return True
```

### Evol-Instruct (WizardLM)

核心思想：通过"进化"让简单指令变复杂：

```python
evolution_strategies = {
    "深化": "在原指令基础上增加约束条件和难度",
    "具体化": "将抽象指令变为具体场景",
    "增加推理": "要求解释原因或提供推理过程",
    "多步骤": "将单步任务改为需要多步完成",
    "反转": "从回答出发，创建能产生该回答的新问题",
}

def evolve_instruction(instruction, strategy="深化"):
    """指令进化"""
    prompts = {
        "深化": f"""
请将以下指令改写得更复杂、更有挑战性。
可以增加约束条件、要求更详细的分析、或引入边界情况。

原始指令: {instruction}

进化后的指令:""",
        "具体化": f"""
请将以下抽象指令改写为一个具体的真实场景。

原始指令: {instruction}

具体化后的指令:""",
    }
    return model.generate(prompts[strategy])
```

### 合成数据的注意事项

```
✓ 应该做的：
- 人工验证采样（至少检查 5-10% 样本）
- 多样性检查（避免模式坍缩）
- 混合真实数据使用（合成 : 真实 = 3:1 ~ 1:1）
- 迭代过滤低质量样本

✗ 不应该做的：
- 不验证直接使用全部合成数据
- 用同一个 prompt 模板生成所有数据
- 忽略合成数据中的幻觉
- 合成数据比例过高（>80%）
```

## 数据工程最佳实践

### 数据管理

```python
# 推荐的数据目录结构
"""
data/
├── raw/                    # 原始数据
│   ├── source_a.jsonl
│   └── source_b.jsonl
├── cleaned/                # 清洗后
│   ├── cleaned_v1.jsonl
│   └── cleaning_report.json
├── processed/              # 处理后（tokenized）
│   └── train_processed/
├── splits/                 # 划分
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
└── metadata/               # 元信息
    ├── statistics.json
    └── quality_scores.json
"""
```

### 数据版本化

```python
# 使用 datasets 库管理数据版本
from datasets import Dataset, DatasetDict

# 保存为 Arrow 格式（高效读取）
dataset = Dataset.from_list(data_list)
dataset.save_to_disk("./data/v1.0")

# 或使用 HuggingFace Hub
dataset.push_to_hub("my-org/my-dataset", private=True)
```

### 数据增强技巧

```python
augmentation_techniques = {
    "回译": "中→英→中，增加表达多样性",
    "同义改写": "用 LLM 改写指令，保持语义",
    "角色互换": "将 user/assistant 调换生成新样本",
    "难度分层": "同一问题生成简单/中等/困难三个版本的回答",
    "格式变换": "将列表改为段落，将段落改为表格",
    "错误注入": "故意加入错误让模型学习纠错",
}
```

## 典型数据量推荐

| 任务类型 | 推荐数据量 | 说明 |
|---------|-----------|------|
| 格式适配 | 100-500 | 教模型新的输出格式 |
| 风格迁移 | 500-2K | 改变回答风格 |
| 指令遵循 | 5K-50K | 通用指令微调 |
| 领域专家 | 10K-100K | 注入领域知识 |
| 多轮对话 | 10K-50K | 提升对话能力 |
| 代码生成 | 50K-200K | 代码能力需要大量数据 |
