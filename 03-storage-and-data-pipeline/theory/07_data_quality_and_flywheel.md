# 07 — 数据质量与飞轮

## 数据质量决定模型质量

一个被反复验证的事实：**模型质量的上限由数据质量决定**。更大的模型 + 更差的数据，往往不如更小的模型 + 更好的数据。

```
经典案例：
  - Phi-2 (2.7B) 在多项基准上超过 Llama-2-7B
    → 核心优势不是架构，而是高质量 "textbook-quality" 数据
  - LIMA 论文：仅 1000 条高质量指令数据微调
    → 效果接近 GPT-4 时代的 ChatGPT
  - Chinchilla：同样的算力，更多高质量数据 > 更大模型

数据质量的维度：
┌──────────────────┬──────────────────────────────────────┐
│ 维度              │ 说明                                  │
├──────────────────┼──────────────────────────────────────┤
│ 准确性           │ 内容是否事实正确                       │
│ 相关性           │ 是否与目标任务相关                     │
│ 完整性           │ 信息是否完整，有无缺失                 │
│ 一致性           │ 同一事实是否有矛盾表述                 │
│ 及时性           │ 信息是否过时                           │
│ 去重             │ 是否存在重复或近重复内容               │
│ 无毒性           │ 是否包含有害、偏见内容                 │
│ 多样性           │ 是否覆盖足够多的话题和风格             │
└──────────────────┴──────────────────────────────────────┘
```

## 数据去重

### 为什么去重如此重要

```
训练数据中的重复问题：
1. 精确重复：完全相同的文本出现多次
   - 网页爬虫抓到同一页面的多个镜像
   - 数据集拼接时重叠的部分

2. 近似重复：内容高度相似但不完全相同
   - 同一新闻的不同转载（标题/排版微调）
   - 模板化内容（产品描述、法律条款）
   - SEO spam（大量相似页面）

重复数据的危害：
  - 模型记忆而非泛化（过拟合到重复内容）
  - 训练效率下降（浪费算力在重复数据上）
  - 偏差放大（高频内容权重过大）
  - 隐私泄露风险（重复的个人信息更容易被记忆）

去重效果（以 C4 数据集为例）：
  - 原始大小：800GB
  - 精确去重后：~750GB（去除 ~6%）
  - 近似去重后：~600GB（去除 ~25%）
  - 训练效果：去重后模型在下游任务上提升 2-5%
```

### MinHash 去重

MinHash 是大规模近似去重的标准方法，核心思想是通过哈希签名快速判断两篇文档的相似度：

```
原理：
1. 将文档转换为 n-gram 集合
   "the cat sat on the mat" → {"the cat", "cat sat", "sat on", ...}
   
2. 对集合计算多个哈希函数的最小值（MinHash 签名）
   hash_1(set) → min value = 42
   hash_2(set) → min value = 17
   ...
   签名 = [42, 17, 89, 3, ...]

3. 两篇文档的 Jaccard 相似度 ≈ 签名中相同值的比例
   sig_A = [42, 17, 89, 3]
   sig_B = [42, 23, 89, 7]
   相似度 ≈ 2/4 = 0.5

4. LSH（局部敏感哈希）加速：
   将签名分为 b 个 band，每个 band r 行
   只要任何一个 band 完全相同 → 候选对
   大幅减少需要精确比较的文档对数量

复杂度：
  暴力两两比较：O(n²) — 10 亿文档需要 10¹⁸ 次比较
  MinHash + LSH：O(n) — 线性扩展，可处理 PB 级数据
```

```python
import hashlib
import struct
from typing import List, Set, Tuple
import numpy as np


class MinHashDeduplicator:
    """MinHash + LSH 近似去重
    
    参数选择指南：
      num_perm=128, bands=16, rows=8 → 检测 >0.5 相似度
      num_perm=128, bands=32, rows=4 → 检测 >0.3 相似度（更激进）
      num_perm=256, bands=16, rows=16 → 检测 >0.8 相似度（更保守）
    """
    
    def __init__(self, num_perm: int = 128, bands: int = 16, 
                 rows: int = 8, ngram_size: int = 5):
        self.num_perm = num_perm
        self.bands = bands
        self.rows = rows
        self.ngram_size = ngram_size
        
        assert bands * rows == num_perm, \
            f"bands({bands}) * rows({rows}) must equal num_perm({num_perm})"
        
        # 生成随机哈希参数
        rng = np.random.RandomState(42)
        self.hash_a = rng.randint(1, 2**31 - 1, size=num_perm, 
                                   dtype=np.int64)
        self.hash_b = rng.randint(0, 2**31 - 1, size=num_perm, 
                                   dtype=np.int64)
        self.prime = (1 << 31) - 1  # Mersenne prime
        
        # LSH 桶
        self.buckets = [{} for _ in range(bands)]
    
    def _get_ngrams(self, text: str) -> Set[str]:
        """提取字符级 n-gram"""
        text = text.lower().strip()
        tokens = text.split()
        if len(tokens) < self.ngram_size:
            return {text}
        return {
            " ".join(tokens[i:i + self.ngram_size])
            for i in range(len(tokens) - self.ngram_size + 1)
        }
    
    def _compute_minhash(self, ngrams: Set[str]) -> np.ndarray:
        """计算 MinHash 签名"""
        signature = np.full(self.num_perm, np.iinfo(np.int64).max, 
                            dtype=np.int64)
        
        for ngram in ngrams:
            h = int(hashlib.md5(ngram.encode()).hexdigest()[:16], 16)
            hashes = (self.hash_a * h + self.hash_b) % self.prime
            signature = np.minimum(signature, hashes)
        
        return signature
    
    def _lsh_insert(self, doc_id: str, signature: np.ndarray):
        """将文档签名插入 LSH 桶"""
        for band_idx in range(self.bands):
            start = band_idx * self.rows
            end = start + self.rows
            band_hash = hashlib.md5(
                signature[start:end].tobytes()
            ).hexdigest()
            
            if band_hash not in self.buckets[band_idx]:
                self.buckets[band_idx][band_hash] = []
            self.buckets[band_idx][band_hash].append(doc_id)
    
    def _lsh_query(self, signature: np.ndarray) -> Set[str]:
        """查询与给定签名相似的文档"""
        candidates = set()
        for band_idx in range(self.bands):
            start = band_idx * self.rows
            end = start + self.rows
            band_hash = hashlib.md5(
                signature[start:end].tobytes()
            ).hexdigest()
            
            if band_hash in self.buckets[band_idx]:
                candidates.update(self.buckets[band_idx][band_hash])
        
        return candidates
    
    def deduplicate(self, documents: List[Tuple[str, str]],
                    threshold: float = 0.8) -> List[str]:
        """执行去重，返回需要保留的文档 ID
        
        Args:
            documents: [(doc_id, text), ...]
            threshold: Jaccard 相似度阈值
        
        Returns:
            保留的文档 ID 列表
        """
        signatures = {}
        duplicates = set()
        
        for doc_id, text in documents:
            ngrams = self._get_ngrams(text)
            if not ngrams:
                duplicates.add(doc_id)
                continue
            
            sig = self._compute_minhash(ngrams)
            
            # 查询候选重复文档
            candidates = self._lsh_query(sig)
            
            is_dup = False
            for candidate_id in candidates:
                if candidate_id == doc_id:
                    continue
                # 计算精确 Jaccard 相似度（用签名近似）
                sim = np.mean(sig == signatures[candidate_id])
                if sim >= threshold:
                    is_dup = True
                    break
            
            if is_dup:
                duplicates.add(doc_id)
            else:
                signatures[doc_id] = sig
                self._lsh_insert(doc_id, sig)
        
        kept = [doc_id for doc_id, _ in documents 
                if doc_id not in duplicates]
        
        print(f"Total: {len(documents)}, "
              f"Duplicates: {len(duplicates)}, "
              f"Kept: {len(kept)} "
              f"({len(kept)/len(documents)*100:.1f}%)")
        
        return kept
```

### SimHash 去重

```
MinHash vs SimHash：
┌──────────────┬──────────────────┬──────────────────┐
│              │ MinHash          │ SimHash          │
├──────────────┼──────────────────┼──────────────────┤
│ 相似度度量    │ Jaccard 相似度   │ 余弦相似度       │
│ 签名大小      │ 可变（128-256值）│ 固定 64/128 bit │
│ 适合内容      │ 集合型数据       │ 向量型/文本数据  │
│ 内存占用      │ 较大             │ 极小             │
│ 速度          │ 中（LSH加速）   │ 快（位运算）     │
│ 典型应用      │ 网页去重         │ 短文本/文档指纹  │
└──────────────┴──────────────────┴──────────────────┘

SimHash 适合：
  - 内存有限但数据量巨大
  - 文档长度差异不大
  - 需要极快的去重速度
```

## 数据质量打分

### 多维度质量评估

```
质量打分模型的设计思路：
  不用一个分数决定一切，而是多维度打分后加权

维度 1：语言质量（Perplexity）
  用小型语言模型计算困惑度
  低困惑度 = 流畅的自然语言
  高困惑度 = 乱码/OCR错误/机器翻译

维度 2：信息密度
  unique token 比例 / 平均句子长度
  高密度 = 有实质内容
  低密度 = 重复性/模板化内容

维度 3：教育价值
  经过分类器判断内容是否有教育意义
  参考 Phi-2 的 "textbook quality" 标准

维度 4：安全性
  有害内容检测（毒性、偏见、个人信息）
  使用专门的安全分类器

综合分数 = w1×语言质量 + w2×信息密度 + w3×教育价值 + w4×安全性
```

```python
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class QualityScore:
    """数据质量评分结果"""
    language_score: float      # 语言质量 [0, 1]
    info_density: float        # 信息密度 [0, 1]
    education_value: float     # 教育价值 [0, 1]
    safety_score: float        # 安全性 [0, 1]
    overall: float             # 综合分数 [0, 1]
    
    def passes_threshold(self, min_score: float = 0.6) -> bool:
        return self.overall >= min_score


class QualityScorer:
    """数据质量打分器"""
    
    def __init__(self, weights=None):
        self.weights = weights or {
            "language": 0.3,
            "info_density": 0.2,
            "education": 0.3,
            "safety": 0.2,
        }
    
    def score_language_quality(self, text: str) -> float:
        """语言质量评分（基于启发式规则 + 困惑度）"""
        score = 1.0
        
        # 规则 1：过短或过长
        word_count = len(text.split())
        if word_count < 50:
            score *= 0.5
        elif word_count > 100000:
            score *= 0.7
        
        # 规则 2：特殊字符比例
        special_ratio = sum(
            1 for c in text if not c.isalnum() and not c.isspace()
        ) / max(len(text), 1)
        if special_ratio > 0.3:
            score *= 0.3
        
        # 规则 3：重复行比例
        lines = text.strip().split("\n")
        if lines:
            unique_ratio = len(set(lines)) / len(lines)
            score *= unique_ratio
        
        # 规则 4：全大写比例
        upper_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if upper_ratio > 0.5:
            score *= 0.5
        
        return min(max(score, 0.0), 1.0)
    
    def score_info_density(self, text: str) -> float:
        """信息密度评分"""
        words = text.split()
        if not words:
            return 0.0
        
        # 唯一词比例（type-token ratio）
        unique_ratio = len(set(words)) / len(words)
        
        # 平均词长（过短可能是碎片内容）
        avg_word_len = sum(len(w) for w in words) / len(words)
        word_len_score = min(avg_word_len / 6.0, 1.0)
        
        # 句子数量（过少可能是不完整内容）
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        sentence_score = min(len(sentences) / 5.0, 1.0)
        
        return (unique_ratio * 0.4 + word_len_score * 0.3 + 
                sentence_score * 0.3)
    
    def score(self, text: str) -> QualityScore:
        """综合打分"""
        lang = self.score_language_quality(text)
        info = self.score_info_density(text)
        edu = 0.5    # 需要分类器模型，此处用默认值
        safety = 0.9  # 需要安全分类器，此处用默认值
        
        overall = (
            self.weights["language"] * lang +
            self.weights["info_density"] * info +
            self.weights["education"] * edu +
            self.weights["safety"] * safety
        )
        
        return QualityScore(
            language_score=lang,
            info_density=info,
            education_value=edu,
            safety_score=safety,
            overall=overall,
        )
```

## 合成数据生成

### Self-Instruct

Self-Instruct 是一种用模型自身生成指令训练数据的方法：

```
核心流程：
1. 准备种子指令集（175 条人工编写的高质量指令）
2. 从种子集随机采样几条作为 few-shot 示例
3. 让 LLM 生成新的指令
4. 让 LLM 为新指令生成输入和输出
5. 过滤低质量生成（重复、过短、格式错误）
6. 将通过过滤的样本加入种子集
7. 重复 2-6，滚雪球式扩大数据集

效果数据（原始论文）：
  - 种子集：175 条
  - 生成量：52K 条指令 + 回复
  - 过滤后：约 82% 保留率
  - 微调效果接近 InstructGPT（text-davinci-001）
```

### Evol-Instruct

Evol-Instruct 通过进化策略逐步提升指令的难度和多样性：

```
进化策略：
1. 增加约束（Add Constraints）
   原始：写一个排序算法
   进化：写一个排序算法，要求时间复杂度 O(n log n)，空间复杂度 O(1)

2. 深化（Deepen）
   原始：解释什么是机器学习
   进化：解释机器学习中正则化如何防止过拟合，并对比 L1 和 L2 的区别

3. 具体化（Concretize）
   原始：写一个数据处理脚本
   进化：写一个 Python 脚本，从 CSV 文件读取销售数据，计算每月环比增长率

4. 增加推理步骤（Increase Reasoning）
   原始：计算 15% 的 tip
   进化：一桌 6 人吃饭，总账 $247.50，要给 18% 小费并平摊，每人付多少

5. 更换主题（Breadth Evolution）
   原始：（编程领域）实现二叉树遍历
   进化：（医学领域）描述心电图各波段的临床意义

进化循环：
  初始指令集 → 随机选择进化策略 → LLM 执行进化 → 
  质量过滤 → 加入数据集 → 重复
  
  通常迭代 3-4 轮，每轮数据量翻倍
```

```python
from typing import List, Dict
import random


class EvolInstructGenerator:
    """Evol-Instruct 合成数据生成器
    
    需要一个 LLM 推理接口（如 OpenAI API / vLLM）
    """
    
    EVOLUTION_PROMPTS = {
        "add_constraints": (
            "I want you to act as a Prompt Rewriter.\n"
            "Your objective is to rewrite a given prompt into a more "
            "complex version by adding one more constraint/requirement.\n"
            "The rewritten prompt must be reasonable and understandable.\n\n"
            "#Given Prompt#:\n{instruction}\n\n"
            "#Rewritten Prompt#:"
        ),
        "deepen": (
            "I want you to act as a Prompt Rewriter.\n"
            "Your objective is to rewrite a given prompt to make it "
            "more complex by increasing the depth and breadth of inquiry.\n"
            "The rewritten prompt must be reasonable.\n\n"
            "#Given Prompt#:\n{instruction}\n\n"
            "#Rewritten Prompt#:"
        ),
        "concretize": (
            "I want you to act as a Prompt Rewriter.\n"
            "Your objective is to rewrite a given prompt to make it "
            "more specific by replacing general concepts with concrete "
            "examples.\n\n"
            "#Given Prompt#:\n{instruction}\n\n"
            "#Rewritten Prompt#:"
        ),
        "increase_reasoning": (
            "I want you to act as a Prompt Rewriter.\n"
            "Your objective is to rewrite a given prompt to require "
            "multi-step reasoning.\n\n"
            "#Given Prompt#:\n{instruction}\n\n"
            "#Rewritten Prompt#:"
        ),
    }
    
    def __init__(self, llm_generate_fn):
        """
        Args:
            llm_generate_fn: 接受 prompt 返回生成文本的函数
        """
        self.llm_generate = llm_generate_fn
    
    def evolve_instruction(self, instruction: str, 
                           strategy: str = None) -> str:
        """对单条指令执行一次进化"""
        if strategy is None:
            strategy = random.choice(list(self.EVOLUTION_PROMPTS.keys()))
        
        prompt = self.EVOLUTION_PROMPTS[strategy].format(
            instruction=instruction
        )
        evolved = self.llm_generate(prompt)
        return evolved.strip()
    
    def generate_response(self, instruction: str) -> str:
        """为指令生成回复"""
        prompt = (
            f"Below is an instruction. Write a high-quality, "
            f"detailed response.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:"
        )
        return self.llm_generate(prompt).strip()
    
    def run_evolution(
        self,
        seed_instructions: List[str],
        num_rounds: int = 3,
        evolutions_per_round: int = 2,
    ) -> List[Dict[str, str]]:
        """执行多轮进化，生成训练数据"""
        all_data = []
        current_pool = list(seed_instructions)
        
        for round_idx in range(num_rounds):
            new_instructions = []
            
            for instruction in current_pool:
                for _ in range(evolutions_per_round):
                    evolved = self.evolve_instruction(instruction)
                    
                    # 质量过滤
                    if self._quality_check(evolved):
                        response = self.generate_response(evolved)
                        all_data.append({
                            "instruction": evolved,
                            "response": response,
                            "source_instruction": instruction,
                            "round": round_idx,
                        })
                        new_instructions.append(evolved)
            
            current_pool = new_instructions
            print(f"Round {round_idx + 1}: generated {len(new_instructions)} "
                  f"new instructions, total: {len(all_data)}")
        
        return all_data
    
    def _quality_check(self, instruction: str) -> bool:
        """基础质量过滤"""
        if len(instruction) < 10:
            return False
        if len(instruction) > 2000:
            return False
        if instruction.count("#") > 5:
            return False
        return True
```

## 数据飞轮

### 什么是数据飞轮

```
数据飞轮是一个持续改进的闭环：

    ┌──────────────┐
    │  模型训练/更新 │
    └──────┬───────┘
           │ 部署新模型
           ▼
    ┌──────────────┐
    │  模型上线推理  │
    └──────┬───────┘
           │ 收集用户交互
           ▼
    ┌──────────────┐
    │  反馈数据收集  │
    │  用户满意度    │
    │  纠错/改写    │
    └──────┬───────┘
           │ 筛选 + 标注
           ▼
    ┌──────────────┐
    │  数据标注/清洗 │
    │  人工 + 自动   │
    └──────┬───────┘
           │ 高质量训练数据
           ▼
    ┌──────────────┐
    │  模型训练/更新 │ ← 循环回到起点
    └──────────────┘

飞轮效应：
  模型越好 → 用户越多 → 反馈数据越多 → 数据质量越高 → 模型更好
  这就是为什么数据飞轮是 AI 公司的核心竞争壁垒
```

### 飞轮的关键组件

```
组件 1：反馈收集
  - 显式反馈：用户点赞/点踩、评分
  - 隐式反馈：用户是否采纳回答、编辑了多少、是否重新提问
  - 错误日志：模型输出导致的异常/投诉

组件 2：数据筛选
  - 自动筛选：根据反馈信号过滤高质量样本
  - 困难样本挖掘：模型表现差的 case 是最有价值的训练数据
  - 多样性采样：确保不只是简单 case

组件 3：标注流水线
  - Tier 1（自动）：规则 + 模型自动标注 → 覆盖 80% 简单 case
  - Tier 2（众包）：众包平台标注 → 覆盖中等难度
  - Tier 3（专家）：领域专家标注 → 覆盖高难度 case
  
  成本控制：
    Tier 1: $0/sample（自动化）
    Tier 2: $0.1-1/sample（众包）
    Tier 3: $5-50/sample（专家）
    
    通过分层标注将平均成本控制在 $0.5/sample 以下

组件 4：数据版本管理
  - 每次迭代的训练数据都需要版本化
  - 能追溯每条样本的来源（用户反馈 / 人工标注 / 合成生成）
  - 支持数据回滚（如果某批数据引入了问题）
```

### 构建反馈收集系统

```python
import json
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, List


class FeedbackType(Enum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    EDIT = "edit"
    REGENERATE = "regenerate"
    REPORT = "report"


@dataclass
class FeedbackRecord:
    """用户反馈记录"""
    request_id: str
    timestamp: float
    feedback_type: FeedbackType
    user_input: str
    model_output: str
    edited_output: Optional[str] = None
    rating: Optional[int] = None       # 1-5
    tags: Optional[List[str]] = None   # ["factual_error", "incomplete"]
    model_version: str = ""
    
    def to_training_sample(self) -> Optional[dict]:
        """将反馈转换为训练样本"""
        if self.feedback_type == FeedbackType.THUMBS_UP:
            # 正面反馈：直接作为正样本
            return {
                "instruction": self.user_input,
                "response": self.model_output,
                "quality": "positive",
                "source": "user_feedback",
            }
        elif self.feedback_type == FeedbackType.EDIT:
            # 用户编辑了回答：编辑后的版本作为正样本
            if self.edited_output:
                return {
                    "instruction": self.user_input,
                    "response": self.edited_output,
                    "rejected_response": self.model_output,
                    "quality": "edited",
                    "source": "user_edit",
                }
        elif self.feedback_type == FeedbackType.THUMBS_DOWN:
            # 负面反馈：作为 DPO 的负样本
            return {
                "instruction": self.user_input,
                "rejected_response": self.model_output,
                "quality": "negative",
                "source": "user_feedback",
            }
        return None
```

### 标注流水线

```python
from dataclasses import dataclass
from typing import List, Callable, Optional
from enum import Enum


class AnnotationTier(Enum):
    AUTO = "auto"          # 自动标注
    CROWD = "crowd"        # 众包标注
    EXPERT = "expert"      # 专家标注


@dataclass
class AnnotationTask:
    """标注任务"""
    task_id: str
    instruction: str
    model_response: str
    tier: AnnotationTier
    annotation: Optional[dict] = None
    annotator_id: Optional[str] = None


class AnnotationPipeline:
    """分层标注流水线"""
    
    def __init__(self):
        self.auto_annotators: List[Callable] = []
        self.quality_threshold = 0.8
    
    def register_auto_annotator(self, fn: Callable):
        """注册自动标注函数"""
        self.auto_annotators.append(fn)
    
    def process(self, tasks: List[AnnotationTask]) -> List[AnnotationTask]:
        """处理标注任务"""
        auto_tasks = []
        escalated_tasks = []
        
        for task in tasks:
            # Tier 1: 尝试自动标注
            auto_result = self._try_auto_annotate(task)
            
            if auto_result and auto_result["confidence"] >= self.quality_threshold:
                task.annotation = auto_result
                task.tier = AnnotationTier.AUTO
                auto_tasks.append(task)
            else:
                # 升级到人工标注
                task.tier = AnnotationTier.CROWD
                escalated_tasks.append(task)
        
        print(f"Auto-annotated: {len(auto_tasks)}, "
              f"Escalated to human: {len(escalated_tasks)}")
        
        return auto_tasks + escalated_tasks
    
    def _try_auto_annotate(self, task: AnnotationTask) -> Optional[dict]:
        """尝试自动标注"""
        results = []
        for annotator in self.auto_annotators:
            result = annotator(task.instruction, task.model_response)
            if result:
                results.append(result)
        
        if not results:
            return None
        
        # 多个自动标注器投票
        avg_confidence = sum(r.get("confidence", 0) for r in results) / len(results)
        
        return {
            "auto_annotations": results,
            "confidence": avg_confidence,
        }
```

## 你的 8 卡 H20 环境实践路径

```
阶段一：数据质量基础（1-2 天）
  1. 用 MinHash 对训练数据做一次全量去重
  2. 用启发式规则 + 质量打分过滤低质量数据
  3. 统计数据分布，了解数据的组成

阶段二：合成数据扩充（2-3 天）
  1. 用你的 8 卡 H20 跑一个 7B 模型做推理
  2. 用 Self-Instruct 从种子集生成 5K-10K 条指令数据
  3. 用 Evol-Instruct 对已有指令做 2-3 轮进化
  4. 质量过滤后加入训练集

阶段三：飞轮闭环（持续）
  1. 模型上线后收集用户反馈
  2. 每周从反馈中筛选高质量样本
  3. 每月用新数据迭代微调
  4. 建立数据版本管理和质量追踪

工具选择：
  - 去重：datasketch（Python MinHash 库）/ 自研
  - 质量打分：fasttext 分类器 + 启发式规则
  - 合成数据：vLLM（在你的 8 卡上跑推理）
  - 标注管理：Label Studio（开源标注平台）
  - 数据版本：DVC（Data Version Control）
```

## 本章小结

- 数据质量是模型质量的天花板：好数据 + 小模型 > 差数据 + 大模型
- MinHash + LSH 是大规模去重的标准方案，能处理 TB 级数据
- 质量打分应多维度评估：语言质量、信息密度、教育价值、安全性
- Self-Instruct 和 Evol-Instruct 是低成本扩充指令数据的有效方法
- 数据飞轮是 AI 公司的核心竞争壁垒：部署→反馈→标注→训练→更好的部署
- 分层标注（自动→众包→专家）在质量和成本间找到平衡

## 延伸阅读

- [Self-Instruct 论文](https://arxiv.org/abs/2212.10560)
- [WizardLM / Evol-Instruct 论文](https://arxiv.org/abs/2304.12244)
- [The Pile 数据集构建](https://arxiv.org/abs/2101.00027)
- [Deduplicating Training Data 论文](https://arxiv.org/abs/2107.06499)
- [datasketch 库文档](https://ekzhu.com/datasketch/)
- [Data-centric AI 综述](https://arxiv.org/abs/2303.10158)
