# 数据隐私

## 1. AI 系统中的隐私挑战

### 1.1 隐私风险全景

```
┌───────────────────────────────────────────────┐
│              AI 系统隐私风险                    │
├──────────────┬────────────────┬───────────────┤
│   训练阶段    │    推理阶段    │   存储阶段    │
├──────────────┼────────────────┼───────────────┤
│ 训练数据含 PII│ 输入含敏感信息 │ 日志中的 PII  │
│ 模型记忆攻击  │ 输出泄露隐私   │ 缓存中的对话  │
│ 梯度泄露     │ 上下文窗口残留 │ 嵌入向量反推  │
│ 成员推断攻击  │ 侧信道泄露    │ 备份中的数据  │
└──────────────┴────────────────┴───────────────┘
```

### 1.2 PII（个人可识别信息）分类

| 类别 | 示例 | 风险级别 |
|------|------|----------|
| 直接标识符 | 姓名、身份证号、手机号 | 高 |
| 准标识符 | 年龄、性别、职业、邮编 | 中 |
| 敏感属性 | 健康状况、政治倾向、性取向 | 极高 |
| 关联信息 | 浏览历史、位置数据、消费记录 | 中-高 |

## 2. PII 检测

### 2.1 基于 NER 的检测

```python
"""
Named Entity Recognition (NER) 方法：
使用训练好的 NER 模型识别文本中的实体

优势：能识别语义上的 PII（如 "张三在北京工作"）
劣势：需要针对不同语言/领域训练
"""

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider

class PIIDetectorNER:
    """基于 NER 的 PII 检测器"""
    
    def __init__(self, languages=None):
        if languages is None:
            languages = ["zh", "en"]
        
        # 配置 NLP 引擎
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "zh", "model_name": "zh_core_web_trf"},
                {"lang_code": "en", "model_name": "en_core_web_trf"},
            ]
        })
        
        self.analyzer = AnalyzerEngine(
            nlp_engine=provider.create_engine(),
            supported_languages=languages
        )
    
    def detect(self, text: str, language: str = "zh") -> list:
        """
        检测文本中的 PII
        返回: [{"type": "PHONE_NUMBER", "start": 10, "end": 21, "score": 0.95}]
        """
        results = self.analyzer.analyze(
            text=text,
            language=language,
            entities=[
                "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS",
                "CREDIT_CARD", "IBAN_CODE", "ID_NUMBER",
                "LOCATION", "DATE_TIME", "IP_ADDRESS",
                "MEDICAL_LICENSE", "URL"
            ]
        )
        
        return [
            {
                "type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "score": r.score,
                "text": text[r.start:r.end]
            }
            for r in results
        ]
```

### 2.2 基于正则的检测

```python
import re
from dataclasses import dataclass
from typing import List

@dataclass
class PIIMatch:
    type: str
    value: str
    start: int
    end: int
    confidence: float

class PIIDetectorRegex:
    """基于正则表达式的 PII 检测器"""
    
    PATTERNS = {
        "CHINA_ID": {
            "pattern": r"\b[1-9]\d{5}(18|19|20)\d{2}(0[1-9]|1[0-2])"
                       r"(0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b",
            "description": "中国身份证号",
            "confidence": 0.95
        },
        "CHINA_PHONE": {
            "pattern": r"\b1[3-9]\d{9}\b",
            "description": "中国手机号",
            "confidence": 0.90
        },
        "EMAIL": {
            "pattern": r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
            "description": "邮箱地址",
            "confidence": 0.95
        },
        "CREDIT_CARD": {
            "pattern": r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|"
                       r"3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
            "description": "信用卡号",
            "confidence": 0.90
        },
        "BANK_CARD": {
            "pattern": r"\b[1-9]\d{15,18}\b",
            "description": "银行卡号（需要结合上下文验证）",
            "confidence": 0.60
        },
        "IP_ADDRESS": {
            "pattern": r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
                       r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
            "description": "IP 地址",
            "confidence": 0.85
        },
        "CHINA_PASSPORT": {
            "pattern": r"\b[EeGg]\d{8}\b",
            "description": "中国护照号",
            "confidence": 0.80
        },
        "LICENSE_PLATE": {
            "pattern": r"[一-龥][A-Z][A-Z0-9]{5}",
            "description": "中国车牌号",
            "confidence": 0.90
        },
    }
    
    def __init__(self):
        self.compiled = {
            name: re.compile(info["pattern"])
            for name, info in self.PATTERNS.items()
        }
    
    def detect(self, text: str) -> List[PIIMatch]:
        """检测文本中的 PII"""
        matches = []
        for name, pattern in self.compiled.items():
            for match in pattern.finditer(text):
                matches.append(PIIMatch(
                    type=name,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=self.PATTERNS[name]["confidence"]
                ))
        return sorted(matches, key=lambda m: m.start)
```

### 2.3 组合检测

```python
class PIIDetectorCombined:
    """组合 NER + 正则的 PII 检测器"""
    
    def __init__(self):
        self.ner_detector = PIIDetectorNER()
        self.regex_detector = PIIDetectorRegex()
    
    def detect(self, text: str, language: str = "zh") -> list:
        """综合两种方法的检测结果"""
        ner_results = self.ner_detector.detect(text, language)
        regex_results = self.regex_detector.detect(text)
        
        # 合并去重
        all_results = []
        for r in ner_results:
            all_results.append({
                "source": "NER",
                **r
            })
        for r in regex_results:
            all_results.append({
                "source": "Regex",
                "type": r.type,
                "text": r.value,
                "start": r.start,
                "end": r.end,
                "score": r.confidence
            })
        
        return self._deduplicate(all_results)
    
    def _deduplicate(self, results: list) -> list:
        """对重叠的检测结果去重，保留置信度更高的"""
        if not results:
            return results
        results.sort(key=lambda x: (x["start"], -x["score"]))
        deduped = [results[0]]
        for r in results[1:]:
            if r["start"] >= deduped[-1]["end"]:
                deduped.append(r)
            elif r["score"] > deduped[-1]["score"]:
                deduped[-1] = r
        return deduped
```

## 3. 数据脱敏

### 3.1 脱敏策略

```python
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

class DataAnonymizer:
    """数据脱敏器"""
    
    STRATEGIES = {
        "MASK": "用固定字符替换",      # 张三 → **
        "HASH": "用哈希值替换",        # 张三 → a1b2c3
        "REPLACE": "用占位符替换",     # 张三 → <PERSON>
        "ENCRYPT": "加密后存储",       # 张三 → enc(xxx)
        "GENERALIZE": "泛化处理",      # 25岁 → 20-30岁
        "REDACT": "完全删除",          # 张三 → [已删除]
    }
    
    def __init__(self):
        self.anonymizer = AnonymizerEngine()
        self.default_operators = {
            "PERSON": OperatorConfig("replace", {"new_value": "<姓名>"}),
            "PHONE_NUMBER": OperatorConfig("mask", {
                "type": "mask", "masking_char": "*",
                "chars_to_mask": 7, "from_end": True
            }),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<邮箱>"}),
            "CREDIT_CARD": OperatorConfig("mask", {
                "type": "mask", "masking_char": "*",
                "chars_to_mask": 12, "from_end": False
            }),
            "ID_NUMBER": OperatorConfig("mask", {
                "type": "mask", "masking_char": "*",
                "chars_to_mask": 14, "from_end": True
            }),
        }
    
    def anonymize(self, text: str, pii_results: list) -> str:
        """对检测到的 PII 进行脱敏"""
        result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=pii_results,
            operators=self.default_operators
        )
        return result.text
```

### 3.2 可逆脱敏（Tokenization）

```python
import hashlib
import json
from cryptography.fernet import Fernet

class ReversibleAnonymizer:
    """可逆脱敏：脱敏后可以还原原始数据"""
    
    def __init__(self, encryption_key: bytes = None):
        if encryption_key is None:
            encryption_key = Fernet.generate_key()
        self.fernet = Fernet(encryption_key)
        self.token_map = {}  # token → 原始值
    
    def tokenize(self, text: str, pii_matches: list) -> str:
        """将 PII 替换为加密 token"""
        result = text
        offset = 0
        
        for match in sorted(pii_matches, key=lambda m: m["start"]):
            original = match["text"]
            token = self._generate_token(original, match["type"])
            
            start = match["start"] + offset
            end = match["end"] + offset
            result = result[:start] + token + result[end:]
            offset += len(token) - len(original)
        
        return result
    
    def detokenize(self, tokenized_text: str) -> str:
        """还原脱敏后的文本"""
        result = tokenized_text
        for token, original in self.token_map.items():
            result = result.replace(token, original)
        return result
    
    def _generate_token(self, value: str, pii_type: str) -> str:
        """生成加密 token"""
        encrypted = self.fernet.encrypt(value.encode()).decode()
        token = f"<{pii_type}:{encrypted[:16]}>"
        self.token_map[token] = value
        return token
```

## 4. 差分隐私（Differential Privacy）

### 4.1 核心概念

```
差分隐私的数学定义：

对于随机化算法 M，如果对于任意两个相邻数据集 D1 和 D2
（仅在一条记录上有差异），以及任意输出集合 S：

    Pr[M(D1) ∈ S] ≤ e^ε × Pr[M(D2) ∈ S]

其中 ε 是隐私预算（privacy budget）：
- ε 越小 → 隐私保护越强 → 数据可用性越低
- ε 越大 → 隐私保护越弱 → 数据可用性越高

直觉理解：
- 添加或删除任何一个人的数据，不会显著改变算法的输出
- 攻击者无法从输出推断任何个体是否在数据集中
```

### 4.2 DP-SGD（差分隐私随机梯度下降）

```python
"""
DP-SGD 的核心步骤：

1. 正常计算每个样本的梯度
2. 裁剪梯度（Gradient Clipping）—— 限制每个样本的影响
3. 添加高斯噪声 —— 模糊化梯度
4. 用加噪后的梯度更新模型参数

公式：
  g_clipped = g / max(1, ||g|| / C)    # C 是裁剪阈值
  g_noisy = g_clipped + N(0, σ²C²I)    # σ 是噪声乘数
"""

# 使用 Opacus（PyTorch 的 DP 库）
from opacus import PrivacyEngine

def train_with_dp(model, train_loader, epochs=10, 
                  target_epsilon=1.0, target_delta=1e-5):
    """使用差分隐私训练模型"""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # 包装为 DP 训练
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=1.0,  # 梯度裁剪阈值
    )
    
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            loss = compute_loss(model, batch)
            loss.backward()
            optimizer.step()
        
        # 查看当前的隐私开销
        epsilon = privacy_engine.get_epsilon(delta=target_delta)
        print(f"Epoch {epoch}: ε = {epsilon:.2f}")
    
    return model
```

### 4.3 差分隐私对模型质量的影响

```
ε（隐私预算）与模型质量的权衡：

ε = 0.1  → 极强隐私，模型准确率可能下降 15-30%
ε = 1.0  → 强隐私，模型准确率可能下降 5-15%
ε = 10   → 弱隐私，模型准确率接近无 DP 训练
ε = ∞    → 无隐私保护（等同于普通训练）

实践建议：
- 通常 ε = 1-10 是实用范围
- 大模型对 DP 噪声的鲁棒性更好
- 可以通过更多数据/更大 batch size 缓解精度损失
- 预训练阶段不加 DP，微调阶段加 DP 是常见策略
```

## 5. 联邦学习基础

### 5.1 概念

```
联邦学习核心思想：数据不动，模型动

传统训练：  数据 → 集中到中心服务器 → 训练模型
联邦学习：  模型 → 分发到各数据源 → 本地训练 → 聚合梯度

┌─────────┐     ┌─────────┐     ┌─────────┐
│ 客户端 A │     │ 客户端 B │     │ 客户端 C │
│ (医院A)  │     │ (医院B)  │     │ (医院C)  │
│ 本地数据  │     │ 本地数据  │     │ 本地数据  │
│ 本地训练  │     │ 本地训练  │     │ 本地训练  │
└────┬─────┘     └────┬─────┘     └────┬─────┘
     │ 梯度/参数       │ 梯度/参数       │ 梯度/参数
     └────────────┬───┴────────────────┘
              ┌───▼───┐
              │ 聚合器 │  ← FedAvg / FedProx
              │(中心)  │
              └───┬───┘
                  │ 更新后的全局模型
     ┌────────────┼───────────────────┐
     ▼            ▼                   ▼
  客户端 A     客户端 B           客户端 C
```

### 5.2 简化实现

```python
import torch
import copy

class FederatedLearning:
    """联邦学习简化实现（FedAvg）"""
    
    def __init__(self, global_model, num_clients: int):
        self.global_model = global_model
        self.num_clients = num_clients
    
    def train_round(self, client_data_loaders: list, local_epochs: int = 5):
        """一轮联邦训练"""
        client_models = []
        
        for i, data_loader in enumerate(client_data_loaders):
            # 每个客户端从全局模型开始本地训练
            local_model = copy.deepcopy(self.global_model)
            local_model = self._local_train(local_model, data_loader, local_epochs)
            client_models.append(local_model)
        
        # 聚合所有客户端的模型（FedAvg）
        self._federated_average(client_models)
    
    def _local_train(self, model, data_loader, epochs):
        """本地训练"""
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        model.train()
        for epoch in range(epochs):
            for batch in data_loader:
                optimizer.zero_grad()
                loss = compute_loss(model, batch)
                loss.backward()
                optimizer.step()
        return model
    
    def _federated_average(self, client_models: list):
        """FedAvg: 对所有客户端模型参数取平均"""
        global_dict = self.global_model.state_dict()
        
        for key in global_dict.keys():
            global_dict[key] = torch.stack([
                client.state_dict()[key].float()
                for client in client_models
            ], 0).mean(0)
        
        self.global_model.load_state_dict(global_dict)
```

## 6. 合规要求

### 6.1 GDPR 关键要求

| 原则 | 要求 | AI 系统影响 |
|------|------|-------------|
| 数据最小化 | 只收集必要数据 | 训练数据需要审计 |
| 目的限制 | 数据只能用于声明的目的 | 不能将用户对话用于训练 |
| 存储限制 | 数据保留期限 | 对话日志定期清理 |
| 被遗忘权 | 用户可要求删除数据 | 需要从模型中"遗忘"数据 |
| 可解释性 | 自动化决策需要解释 | AI 决策需要可解释 |
| 数据可移植性 | 用户可导出数据 | 需要提供数据导出接口 |

### 6.2 中国《个人信息保护法》关键要求

```
核心原则：
1. 合法、正当、必要和诚信
2. 明确、合理的处理目的
3. 最小必要原则
4. 公开透明
5. 保证信息质量
6. 安全保障

对 AI 系统的特殊要求：
- 自动化决策需要保证透明度和公平性
- 不得通过自动化决策方式在交易条件上实行不合理差别待遇
- 个人有权拒绝仅通过自动化决策方式作出的决定
- 敏感个人信息处理需要单独同意
- 跨境传输需要安全评估
```

### 6.3 实施检查清单

```
数据隐私实施检查清单：

□ 数据收集
  □ 明确告知用户数据收集范围和用途
  □ 获取用户明确同意
  □ 实施数据最小化（只收集必要数据）

□ 数据处理
  □ PII 检测和脱敏机制已部署
  □ 训练数据已去除/脱敏 PII
  □ 差分隐私或联邦学习方案已评估

□ 数据存储
  □ 数据加密存储
  □ 访问控制已实施
  □ 数据保留策略已定义
  □ 定期清理过期数据

□ 用户权利
  □ 支持数据查看/导出
  □ 支持数据删除请求
  □ 支持退出自动化决策
  □ 投诉处理机制

□ 审计
  □ 数据处理日志
  □ 定期隐私影响评估
  □ 第三方审计机制
```

## 7. 小结

数据隐私是 AI 安全的基石。核心要点：
1. PII 检测需要 NER + 正则组合使用
2. 脱敏策略根据场景选择（不可逆/可逆）
3. 差分隐私通过数学保证保护隐私，但会影响模型质量
4. 联邦学习适用于数据不能出域的场景
5. 必须满足 GDPR/个人信息保护法等合规要求
