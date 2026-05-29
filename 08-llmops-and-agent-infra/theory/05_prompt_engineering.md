# Prompt 工程化

## 1. 从 Prompt 到 Prompt Engineering System

### 1.1 Prompt 的演进阶段

```
阶段 1：手工 Prompt
  "帮我总结这篇文章"
  → 随意编写，效果不稳定

阶段 2：模板化 Prompt
  "请以{format}格式总结以下文章，重点关注{focus}..."
  → 参数化，可复用

阶段 3：工程化 Prompt
  版本控制 + A/B 测试 + 自动优化 + 监控
  → 完整的工程体系

阶段 4：自动化 Prompt (DSPy)
  定义目标 → 自动搜索最优 Prompt
  → 由算法替代手工调优
```

### 1.2 为什么需要 Prompt 工程化

对后端工程师的类比：

```
API 代码需要：                Prompt 同样需要：
├── Git 版本控制       →     ├── Prompt 版本管理
├── 单元测试          →     ├── Prompt 评估用例
├── CI/CD             →     ├── Prompt 自动化测试
├── A/B 测试          →     ├── Prompt A/B 测试
├── 配置中心          →     ├── Prompt 注册中心
├── 监控告警          →     ├── Prompt 质量监控
└── 代码审查          →     └── Prompt Review
```

## 2. Prompt 版本控制

### 2.1 版本管理策略

```
方案 1：Git 管理（推荐起步方案）

prompts/
├── summarization/
│   ├── v1.0.0.yaml
│   ├── v1.1.0.yaml
│   └── v2.0.0.yaml
├── qa/
│   ├── v1.0.0.yaml
│   └── v1.1.0.yaml
└── prompt_registry.yaml    # 注册表：哪个版本在生产使用

语义版本号：
  major.minor.patch
  major: Prompt 结构重大变更
  minor: 添加/修改指令
  patch: 修复拼写/格式
```

### 2.2 Prompt 模板格式

```yaml
# prompts/summarization/v2.0.0.yaml
name: summarization
version: "2.0.0"
description: "文档摘要 Prompt - 使用 CoT 策略"
author: "team-ml"
created_at: "2024-11-20"

model_requirements:
  min_capability: "gpt-4-level"  # 需要的最低模型能力
  max_tokens: 2000
  temperature: 0.3

template: |
  你是一位专业的文档分析师。

  ## 任务
  请对以下文档进行摘要，遵循以下步骤：

  1. 首先识别文档的核心主题
  2. 提取 3-5 个关键要点
  3. 生成结构化摘要

  ## 输出格式
  - 核心主题：（一句话）
  - 关键要点：（列表）
  - 摘要：（{max_words}字以内）

  ## 文档内容
  {document}

variables:
  - name: document
    type: string
    required: true
  - name: max_words
    type: integer
    default: 200

evaluation:
  dataset: "summarization_eval_v2"
  metrics:
    - name: relevancy
      threshold: 0.85
    - name: coherence
      threshold: 0.80
  last_eval_score: 0.89
  last_eval_date: "2024-11-19"
```

### 2.3 Prompt 注册中心

```python
class PromptRegistry:
    """Prompt 注册中心 - 管理所有 Prompt 的版本和路由"""
    
    def __init__(self, config_path: str = "prompt_registry.yaml"):
        self.config = self._load_config(config_path)
        self.cache = {}
        self.metrics = PromptMetrics()
    
    def get_prompt(
        self,
        name: str,
        version: str = None,
        ab_test_group: str = None,
    ) -> PromptTemplate:
        """
        获取 Prompt 模板
        - version=None: 使用生产版本
        - ab_test_group: 参与 A/B 测试时使用对应版本
        """
        if ab_test_group:
            version = self._get_ab_version(name, ab_test_group)
        elif version is None:
            version = self.config["production"][name]
        
        return self._load_prompt(name, version)
    
    def promote(self, name: str, version: str):
        """将指定版本提升为生产版本（需通过评估门禁）"""
        eval_result = self._run_evaluation(name, version)
        if eval_result.passes_threshold():
            self.config["production"][name] = version
            self._save_config()
            self.metrics.record_promotion(name, version)
        else:
            raise ValueError(f"评估未通过: {eval_result}")
```

## 3. A/B 测试

### 3.1 Prompt A/B 测试架构

```
                    ┌──────────────┐
                    │   用户请求    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Traffic     │
                    │  Splitter    │
                    │  (流量分配)  │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │ 70%        │ 20%        │ 10%
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Prompt   │ │ Prompt   │ │ Prompt   │
        │ v1.0     │ │ v2.0     │ │ v2.1     │
        │ (控制组) │ │ (实验A)  │ │ (实验B)  │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             └─────────────┼─────────────┘
                           ▼
                    ┌──────────────┐
                    │  结果收集    │
                    │  质量评估    │
                    │  统计显著性  │
                    └──────────────┘
```

### 3.2 A/B 测试指标

```python
class ABTestMetrics:
    """Prompt A/B 测试指标体系"""
    
    quality_metrics = {
        "relevancy": "回答与问题的相关性",
        "faithfulness": "回答与上下文的忠实度",
        "correctness": "答案正确性",
        "helpfulness": "对用户的帮助程度",
    }
    
    efficiency_metrics = {
        "latency_p50": "P50 延迟",
        "latency_p95": "P95 延迟",
        "token_usage": "平均 Token 消耗",
        "cost_per_query": "每次查询成本",
    }
    
    business_metrics = {
        "user_satisfaction": "用户满意度评分",
        "task_completion": "任务完成率",
        "follow_up_rate": "追问率（越低越好）",
    }
```

### 3.3 统计显著性判断

```python
from scipy import stats

def is_significant(control_scores, treatment_scores, alpha=0.05):
    """判断 A/B 测试结果是否显著"""
    t_stat, p_value = stats.ttest_ind(control_scores, treatment_scores)
    
    effect_size = (
        (np.mean(treatment_scores) - np.mean(control_scores))
        / np.std(control_scores)
    )
    
    return {
        "significant": p_value < alpha,
        "p_value": p_value,
        "effect_size": effect_size,
        "recommendation": (
            "采用新版本" if p_value < alpha and effect_size > 0.1
            else "维持现状" if p_value >= alpha
            else "新版本更差，拒绝"
        )
    }
```

## 4. DSPy 自动优化

### 4.1 DSPy 核心理念

```
传统 Prompt 工程：
  人工编写 Prompt → 测试 → 手动调整 → 测试 → ... (费时费力)

DSPy 方式：
  定义任务签名 → 提供训练数据 → 自动搜索最优 Prompt → 编译为优化后的程序

类比：
  传统 Prompt = 手写汇编
  DSPy = 高级语言 + 编译器优化
```

### 4.2 DSPy 基本概念

```python
import dspy

# 1. 定义 Signature（任务签名）
class SummarizeDoc(dspy.Signature):
    """将技术文档总结为简洁的摘要"""
    document: str = dspy.InputField(desc="需要总结的技术文档")
    summary: str = dspy.OutputField(desc="200字以内的结构化摘要")

# 2. 构建 Module（模块）
class DocSummarizer(dspy.Module):
    def __init__(self):
        self.summarize = dspy.ChainOfThought(SummarizeDoc)
    
    def forward(self, document):
        return self.summarize(document=document)

# 3. 定义评估指标
def quality_metric(example, prediction, trace=None):
    relevancy = assess_relevancy(example.document, prediction.summary)
    conciseness = len(prediction.summary) < 200
    return relevancy > 0.8 and conciseness

# 4. 编译优化
optimizer = dspy.MIPROv2(metric=quality_metric, num_threads=4)
optimized_summarizer = optimizer.compile(
    DocSummarizer(),
    trainset=train_examples,
)
```

### 4.3 DSPy Optimizer 类型

```
Optimizer          策略                   适用场景
├── BootstrapFewShot    自动选择 Few-shot 示例    快速优化
├── MIPROv2             Prompt 指令 + 示例联合优化  最佳效果
├── BootstrapFinetune   生成微调数据              有微调需求
└── KNNFewShot          基于相似度选示例           大量示例可用
```

### 4.4 DSPy 生产集成

```python
# 优化后的程序可以导出为标准 Prompt
optimized_program = optimizer.compile(...)

# 方式 1：直接使用优化后的程序
result = optimized_program(document="...")

# 方式 2：导出为 Prompt 模板
# 将优化后的 few-shot 示例和指令提取出来
# 集成到现有的 Prompt Registry
```

## 5. Few-shot 管理

### 5.1 Few-shot 示例池

```python
class FewShotManager:
    """Few-shot 示例管理器"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store  # 存储示例的向量库
    
    def add_example(self, input_text: str, output_text: str,
                    metadata: dict = None):
        """添加示例到池中"""
        self.vector_store.add(
            text=f"Input: {input_text}\nOutput: {output_text}",
            metadata={"input": input_text, "output": output_text,
                      **(metadata or {})}
        )
    
    def get_relevant_examples(self, query: str, k: int = 3) -> list:
        """根据查询检索最相关的 Few-shot 示例"""
        results = self.vector_store.search(query, top_k=k)
        return [
            {"input": r.metadata["input"], "output": r.metadata["output"]}
            for r in results
        ]
    
    def get_diverse_examples(self, query: str, k: int = 3) -> list:
        """检索相关但多样化的示例（MMR 策略）"""
        return self.vector_store.max_marginal_relevance_search(
            query, k=k, lambda_mult=0.5
        )
```

### 5.2 动态 Few-shot 策略

```
静态 Few-shot：
  所有查询使用固定示例
  → 简单但不够精准

动态 Few-shot：
  根据用户查询检索最相关的示例
  → 效果好但增加延迟

自适应 Few-shot：
  根据任务难度动态调整示例数量
  简单任务 → 0-1 个示例
  中等任务 → 2-3 个示例
  复杂任务 → 4-5 个示例
```

## 6. Prompt 模板系统设计

### 6.1 模板引擎

```python
from jinja2 import Template
from typing import Any

class PromptTemplate:
    """生产级 Prompt 模板"""
    
    def __init__(self, template_str: str, metadata: dict = None):
        self.template = Template(template_str)
        self.metadata = metadata or {}
        self._validate_template()
    
    def render(self, **kwargs) -> str:
        """渲染模板，自动注入 few-shot 示例"""
        # 注入动态 few-shot
        if "few_shot_examples" not in kwargs and self.metadata.get("few_shot"):
            kwargs["few_shot_examples"] = self._get_few_shot(
                kwargs.get("query", "")
            )
        
        rendered = self.template.render(**kwargs)
        
        # 记录渲染结果的 token 数（用于成本估算）
        self._record_token_count(rendered)
        
        return rendered
    
    def _validate_template(self):
        """验证模板变量完整性"""
        required_vars = self.metadata.get("required_variables", [])
        template_vars = self.template.environment.parse(
            self.template.source
        ).find_all("Name")
        # 确保所有必需变量在模板中存在
```

### 6.2 Prompt 组合模式

```python
class PromptComposer:
    """Prompt 组合器 - 将多个 Prompt 片段组合"""
    
    def __init__(self):
        self.sections = {}
    
    def add_system_prompt(self, content: str):
        self.sections["system"] = content
    
    def add_persona(self, role: str, expertise: str):
        self.sections["persona"] = f"你是一位{role}，擅长{expertise}。"
    
    def add_context(self, documents: list[str]):
        context = "\n---\n".join(documents)
        self.sections["context"] = f"## 参考资料\n{context}"
    
    def add_few_shot(self, examples: list[dict]):
        formatted = "\n".join(
            f"问：{e['input']}\n答：{e['output']}" for e in examples
        )
        self.sections["few_shot"] = f"## 示例\n{formatted}"
    
    def add_output_format(self, format_spec: str):
        self.sections["format"] = f"## 输出格式\n{format_spec}"
    
    def add_guardrails(self, rules: list[str]):
        rules_text = "\n".join(f"- {r}" for r in rules)
        self.sections["guardrails"] = f"## 注意事项\n{rules_text}"
    
    def compose(self) -> str:
        order = ["persona", "system", "context", "few_shot",
                 "format", "guardrails"]
        parts = [self.sections[k] for k in order if k in self.sections]
        return "\n\n".join(parts)
```

## 7. Prompt 质量监控

### 7.1 监控指标

```
线上监控：
├── 输出质量指标
│   • LLM-as-Judge 评分（定期采样评估）
│   • 用户反馈（点赞/踩）
│   • 任务完成率
│
├── 效率指标
│   • Token 消耗趋势
│   • 延迟分布
│   • 缓存命中率
│
├── 安全指标
│   • Prompt 注入检测率
│   • 有害内容产生率
│   • 敏感信息泄露检测
│
└── 漂移检测
    • 输出长度分布变化
    • 主题分布变化
    • 拒绝率变化
```

### 7.2 告警规则

```python
alert_rules = {
    "quality_drop": {
        "condition": "avg_quality_score < 0.8 for 1h",
        "severity": "critical",
        "action": "auto_rollback_prompt_version"
    },
    "token_spike": {
        "condition": "avg_tokens_per_request > baseline * 1.5",
        "severity": "warning",
        "action": "notify_team"
    },
    "injection_detected": {
        "condition": "injection_count > 10 in 5m",
        "severity": "critical",
        "action": "enable_strict_mode + notify_security"
    }
}
```

## 8. 小结

Prompt 工程化的核心实践：

1. **版本控制**：每个 Prompt 变更都可追溯
2. **评估驱动**：变更前必须通过评估门禁
3. **A/B 测试**：灰度验证新 Prompt 的效果
4. **自动优化**：使用 DSPy 等工具自动搜索最优 Prompt
5. **动态 Few-shot**：根据查询动态选择最佳示例
6. **持续监控**：线上质量、成本、安全三维监控

从手工调 Prompt 到 Prompt 工程化，核心思维转变是：**把 Prompt 当代码来管理**。
