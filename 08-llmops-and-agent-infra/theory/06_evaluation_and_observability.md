# LLM 评估体系与可观测性

## 1. LLM 评估的独特挑战

### 1.1 为什么传统评估方法不够用

```
传统 ML 评估：
  • 有明确的标准答案
  • 指标计算是确定性的
  • 评估一次性完成

LLM 评估面临的挑战：
  • 开放式输出，无唯一正确答案
  • "好"与"不好"的界限模糊
  • 需要评估多个维度（正确性/相关性/忠实度/有害性/流畅度）
  • 评估本身可能需要 LLM（LLM-as-Judge）
  • 同一输入多次输出可能不同
```

### 1.2 评估维度框架

```
┌─────────────────────────────────────────────────────┐
│                LLM 评估维度                           │
│                                                       │
│  ┌─────────────────────────────────────────────┐     │
│  │  Faithfulness（忠实度）                       │     │
│  │  回答是否基于提供的上下文？是否有幻觉？       │     │
│  └─────────────────────────────────────────────┘     │
│                                                       │
│  ┌─────────────────────────────────────────────┐     │
│  │  Relevancy（相关性）                          │     │
│  │  回答是否与用户问题相关？是否答非所问？       │     │
│  └─────────────────────────────────────────────┘     │
│                                                       │
│  ┌─────────────────────────────────────────────┐     │
│  │  Correctness（正确性）                        │     │
│  │  回答的内容是否事实准确？                      │     │
│  └─────────────────────────────────────────────┘     │
│                                                       │
│  ┌─────────────────────────────────────────────┐     │
│  │  Harmlessness（无害性）                       │     │
│  │  回答是否包含有害/偏见/不当内容？             │     │
│  └─────────────────────────────────────────────┘     │
│                                                       │
│  ┌─────────────────────────────────────────────┐     │
│  │  Helpfulness（有用性）                        │     │
│  │  回答对用户是否真正有帮助？                    │     │
│  └─────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────┘
```

## 2. 评估框架

### 2.1 Ragas

```
定位：RAG 系统专用评估框架
核心指标：

┌────────────────────────────────────────────────────┐
│                  Ragas 指标体系                      │
│                                                      │
│  Generation 指标：                                    │
│  ├── Faithfulness: 回答中的陈述是否能从上下文推导     │
│  ├── Answer Relevancy: 回答与问题的相关程度           │
│  └── Answer Correctness: 与标准答案的一致性           │
│                                                      │
│  Retrieval 指标：                                     │
│  ├── Context Precision: 检索结果中相关文档的排序质量  │
│  ├── Context Recall: 标准答案所需信息的覆盖率         │
│  └── Context Relevancy: 检索结果与问题的相关性        │
│                                                      │
│  End-to-End 指标：                                    │
│  └── Answer Similarity: 语义相似度评分               │
└────────────────────────────────────────────────────┘
```

**Ragas 评估流程**：
```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# 准备评估数据
eval_dataset = {
    "question": ["公司的年假政策是什么？", ...],
    "answer": ["员工入职满一年享有5天年假...", ...],
    "contexts": [["HR政策文档第3.2节...", ...], ...],
    "ground_truth": ["入职满1年5天，满5年10天...", ...],
}

# 执行评估
results = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy,
             context_precision, context_recall],
)
# 输出：每个指标的分数（0-1）
```

### 2.2 DeepEval

```
定位：全面的 LLM 评估框架（类似 pytest 的体验）
特点：
  • 集成测试风格的评估
  • 丰富的内置指标
  • 支持自定义指标
  • CI/CD 友好
```

```python
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    ToxicityMetric,
)

# 定义测试用例（类似单元测试）
def test_rag_response():
    test_case = LLMTestCase(
        input="公司的年假政策是什么？",
        actual_output="员工入职满一年享有5天年假...",
        retrieval_context=["HR政策文档..."],
        expected_output="入职满1年5天年假..."
    )
    
    # 多维度评估
    metrics = [
        AnswerRelevancyMetric(threshold=0.8),
        FaithfulnessMetric(threshold=0.9),
        HallucinationMetric(threshold=0.1),  # 越低越好
    ]
    
    assert_test(test_case, metrics)

# 可以用 pytest 运行：pytest test_rag.py
```

### 2.3 TruLens

```
定位：LLM 应用的全链路评估与追踪
特点：
  • 与 LangChain/LlamaIndex 深度集成
  • 反馈函数（Feedback Functions）机制
  • 仪表板可视化
  • 支持线上持续评估
```

```python
from trulens_eval import TruChain, Feedback
from trulens_eval.feedback import Groundedness, GroundTruthAgreement

# 定义反馈函数
groundedness = Groundedness()
f_groundedness = Feedback(groundedness.groundedness_measure).on(
    "context"  # 评估回答对上下文的基础性
).on_output()

# 包装现有的 LangChain 链
tru_chain = TruChain(
    rag_chain,
    feedbacks=[f_groundedness, f_relevancy, f_harmfulness],
    app_id="rag_v2"
)

# 正常调用，自动记录评估
response = tru_chain("公司的年假政策是什么？")
```

### 2.4 框架对比

| 维度 | Ragas | DeepEval | TruLens |
|------|-------|----------|---------|
| 专注领域 | RAG | 通用 LLM | 全链路 |
| 使用方式 | 批量评估 | 测试框架 | 在线监控 |
| 学习曲线 | 低 | 低 | 中 |
| CI/CD | 需封装 | 原生 | 需封装 |
| 可视化 | 基础 | Dashboard | Dashboard |
| 自定义指标 | 支持 | 支持 | 支持 |
| 推荐场景 | RAG 开发 | 回归测试 | 生产监控 |

## 3. 幻觉检测

### 3.1 幻觉类型

```
幻觉类别：
├── 事实幻觉 (Factual Hallucination)
│   回答包含与事实不符的信息
│   例："Python 于 1995 年发布"（实际是 1991 年）
│
├── 忠实度幻觉 (Faithfulness Hallucination)
│   回答声称基于上下文，但实际编造了内容
│   例：上下文说"年假5天"，回答说"年假10天"
│
├── 不一致幻觉 (Inconsistency Hallucination)
│   回答内部自相矛盾
│   例：前文说"支持"，后文说"不支持"
│
└── 过度推断 (Over-inference)
    从有限证据中得出过于确定的结论
    例：上下文仅提到"可能相关"，回答说"已证实"
```

### 3.2 幻觉检测方法

```python
class HallucinationDetector:
    """多策略幻觉检测器"""
    
    def detect(self, question: str, answer: str,
               context: list[str]) -> dict:
        results = {}
        
        # 策略 1：NLI-based（自然语言推理）
        results["nli"] = self._nli_check(answer, context)
        
        # 策略 2：Claim Decomposition（声明分解）
        claims = self._extract_claims(answer)
        results["unsupported_claims"] = [
            claim for claim in claims
            if not self._is_supported(claim, context)
        ]
        
        # 策略 3：LLM-as-Judge
        results["llm_judge"] = self._llm_judge(
            question, answer, context
        )
        
        # 综合判断
        results["hallucination_score"] = self._aggregate(results)
        return results
    
    def _extract_claims(self, text: str) -> list[str]:
        """将回答分解为原子化的声明"""
        # 使用 LLM 将长文本拆分为独立的事实陈述
        prompt = f"请将以下文本分解为独立的事实陈述：\n{text}"
        return llm.generate(prompt).split("\n")
    
    def _is_supported(self, claim: str, context: list[str]) -> bool:
        """检查单个声明是否被上下文支持"""
        # NLI 模型判断 context -> claim 是否为蕴含关系
        for ctx in context:
            if self.nli_model.entails(premise=ctx, hypothesis=claim):
                return True
        return False
```

### 3.3 生产级幻觉防护

```
防护策略：
├── 预防（Generation 阶段）
│   • 低 Temperature 减少随机性
│   • 明确指令"仅基于提供的文档回答"
│   • "如果不确定请说不知道"
│
├── 检测（Post-processing 阶段）
│   • 自动幻觉检测
│   • 声明级别的可信度标注
│   • 引用追溯（标注每句话的来源）
│
└── 缓解（Response 阶段）
    • 高幻觉风险时拒绝回答
    • 标注置信度
    • 提供原始文档链接供验证
```

## 4. 全链路追踪

### 4.1 LangSmith

```
LangChain 官方的追踪与评估平台：
  • 自动记录 LangChain/LangGraph 调用链
  • 可视化每一步的输入输出
  • 延迟和 Token 消耗统计
  • 数据集管理和批量评估
  • 在线评估和标注

集成方式：
  export LANGCHAIN_TRACING_V2=true
  export LANGCHAIN_API_KEY=...
  # 自动追踪所有 LangChain 调用
```

### 4.2 Langfuse

```
开源的 LLM 可观测性平台：
  • 自托管或 Cloud
  • 支持多框架（LangChain/LlamaIndex/OpenAI）
  • Trace/Span/Generation 三层结构
  • 成本追踪
  • 评估 + 标注
  • Prompt 管理

优势：
  • 开源可自托管（数据不出域）
  • 轻量级 SDK，侵入性低
  • 支持自定义事件和指标
```

```python
from langfuse import Langfuse
from langfuse.decorators import observe

langfuse = Langfuse()

@observe()  # 自动追踪
def rag_pipeline(query: str) -> str:
    # 每个步骤自动记录
    docs = retrieve(query)      # Span: retrieve
    context = rerank(docs)      # Span: rerank
    answer = generate(context)  # Span: generate (含 LLM 调用详情)
    return answer

# Langfuse 自动记录：
# - 每步的输入输出
# - LLM 调用的模型/Token/延迟
# - 整体 Trace 耗时和成本
```

### 4.3 追踪数据模型

```
Trace（一次完整请求）
  ├── metadata: user_id, session_id, version
  ├── input: 用户原始输入
  ├── output: 最终输出
  ├── duration: 总耗时
  ├── cost: 总成本
  │
  ├── Span: query_processing (50ms)
  │     └── input/output
  │
  ├── Span: retrieval (200ms)
  │     ├── input: query + params
  │     └── output: 5 documents
  │
  ├── Span: reranking (150ms)
  │     ├── input: 5 documents
  │     └── output: 3 documents (reranked)
  │
  ├── Generation: llm_call (2000ms)
  │     ├── model: qwen2.5-72b
  │     ├── prompt_tokens: 1500
  │     ├── completion_tokens: 300
  │     ├── temperature: 0.3
  │     └── cost: $0.02
  │
  └── Span: post_processing (100ms)
        ├── hallucination_check: passed
        └── output: final answer

Total: 2500ms, $0.02, quality_score: 0.92
```

## 5. 评估流水线设计

### 5.1 离线评估流水线

```
┌────────────────────────────────────────────────────┐
│              离线评估流水线                           │
│                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │ 评估数据集│ →  │ 批量推理  │ →  │ 指标计算  │      │
│  │ (Golden) │    │ (并行)   │    │ (多维度) │      │
│  └──────────┘    └──────────┘    └──────────┘      │
│                                        │            │
│                                 ┌──────▼──────┐     │
│                                 │  门禁判断   │     │
│                                 │ Pass/Fail   │     │
│                                 └──────┬──────┘     │
│                          ┌─────────────┼──────────┐ │
│                          ▼             ▼          │ │
│                    ┌──────────┐  ┌──────────┐    │ │
│                    │ 允许发布  │  │ 阻止发布  │    │ │
│                    └──────────┘  └──────────┘    │ │
└────────────────────────────────────────────────────┘

触发时机：
  • Prompt 变更时
  • 模型版本切换时
  • 定期回归测试（每天/每周）
  • PR 合并前
```

### 5.2 在线评估

```python
class OnlineEvaluator:
    """在线评估器 - 采样评估生产流量"""
    
    def __init__(self, sample_rate: float = 0.1):
        self.sample_rate = sample_rate
        self.metrics_buffer = []
    
    async def evaluate_if_sampled(self, request, response):
        """按采样率评估"""
        if random.random() > self.sample_rate:
            return
        
        # 异步评估，不阻塞主流程
        metrics = await asyncio.gather(
            self._evaluate_relevancy(request, response),
            self._evaluate_faithfulness(request, response),
            self._detect_hallucination(request, response),
        )
        
        self.metrics_buffer.append({
            "timestamp": time.time(),
            "request_id": request.id,
            "metrics": metrics,
        })
        
        # 检查是否触发告警
        self._check_alerts(metrics)
```

### 5.3 人工评估

```
人工评估最佳实践：
├── 标注流程
│   • 双人标注 + 仲裁
│   • 标注指南文档化
│   • 定期校准会议
│
├── 标注工具
│   • Argilla（开源，推荐）
│   • Label Studio（通用）
│   • Langfuse Annotation（集成在追踪中）
│
├── 标注维度
│   • 5 分制评分：1(很差) - 5(很好)
│   • 每个维度独立评分
│   • 提供文字反馈
│
└── 与自动评估校准
    • 人工评分作为 Ground Truth
    • 验证自动评估与人工的相关性
    • 定期更新自动评估的校准
```

## 6. 自定义评估指标

### 6.1 设计原则

```python
class CustomMetric:
    """自定义指标的标准接口"""
    
    @property
    def name(self) -> str:
        """指标名称"""
        raise NotImplementedError
    
    @property
    def description(self) -> str:
        """指标说明"""
        raise NotImplementedError
    
    def score(self, question: str, answer: str,
              context: list[str] = None,
              ground_truth: str = None) -> float:
        """
        计算分数
        Returns: 0.0 - 1.0
        """
        raise NotImplementedError


class DomainAccuracyMetric(CustomMetric):
    """领域特定准确性指标"""
    
    name = "domain_accuracy"
    description = "检查回答是否符合特定领域的规则和术语"
    
    def __init__(self, domain_rules: list[str]):
        self.domain_rules = domain_rules
    
    def score(self, question, answer, **kwargs) -> float:
        violations = 0
        for rule in self.domain_rules:
            if self._violates_rule(answer, rule):
                violations += 1
        return 1.0 - (violations / len(self.domain_rules))
```

## 7. 评估驱动的开发流程

### 7.1 完整流程

```
┌─────────────────────────────────────────────────────┐
│              评估驱动开发                             │
│                                                       │
│  1. 定义评估数据集                                    │
│     ├── 收集真实用户问题                              │
│     ├── 标注期望答案                                  │
│     └── 定义通过阈值                                  │
│                                                       │
│  2. 建立基线                                          │
│     ├── 当前系统跑评估                                │
│     └── 记录基线分数                                  │
│                                                       │
│  3. 迭代优化                                          │
│     ├── 修改 Prompt / RAG 配置 / 模型                 │
│     ├── 跑评估                                        │
│     ├── 对比基线                                      │
│     └── 通过 → 合并，未通过 → 继续优化               │
│                                                       │
│  4. 部署与监控                                        │
│     ├── 灰度发布                                      │
│     ├── 在线评估                                      │
│     └── 效果不佳 → 回滚                              │
└─────────────────────────────────────────────────────┘
```

## 8. 小结

构建 LLM 评估体系的关键：

1. **多维度评估**：不存在单一指标能衡量 LLM 应用质量
2. **自动 + 人工**：自动评估提效，人工评估兜底
3. **离线 + 在线**：离线门禁阻止劣质变更，在线监控发现退化
4. **全链路追踪**：每个环节都可观测，问题可快速定位
5. **持续校准**：评估系统本身也需要持续迭代

对后端工程师的映射：
```
API 测试       →  LLM 评估数据集
集成测试       →  端到端评估
APM 追踪      →  LLM Trace (Langfuse)
SLA 监控      →  质量指标监控
错误率告警     →  幻觉率告警
```
