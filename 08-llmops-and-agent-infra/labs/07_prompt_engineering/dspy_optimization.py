"""
Lab 07: DSPy 自动 Prompt 优化
使用 DSPy 自动搜索最优 Prompt
"""
import os

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    print("dspy 未安装，请运行: pip install dspy-ai")


LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-72b-instruct")


# =============================================================================
# DSPy 基础用法
# =============================================================================

def demo_dspy_basics():
    """DSPy 基础用法演示"""
    if not DSPY_AVAILABLE:
        print("DSPy 不可用，展示概念")
        _show_concept()
        return

    print("=" * 60)
    print("DSPy 自动 Prompt 优化演示")
    print("=" * 60)

    # 1. 配置 LLM
    lm = dspy.LM(
        model=f"openai/{LLM_MODEL}",
        api_base=LLM_BASE_URL,
        api_key="not-needed",
    )
    dspy.configure(lm=lm)

    # 2. 定义 Signature（任务签名）
    class QASignature(dspy.Signature):
        """回答关于 LLM 和 AI 基础设施的技术问题"""
        context: str = dspy.InputField(desc="相关的技术文档")
        question: str = dspy.InputField(desc="技术问题")
        answer: str = dspy.OutputField(desc="准确、有据可查的回答")

    # 3. 构建 Module
    class SimpleQA(dspy.Module):
        def __init__(self):
            self.qa = dspy.ChainOfThought(QASignature)

        def forward(self, context, question):
            return self.qa(context=context, question=question)

    # 4. 测试
    qa = SimpleQA()
    result = qa(
        context="H20 GPU 有 96GB HBM3 显存，带宽 4TB/s。8 卡可部署 72B 模型（TP=4）。",
        question="8 张 H20 能同时部署几个 72B 模型？",
    )
    print(f"\n基础模式结果:")
    print(f"  问题: 8 张 H20 能同时部署几个 72B 模型？")
    print(f"  回答: {result.answer}")

    # 5. 准备训练数据
    trainset = [
        dspy.Example(
            context="Milvus 支持 HNSW、IVF_FLAT、IVF_PQ 索引。HNSW 精度最高但内存大。",
            question="小规模场景用什么索引？",
            answer="小规模场景推荐 HNSW 索引，精度高且构建快。",
        ).with_inputs("context", "question"),
        dspy.Example(
            context="RAG 的核心指标包括 Faithfulness（忠实度）和 Relevancy（相关性）。",
            question="如何评估 RAG 质量？",
            answer="使用 Faithfulness 评估回答是否基于上下文，Relevancy 评估是否回答了问题。",
        ).with_inputs("context", "question"),
        dspy.Example(
            context="vLLM 使用 PagedAttention 优化显存管理，支持 Tensor Parallel。",
            question="vLLM 的核心优化是什么？",
            answer="PagedAttention 减少显存浪费，Tensor Parallel 支持多卡部署大模型。",
        ).with_inputs("context", "question"),
    ]

    # 6. 定义评估指标
    def quality_metric(example, prediction, trace=None):
        # 简化的评估：检查答案中是否包含关键信息
        answer = prediction.answer.lower()
        expected = example.answer.lower()
        # 计算关键词重叠
        expected_words = set(expected.split())
        answer_words = set(answer.split())
        overlap = len(expected_words & answer_words)
        return overlap / len(expected_words) if expected_words else 0

    # 7. 优化
    print("\n--- 开始优化 ---")
    try:
        optimizer = dspy.BootstrapFewShot(
            metric=quality_metric,
            max_bootstrapped_demos=2,
            max_labeled_demos=2,
        )
        optimized_qa = optimizer.compile(SimpleQA(), trainset=trainset)

        # 测试优化后的效果
        result_optimized = optimized_qa(
            context="H20 GPU 有 96GB HBM3 显存，带宽 4TB/s。",
            question="H20 GPU 的核心优势是什么？",
        )
        print(f"\n优化后结果:")
        print(f"  回答: {result_optimized.answer}")
    except Exception as e:
        print(f"  优化过程出错: {e}")
        print("  这在本地模型上是正常的，DSPy 优化需要稳定的 LLM 输出")


def _show_concept():
    """展示 DSPy 概念"""
    print("=" * 60)
    print("DSPy 概念介绍")
    print("=" * 60)

    print("""
DSPy 核心理念：
  传统 Prompt Engineering → 手工编写和调优 Prompt
  DSPy → 定义任务签名，自动搜索最优 Prompt

关键概念：

1. Signature（任务签名）
   定义输入输出的语义描述
   ```python
   class QA(dspy.Signature):
       context: str = dspy.InputField(desc="参考文档")
       question: str = dspy.InputField(desc="用户问题")
       answer: str = dspy.OutputField(desc="准确回答")
   ```

2. Module（模块）
   可组合的处理单元
   - dspy.Predict: 基础预测
   - dspy.ChainOfThought: 思维链
   - dspy.ReAct: ReAct Agent

3. Optimizer（优化器）
   自动搜索最优 Prompt
   - BootstrapFewShot: 自动选择最佳 Few-shot 示例
   - MIPROv2: Prompt 指令 + 示例联合优化
   - BootstrapFinetune: 生成微调数据

4. Metric（评估指标）
   定义"好"的标准
   ```python
   def metric(example, prediction, trace=None):
       return score  # 0.0 - 1.0
   ```

工作流：
  定义 Signature → 构建 Module → 准备训练数据 → 定义 Metric → Compile 优化
""")


if __name__ == "__main__":
    demo_dspy_basics()
