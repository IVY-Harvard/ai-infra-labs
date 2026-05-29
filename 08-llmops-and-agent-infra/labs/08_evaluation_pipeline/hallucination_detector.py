"""
Lab 08: 幻觉检测器
多策略检测 LLM 回答中的幻觉
"""
import os
import re
from typing import Optional
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser


LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-72b-instruct")


@dataclass
class HallucinationResult:
    score: float  # 0=无幻觉, 1=完全幻觉
    claims: list[dict]  # 每个声明的检查结果
    unsupported_claims: list[str]
    method: str


class HallucinationDetector:
    """多策略幻觉检测器"""

    def __init__(self):
        self.llm = ChatOpenAI(
            base_url=LLM_BASE_URL, model=LLM_MODEL,
            api_key="not-needed", temperature=0,
        )

    def detect(self, question: str, answer: str,
               context: list[str]) -> dict:
        """综合多策略检测幻觉"""
        results = {}

        # 策略 1: 声明分解 + 验证
        results["claim_verification"] = self.claim_based_detection(
            answer, context
        )

        # 策略 2: NLI 方式
        results["nli"] = self.nli_based_detection(answer, context)

        # 策略 3: 直接判断
        results["direct_judge"] = self.direct_judge(
            question, answer, context
        )

        # 综合分数
        scores = [r.score for r in results.values()]
        avg_score = sum(scores) / len(scores)

        return {
            "hallucination_score": avg_score,
            "is_hallucinated": avg_score > 0.3,
            "details": {k: {"score": v.score, "method": v.method}
                       for k, v in results.items()},
            "unsupported_claims": results["claim_verification"].unsupported_claims,
        }

    def claim_based_detection(self, answer: str,
                               context: list[str]) -> HallucinationResult:
        """策略 1: 声明分解 → 逐一验证"""
        # Step 1: 分解声明
        decompose_prompt = ChatPromptTemplate.from_template(
            """请将以下文本分解为独立的事实性陈述。每行一个陈述。
只输出事实性陈述，跳过主观评价和连接词。

文本：{answer}

事实性陈述列表："""
        )
        chain = decompose_prompt | self.llm | StrOutputParser()
        claims_text = chain.invoke({"answer": answer})
        claims = [c.strip() for c in claims_text.strip().split("\n") if c.strip()]

        # Step 2: 验证每个声明
        context_text = "\n".join(context)
        verify_prompt = ChatPromptTemplate.from_template(
            """判断以下声明是否能从上下文中推导出来。

上下文：
{context}

声明：{claim}

请只回答：SUPPORTED（有支持）/ NOT_SUPPORTED（无支持）/ PARTIALLY（部分支持）
判断："""
        )
        verify_chain = verify_prompt | self.llm | StrOutputParser()

        claim_results = []
        unsupported = []

        for claim in claims[:10]:  # 限制最多 10 个声明
            verdict = verify_chain.invoke({
                "context": context_text, "claim": claim
            }).strip()

            is_supported = "SUPPORTED" in verdict and "NOT" not in verdict
            claim_results.append({
                "claim": claim,
                "verdict": verdict,
                "supported": is_supported,
            })
            if not is_supported:
                unsupported.append(claim)

        score = len(unsupported) / len(claims) if claims else 0

        return HallucinationResult(
            score=score,
            claims=claim_results,
            unsupported_claims=unsupported,
            method="claim_verification",
        )

    def nli_based_detection(self, answer: str,
                            context: list[str]) -> HallucinationResult:
        """策略 2: NLI (Natural Language Inference) 方式"""
        context_text = "\n".join(context)

        nli_prompt = ChatPromptTemplate.from_template(
            """你是一个自然语言推理专家。判断假设是否能从前提推导出来。

前提（上下文）：
{context}

假设（回答）：
{answer}

请判断：
- ENTAILMENT（蕴含）：回答完全可以从上下文推导
- CONTRADICTION（矛盾）：回答与上下文矛盾
- NEUTRAL（中性）：回答中有信息既不能确认也不能否定

判断结果（只输出一个词）："""
        )

        chain = nli_prompt | self.llm | StrOutputParser()
        result = chain.invoke({"context": context_text, "answer": answer}).strip()

        if "ENTAILMENT" in result:
            score = 0.0
        elif "CONTRADICTION" in result:
            score = 1.0
        else:  # NEUTRAL
            score = 0.5

        return HallucinationResult(
            score=score, claims=[], unsupported_claims=[],
            method="nli",
        )

    def direct_judge(self, question: str, answer: str,
                     context: list[str]) -> HallucinationResult:
        """策略 3: 直接让 LLM 判断"""
        context_text = "\n".join(context)

        judge_prompt = ChatPromptTemplate.from_template(
            """请判断以下回答是否包含幻觉（即回答中有信息不是来自上下文）。

上下文：
{context}

问题：{question}
回答：{answer}

请用 0-10 分评估幻觉程度（0=完全忠实于上下文，10=完全编造）：
分数：""")

        chain = judge_prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "context": context_text, "question": question, "answer": answer,
        })

        try:
            score_match = re.search(r'\d+', result)
            score = int(score_match.group()) / 10 if score_match else 0.5
        except (ValueError, AttributeError):
            score = 0.5

        return HallucinationResult(
            score=score, claims=[], unsupported_claims=[],
            method="direct_judge",
        )


def main():
    print("=" * 60)
    print("幻觉检测器演示")
    print("=" * 60)

    detector = HallucinationDetector()

    # 测试用例
    test_cases = [
        {
            "question": "H20 GPU 的显存是多少？",
            "answer": "H20 GPU 拥有 96GB HBM3 显存，带宽为 4TB/s。",
            "context": ["H20 GPU 搭载 96GB HBM3 显存，带宽为 4TB/s。"],
            "expected": "无幻觉",
        },
        {
            "question": "H20 GPU 的显存是多少？",
            "answer": "H20 GPU 拥有 128GB HBM3 显存，是目前最强的推理卡，性能超越 A100 十倍。",
            "context": ["H20 GPU 搭载 96GB HBM3 显存，带宽为 4TB/s。"],
            "expected": "有幻觉（128GB错误，十倍不在上下文中）",
        },
        {
            "question": "推荐什么向量数据库？",
            "answer": "推荐使用 Milvus，它由 Google 开发，是全球最流行的向量数据库。",
            "context": ["Milvus 是 Zilliz 开源的向量数据库，适合大规模场景。"],
            "expected": "有幻觉（Google开发是错误的）",
        },
    ]

    for case in test_cases:
        print(f"\n{'─'*60}")
        print(f"问题: {case['question']}")
        print(f"回答: {case['answer']}")
        print(f"预期: {case['expected']}")

        result = detector.detect(
            case["question"], case["answer"], case["context"]
        )

        print(f"\n检测结果:")
        print(f"  幻觉分数: {result['hallucination_score']:.2f}")
        print(f"  判定: {'有幻觉' if result['is_hallucinated'] else '无幻觉'}")
        if result["unsupported_claims"]:
            print(f"  不支持的声明:")
            for claim in result["unsupported_claims"]:
                print(f"    - {claim}")


if __name__ == "__main__":
    main()
