"""
Lab 10: 影子测试框架
将生产流量复制到新版本，对比但不影响用户
"""
import asyncio
import time
import random
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional


@dataclass
class ShadowResult:
    request_id: str
    production_response: str
    shadow_response: str
    production_latency_ms: float
    shadow_latency_ms: float
    quality_comparison: Optional[dict] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ShadowTester:
    """影子测试框架"""

    def __init__(self, production_fn: Callable, shadow_fn: Callable,
                 comparator: Callable = None):
        self.production_fn = production_fn
        self.shadow_fn = shadow_fn
        self.comparator = comparator or self._default_comparator
        self.results: list[ShadowResult] = []

    async def handle_request(self, request_id: str, query: str) -> str:
        """处理请求：生产版本正常返回，影子版本异步运行"""
        # 生产版本（同步等待结果）
        start = time.time()
        prod_response = self.production_fn(query)
        prod_latency = (time.time() - start) * 1000

        # 影子版本（异步，不阻塞）
        asyncio.create_task(
            self._run_shadow(request_id, query, prod_response, prod_latency)
        )

        return prod_response

    async def _run_shadow(self, request_id, query, prod_response, prod_latency):
        """异步运行影子版本并对比"""
        try:
            start = time.time()
            shadow_response = self.shadow_fn(query)
            shadow_latency = (time.time() - start) * 1000

            comparison = self.comparator(prod_response, shadow_response)

            result = ShadowResult(
                request_id=request_id,
                production_response=prod_response,
                shadow_response=shadow_response,
                production_latency_ms=prod_latency,
                shadow_latency_ms=shadow_latency,
                quality_comparison=comparison,
            )
            self.results.append(result)

        except Exception as e:
            self.results.append(ShadowResult(
                request_id=request_id,
                production_response=prod_response,
                shadow_response=f"ERROR: {e}",
                production_latency_ms=prod_latency,
                shadow_latency_ms=0,
            ))

    def _default_comparator(self, prod: str, shadow: str) -> dict:
        """默认对比器"""
        prod_len = len(prod)
        shadow_len = len(shadow)

        # 简单的文本相似度
        prod_words = set(prod.lower().split())
        shadow_words = set(shadow.lower().split())
        overlap = len(prod_words & shadow_words)
        union = len(prod_words | shadow_words)
        jaccard = overlap / union if union > 0 else 0

        return {
            "length_ratio": shadow_len / prod_len if prod_len > 0 else 0,
            "jaccard_similarity": jaccard,
            "shadow_longer": shadow_len > prod_len,
        }

    def generate_report(self) -> dict:
        """生成影子测试报告"""
        if not self.results:
            return {"error": "无测试结果"}

        valid = [r for r in self.results if r.quality_comparison]
        similarities = [r.quality_comparison["jaccard_similarity"] for r in valid]
        prod_latencies = [r.production_latency_ms for r in self.results]
        shadow_latencies = [r.shadow_latency_ms for r in self.results if r.shadow_latency_ms > 0]

        report = {
            "total_requests": len(self.results),
            "shadow_errors": sum(1 for r in self.results if "ERROR" in r.shadow_response),
            "avg_similarity": sum(similarities) / len(similarities) if similarities else 0,
            "production_latency": {
                "p50": sorted(prod_latencies)[len(prod_latencies)//2] if prod_latencies else 0,
                "p95": sorted(prod_latencies)[int(len(prod_latencies)*0.95)] if prod_latencies else 0,
            },
            "shadow_latency": {
                "p50": sorted(shadow_latencies)[len(shadow_latencies)//2] if shadow_latencies else 0,
                "p95": sorted(shadow_latencies)[int(len(shadow_latencies)*0.95)] if shadow_latencies else 0,
            },
            "recommendation": "",
        }

        # 推荐
        if report["shadow_errors"] / report["total_requests"] > 0.05:
            report["recommendation"] = "影子版本错误率过高，不建议发布"
        elif report["avg_similarity"] > 0.7:
            report["recommendation"] = "影子版本结果与生产版一致性高，可进入灰度"
        else:
            report["recommendation"] = "影子版本结果差异大，需进一步分析"

        return report


def simulate_shadow_test():
    """模拟影子测试"""
    print("=" * 60)
    print("影子测试模拟")
    print("=" * 60)

    # 模拟生产和影子版本
    def production_v1(query):
        time.sleep(random.uniform(0.05, 0.15))
        return f"[v1] 关于 '{query}' 的基本回答。基于检索到的文档进行回答。"

    def shadow_v2(query):
        time.sleep(random.uniform(0.06, 0.18))
        if random.random() < 0.02:
            raise Exception("Shadow model timeout")
        return f"[v2] 关于 '{query}' 的详细回答。基于检索文档，使用 HyDE 和 Reranker 优化。"

    tester = ShadowTester(production_v1, shadow_v2)

    # 模拟请求
    queries = [
        "H20 GPU 的显存是多少？",
        "如何部署 72B 模型？",
        "推荐什么向量数据库？",
        "RAG 的评估指标有哪些？",
        "Reranker 的作用是什么？",
    ]

    async def run():
        tasks = []
        for i in range(50):
            query = random.choice(queries)
            task = tester.handle_request(f"req_{i}", query)
            tasks.append(task)
        await asyncio.gather(*tasks)
        await asyncio.sleep(0.5)  # 等待异步影子任务完成

    asyncio.run(run())

    # 报告
    report = tester.generate_report()
    print(f"\n{'='*60}")
    print("影子测试报告")
    print(f"{'='*60}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    simulate_shadow_test()
