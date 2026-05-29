"""
Evol-Instruct 进化数据生成

通过多种进化策略逐步提升指令的复杂度和多样性：
- 增加约束
- 深化
- 具体化
- 增加推理步骤

用法：
    python evol_instruct.py --input instructions.jsonl --output evolved.jsonl --rounds 3
"""

import os
import json
import time
import random
import argparse
from typing import List, Dict, Callable, Optional


EVOLUTION_STRATEGIES = {
    "add_constraints": {
        "description": "增加约束条件",
        "prompt": (
            "请将以下指令改写为更复杂的版本，通过增加 1-2 个额外的约束条件或要求。"
            "改写后的指令必须合理且可完成。\n\n"
            "原始指令：{instruction}\n\n"
            "改写后的指令："
        ),
    },
    "deepen": {
        "description": "深化问题",
        "prompt": (
            "请将以下指令改写为更有深度的版本，要求更深入的分析或更专业的知识。\n\n"
            "原始指令：{instruction}\n\n"
            "深化后的指令："
        ),
    },
    "concretize": {
        "description": "具体化",
        "prompt": (
            "请将以下指令改写为更具体的版本，用具体的场景、数据或例子替换通用描述。\n\n"
            "原始指令：{instruction}\n\n"
            "具体化后的指令："
        ),
    },
    "increase_reasoning": {
        "description": "增加推理步骤",
        "prompt": (
            "请将以下指令改写为需要多步推理才能完成的版本。\n\n"
            "原始指令：{instruction}\n\n"
            "需要多步推理的指令："
        ),
    },
}


class EvolInstructGenerator:
    """Evol-Instruct 进化数据生成器"""

    def __init__(self, llm_fn: Optional[Callable] = None):
        self.llm_fn = llm_fn or self._mock_llm
        self.evolution_history: List[Dict] = []

    def _mock_llm(self, prompt: str) -> str:
        """模拟 LLM（开发用）"""
        time.sleep(0.05)

        if "增加约束" in prompt or "add_constraints" in prompt:
            return "用 Python 写一个排序算法，要求时间复杂度 O(n log n)，空间复杂度 O(1)，且支持自定义比较函数。"
        elif "深化" in prompt or "deepen" in prompt:
            return "深入解释 Transformer 中多头注意力机制的数学原理，包括 Q/K/V 的计算过程和为什么要使用缩放点积注意力。"
        elif "具体化" in prompt or "concretize" in prompt:
            return "使用 FastAPI 和 PostgreSQL 设计一个图书管理系统的 RESTful API，需要支持图书的 CRUD 操作和按作者/分类的搜索功能。"
        elif "推理" in prompt or "reasoning" in prompt:
            return "一个电商平台有 3 个仓库和 5 个配送区域，已知各仓库库存和各区域需求量，请设计最优配送方案使总运输成本最小。"
        else:
            return "这是一个进化后的指令示例。"

    def evolve(self, instruction: str,
               strategy: str = None) -> Optional[str]:
        """对一条指令执行一次进化"""
        if strategy is None:
            strategy = random.choice(list(EVOLUTION_STRATEGIES.keys()))

        config = EVOLUTION_STRATEGIES[strategy]
        prompt = config["prompt"].format(instruction=instruction)

        evolved = self.llm_fn(prompt).strip()

        # 基础验证
        if not self._validate_evolution(instruction, evolved):
            return None

        return evolved

    def generate_response(self, instruction: str) -> str:
        """为进化后的指令生成回复"""
        prompt = f"请为以下指令生成详细、准确的回复。\n\n指令：{instruction}\n\n回复："
        return self.llm_fn(prompt).strip()

    def _validate_evolution(self, original: str, evolved: str) -> bool:
        """验证进化结果"""
        if not evolved or len(evolved) < 10:
            return False
        if evolved == original:
            return False
        if len(evolved) > 2000:
            return False
        # 确保确实比原始更复杂（长度增加或关键词增加）
        if len(evolved) < len(original) * 0.8:
            return False
        return True

    def run_evolution(self, seed_instructions: List[str],
                      num_rounds: int = 3,
                      evolutions_per_instruction: int = 2) -> List[Dict]:
        """多轮进化"""
        print(f"Evol-Instruct 启动")
        print(f"  种子指令数: {len(seed_instructions)}")
        print(f"  进化轮数: {num_rounds}")
        print(f"  每条进化次数: {evolutions_per_instruction}")
        print()

        all_results = []
        current_pool = list(seed_instructions)

        for round_idx in range(num_rounds):
            print(f"--- 第 {round_idx + 1} 轮 ---")
            new_instructions = []
            round_results = []

            for i, instruction in enumerate(current_pool):
                for _ in range(evolutions_per_instruction):
                    strategy = random.choice(
                        list(EVOLUTION_STRATEGIES.keys())
                    )
                    evolved = self.evolve(instruction, strategy)

                    if evolved:
                        response = self.generate_response(evolved)
                        result = {
                            "instruction": evolved,
                            "response": response,
                            "source_instruction": instruction,
                            "strategy": strategy,
                            "round": round_idx,
                        }
                        round_results.append(result)
                        new_instructions.append(evolved)

            all_results.extend(round_results)
            current_pool = new_instructions

            print(f"  生成: {len(round_results)} 条")
            print(f"  累计: {len(all_results)} 条")

            # 打印示例
            if round_results:
                sample = random.choice(round_results)
                print(f"  示例: [{sample['strategy']}] "
                      f"{sample['instruction'][:60]}...")

        self.evolution_history = all_results
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Evol-Instruct 生成")
    parser.add_argument("--input", type=str, default=None,
                       help="输入指令文件(jsonl)")
    parser.add_argument("--output", type=str,
                       default="/tmp/evol_instruct_output.jsonl")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--evolutions-per", type=int, default=2)
    args = parser.parse_args()

    # 加载种子指令
    if args.input and os.path.exists(args.input):
        with open(args.input) as f:
            seeds = [json.loads(line)["instruction"] for line in f]
    else:
        seeds = [
            "写一个排序算法",
            "解释什么是机器学习",
            "写一个数据处理脚本",
            "比较两种编程语言的优缺点",
            "设计一个简单的 API",
        ]

    generator = EvolInstructGenerator()
    results = generator.run_evolution(
        seeds,
        num_rounds=args.rounds,
        evolutions_per_instruction=args.evolutions_per,
    )

    # 保存
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n结果保存到: {args.output} ({len(results)} 条)")


if __name__ == "__main__":
    main()
