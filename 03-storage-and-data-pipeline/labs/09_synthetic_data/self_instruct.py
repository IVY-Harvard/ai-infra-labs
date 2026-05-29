"""
Self-Instruct 数据生成

从少量种子指令出发，让 LLM 生成新的指令和回复，滚雪球式扩大数据集。

流程：
1. 从种子集随机采样 few-shot 示例
2. 让 LLM 生成新指令
3. 让 LLM 为新指令生成回复
4. 过滤低质量生成
5. 加入种子集，循环

用法：
    python self_instruct.py --num-generate 100 --output /tmp/self_instruct_output.jsonl
"""

import os
import json
import time
import random
import argparse
from typing import List, Dict, Optional, Callable


# 默认种子指令集
DEFAULT_SEEDS = [
    {
        "instruction": "给出三个保持健康的小贴士。",
        "response": "1. 保持均衡饮食：多吃蔬菜、水果和全谷物，限制加工食品和含糖饮料。\n2. 规律运动：每周至少进行 150 分钟中等强度有氧运动。\n3. 充足睡眠：成年人每晚需要 7-9 小时的优质睡眠。"
    },
    {
        "instruction": "用 Python 写一个函数，计算列表中所有偶数的和。",
        "response": "```python\ndef sum_even(numbers):\n    return sum(n for n in numbers if n % 2 == 0)\n\n# 示例\nprint(sum_even([1, 2, 3, 4, 5, 6]))  # 输出: 12\n```"
    },
    {
        "instruction": "解释什么是机器学习中的过拟合，以及如何避免。",
        "response": "过拟合是指模型在训练数据上表现很好，但在新数据上泛化能力差。模型学会了训练数据中的噪声而非真正的规律。\n\n避免方法：\n1. 增加训练数据量\n2. 使用正则化（L1/L2/Dropout）\n3. 早停（Early Stopping）\n4. 数据增强\n5. 简化模型结构"
    },
    {
        "instruction": "将以下句子翻译成英文：今天天气很好，适合出去散步。",
        "response": "The weather is beautiful today, perfect for a walk outside."
    },
    {
        "instruction": "写一首关于秋天的短诗。",
        "response": "秋风起，叶渐黄，\n一夜寒霜满地霜。\n归雁南飞排成行，\n丰收时节稻谷香。"
    },
]


class SelfInstructGenerator:
    """Self-Instruct 数据生成器"""

    GENERATE_INSTRUCTION_PROMPT = """你是一个指令生成器。请根据以下示例，生成一条新的、不同于示例的指令。

示例指令：
{examples}

请生成一条新的指令（只输出指令内容，不要编号或前缀）："""

    GENERATE_RESPONSE_PROMPT = """请为以下指令生成一个高质量的回复。

指令：{instruction}

回复："""

    def __init__(self, llm_fn: Optional[Callable] = None,
                 seed_instructions: List[Dict] = None):
        """
        Args:
            llm_fn: LLM 推理函数，接受 prompt 返回文本
            seed_instructions: 种子指令列表
        """
        self.llm_fn = llm_fn or self._mock_llm
        self.seeds = seed_instructions or DEFAULT_SEEDS
        self.generated: List[Dict] = []
        self.pool = list(self.seeds)  # 活跃指令池

    def _mock_llm(self, prompt: str) -> str:
        """模拟 LLM 输出（开发/演示用）"""
        time.sleep(0.1)  # 模拟延迟

        if "生成一条新的指令" in prompt:
            templates = [
                "解释深度学习中 Transformer 架构的核心组件。",
                "用 Python 实现一个简单的 LRU 缓存。",
                "比较 PostgreSQL 和 MongoDB 的优缺点。",
                "给初学者推荐 3 本人工智能入门书籍。",
                "写一个 Bash 脚本，监控磁盘使用率并在超过 80% 时发送警告。",
                "解释 CAP 定理及其在分布式系统中的意义。",
                "如何在 Kubernetes 中实现滚动更新？",
                "写一个 SQL 查询，找出每个部门薪资最高的员工。",
            ]
            return random.choice(templates)
        else:
            return (
                "这是一个模拟的回复。在实际使用中，"
                "需要连接真正的 LLM API（如 OpenAI、vLLM 等）"
                "来生成高质量的回复。回复应当准确、详细且有教育意义。"
            )

    def generate_instruction(self, num_examples: int = 3) -> str:
        """生成新指令"""
        # 随机采样 few-shot 示例
        examples = random.sample(self.pool, min(num_examples, len(self.pool)))
        examples_text = "\n".join(
            f"- {ex['instruction']}" for ex in examples
        )

        prompt = self.GENERATE_INSTRUCTION_PROMPT.format(
            examples=examples_text
        )
        return self.llm_fn(prompt).strip()

    def generate_response(self, instruction: str) -> str:
        """为指令生成回复"""
        prompt = self.GENERATE_RESPONSE_PROMPT.format(
            instruction=instruction
        )
        return self.llm_fn(prompt).strip()

    def is_valid(self, instruction: str, response: str) -> bool:
        """基础质量检查"""
        # 过短
        if len(instruction) < 10 or len(response) < 20:
            return False

        # 过长
        if len(instruction) > 500 or len(response) > 5000:
            return False

        # 与已有指令太相似（简单去重）
        for existing in self.pool:
            if instruction.lower() == existing["instruction"].lower():
                return False

        # 包含不当内容的标记
        if any(marker in instruction.lower()
               for marker in ["sorry", "i cannot", "as an ai"]):
            return False

        return True

    def run(self, num_generate: int = 100, verbose: bool = True) -> List[Dict]:
        """运行 Self-Instruct 生成"""
        print(f"Self-Instruct 生成启动")
        print(f"  种子数: {len(self.seeds)}")
        print(f"  目标生成: {num_generate}")
        print()

        generated = 0
        attempted = 0

        while generated < num_generate:
            attempted += 1

            # 生成新指令
            instruction = self.generate_instruction()

            # 生成回复
            response = self.generate_response(instruction)

            # 质量检查
            if self.is_valid(instruction, response):
                sample = {
                    "instruction": instruction,
                    "response": response,
                    "source": "self_instruct",
                    "round": generated // 10,
                }
                self.generated.append(sample)
                self.pool.append(sample)
                generated += 1

                if verbose and generated % 10 == 0:
                    print(f"  进度: {generated}/{num_generate} "
                          f"(尝试 {attempted} 次, "
                          f"通过率 {generated/attempted*100:.1f}%)")
            else:
                if verbose and attempted % 20 == 0:
                    print(f"  过滤: {attempted - generated} 个低质量生成")

        print(f"\n完成: 生成 {generated} 条, "
              f"尝试 {attempted} 次, "
              f"通过率 {generated/attempted*100:.1f}%")

        return self.generated


def main():
    parser = argparse.ArgumentParser(description="Self-Instruct 数据生成")
    parser.add_argument("--num-generate", type=int, default=50)
    parser.add_argument("--seed-file", type=str, default=None)
    parser.add_argument("--output", type=str,
                       default="/tmp/self_instruct_output.jsonl")
    parser.add_argument("--api-key", type=str, default=None,
                       help="OpenAI API key (可选)")
    args = parser.parse_args()

    # 加载种子
    seeds = DEFAULT_SEEDS
    if args.seed_file and os.path.exists(args.seed_file):
        with open(args.seed_file) as f:
            seeds = [json.loads(line) for line in f]

    # 创建生成器
    generator = SelfInstructGenerator(seed_instructions=seeds)

    # 运行
    results = generator.run(num_generate=args.num_generate)

    # 保存
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n结果保存到: {args.output}")
    print(f"样本展示:")
    for item in results[:3]:
        print(f"  指令: {item['instruction'][:60]}...")
        print(f"  回复: {item['response'][:60]}...")
        print()


if __name__ == "__main__":
    main()
