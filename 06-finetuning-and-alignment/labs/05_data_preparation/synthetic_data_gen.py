"""
合成数据生成工具：使用 LLM 自动生成微调训练数据

支持的方法:
- Self-Instruct: 从种子指令扩展
- Evol-Instruct: 指令进化（增加复杂度）
- 回答生成: 为指令生成高质量回答

用法:
    python synthetic_data_gen.py --method self_instruct --seed_file seeds.json --num 100
    python synthetic_data_gen.py --method evol_instruct --input instructions.jsonl --num 100
    python synthetic_data_gen.py --method generate_answers --input instructions.jsonl
"""

import argparse
import json
import os
import random
import time
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """生成配置"""
    model_name: str = "Qwen/Qwen2-7B-Instruct"
    temperature: float = 0.8
    top_p: float = 0.9
    max_new_tokens: int = 1024
    api_base: str = "http://localhost:8000/v1"  # vLLM 或 OpenAI API


class LLMClient:
    """LLM 调用客户端"""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self.config.api_base,
                    api_key="not-needed",
                )
            except ImportError:
                print("请安装 openai: pip install openai")
                raise
        return self._client

    def generate(self, prompt: str, system: str = "") -> str:
        """调用 LLM 生成"""
        client = self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_new_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  LLM 调用失败: {e}")
            return ""


class SelfInstructGenerator:
    """Self-Instruct 数据生成器"""

    def __init__(self, client: LLMClient, seed_tasks: List[Dict]):
        self.client = client
        self.seed_tasks = seed_tasks
        self.generated = []

    def generate_instructions(self, num_samples: int) -> List[str]:
        """批量生成新指令"""
        new_instructions = []

        for i in range(0, num_samples, 5):
            batch_size = min(5, num_samples - i)
            examples = random.sample(
                self.seed_tasks + self.generated[-20:],
                min(5, len(self.seed_tasks) + len(self.generated))
            )

            prompt = "以下是一些任务指令的示例：\n\n"
            for j, ex in enumerate(examples, 1):
                inst = ex.get("instruction", ex.get("messages", [{}])[0].get("content", ""))
                prompt += f"{j}. {inst}\n"
            prompt += f"\n请生成{batch_size}个新的、不同类型的任务指令。要求：\n"
            prompt += "- 每个指令独立一行，以数字编号开头\n"
            prompt += "- 涵盖不同的任务类型（翻译、摘要、问答、创作、分析等）\n"
            prompt += "- 指令要具体清晰\n"

            response = self.client.generate(prompt, system="你是一个数据生成专家。")

            # 解析生成的指令
            for line in response.split("\n"):
                line = line.strip()
                if line and line[0].isdigit():
                    # 去掉编号
                    inst = line.lstrip("0123456789.、 ")
                    if len(inst) > 5:
                        new_instructions.append(inst)
                        self.generated.append({"instruction": inst})

            print(f"  已生成 {len(new_instructions)}/{num_samples} 条指令")

        return new_instructions[:num_samples]

    def generate_answer(self, instruction: str) -> str:
        """为指令生成回答"""
        prompt = f"请为以下指令提供一个高质量、详细且准确的回答：\n\n指令：{instruction}\n\n回答："
        return self.client.generate(prompt, system="你是一个知识渊博的AI助手，请提供专业、准确的回答。")


class EvolInstructGenerator:
    """Evol-Instruct 指令进化生成器"""

    EVOLUTION_TEMPLATES = {
        "deepening": """请将以下指令改写得更复杂、更有挑战性。
可以增加约束条件、要求更深入的分析、或引入边界情况。
保持原始任务的主题不变。

原始指令: {instruction}

改写后的指令（只输出改写后的指令，不要额外解释）:""",

        "concretizing": """请将以下抽象指令改写为一个具体的实际场景。
加入具体的数字、名称或情境描述。

原始指令: {instruction}

具体化后的指令:""",

        "reasoning": """请在以下指令的基础上，增加需要推理或解释原因的要求。

原始指令: {instruction}

增加推理要求后的指令:""",

        "multi_step": """请将以下简单指令改写为需要多步骤完成的复杂任务。

原始指令: {instruction}

多步骤版本的指令:""",
    }

    def __init__(self, client: LLMClient):
        self.client = client

    def evolve(self, instruction: str, strategy: str = "random") -> str:
        """进化一条指令"""
        if strategy == "random":
            strategy = random.choice(list(self.EVOLUTION_TEMPLATES.keys()))

        template = self.EVOLUTION_TEMPLATES.get(strategy)
        if not template:
            raise ValueError(f"未知策略: {strategy}")

        prompt = template.format(instruction=instruction)
        evolved = self.client.generate(prompt, system="你是一个指令优化专家。")

        # 清理
        evolved = evolved.strip().strip('"').strip("'")
        return evolved

    def evolve_batch(self, instructions: List[str], num_evolve: int = 1) -> List[str]:
        """批量进化指令"""
        evolved = []
        for i, inst in enumerate(instructions):
            for _ in range(num_evolve):
                new_inst = self.evolve(inst)
                if new_inst and len(new_inst) > 10:
                    evolved.append(new_inst)
            if (i + 1) % 10 == 0:
                print(f"  已进化 {i+1}/{len(instructions)} 条指令")
        return evolved


def generate_dataset(
    client: LLMClient,
    method: str,
    seed_data: List[Dict],
    num_samples: int,
    output_file: str,
):
    """生成完整数据集"""
    results = []

    if method == "self_instruct":
        generator = SelfInstructGenerator(client, seed_data)

        # 生成指令
        print("\n[1/2] 生成指令...")
        instructions = generator.generate_instructions(num_samples)

        # 生成回答
        print("\n[2/2] 生成回答...")
        for i, inst in enumerate(instructions):
            answer = generator.generate_answer(inst)
            if answer:
                results.append({
                    "messages": [
                        {"role": "user", "content": inst},
                        {"role": "assistant", "content": answer},
                    ]
                })
            if (i + 1) % 10 == 0:
                print(f"  已生成 {i+1}/{len(instructions)} 条回答")

    elif method == "evol_instruct":
        generator = EvolInstructGenerator(client)
        si_gen = SelfInstructGenerator(client, seed_data)

        # 提取原始指令
        instructions = []
        for item in seed_data:
            if "instruction" in item:
                instructions.append(item["instruction"])
            elif "messages" in item:
                for msg in item["messages"]:
                    if msg["role"] == "user":
                        instructions.append(msg["content"])

        # 进化指令
        print("\n[1/3] 进化指令...")
        evolved_instructions = generator.evolve_batch(instructions, num_evolve=2)

        # 取需要的数量
        all_instructions = (instructions + evolved_instructions)[:num_samples]

        # 生成回答
        print("\n[2/3] 生成回答...")
        for i, inst in enumerate(all_instructions):
            answer = si_gen.generate_answer(inst)
            if answer:
                results.append({
                    "messages": [
                        {"role": "user", "content": inst},
                        {"role": "assistant", "content": answer},
                    ]
                })
            if (i + 1) % 10 == 0:
                print(f"  已生成 {i+1}/{len(all_instructions)} 条")

    elif method == "generate_answers":
        si_gen = SelfInstructGenerator(client, seed_data)

        print("\n为现有指令生成回答...")
        for i, item in enumerate(seed_data[:num_samples]):
            inst = item.get("instruction", "")
            if not inst and "messages" in item:
                for msg in item["messages"]:
                    if msg["role"] == "user":
                        inst = msg["content"]
                        break

            if inst:
                answer = si_gen.generate_answer(inst)
                if answer:
                    results.append({
                        "messages": [
                            {"role": "user", "content": inst},
                            {"role": "assistant", "content": answer},
                        ]
                    })
            if (i + 1) % 10 == 0:
                print(f"  已处理 {i+1}/{min(len(seed_data), num_samples)}")

    # 保存
    print(f"\n生成完成: {len(results)} 条数据")
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"已保存到: {output_file}")

    return results


def create_demo_seeds() -> List[Dict]:
    """创建演示用的种子数据"""
    return [
        {"instruction": "解释什么是机器学习"},
        {"instruction": "写一首关于秋天的诗"},
        {"instruction": "列出Python的5个核心特性"},
        {"instruction": "如何做番茄炒蛋"},
        {"instruction": "解释TCP和UDP的区别"},
        {"instruction": "给一个5岁小孩解释什么是互联网"},
        {"instruction": "写一封请假邮件"},
        {"instruction": "分析'三国演义'中诸葛亮的性格特点"},
    ]


def main():
    parser = argparse.ArgumentParser(description="合成数据生成工具")
    parser.add_argument("--method", choices=["self_instruct", "evol_instruct", "generate_answers"],
                       default="self_instruct")
    parser.add_argument("--seed_file", default=None, help="种子数据文件")
    parser.add_argument("--input", default=None, help="输入指令文件")
    parser.add_argument("--num", type=int, default=100, help="生成数量")
    parser.add_argument("--output", default="./synthetic_data.jsonl")
    parser.add_argument("--api_base", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2-7B-Instruct")
    args = parser.parse_args()

    # 配置 LLM 客户端
    config = GenerationConfig(
        model_name=args.model,
        api_base=args.api_base,
    )
    client = LLMClient(config)

    # 加载种子数据
    if args.seed_file:
        with open(args.seed_file, "r", encoding="utf-8") as f:
            seed_data = json.load(f)
    elif args.input:
        seed_data = []
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    seed_data.append(json.loads(line))
    else:
        print("使用内置种子数据...")
        seed_data = create_demo_seeds()

    print(f"方法: {args.method}")
    print(f"种子数据: {len(seed_data)} 条")
    print(f"目标数量: {args.num}")
    print(f"LLM: {args.model}")
    print(f"API: {args.api_base}")

    # 生成
    generate_dataset(client, args.method, seed_data, args.num, args.output)


if __name__ == "__main__":
    main()
