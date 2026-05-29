"""
DPO vs SFT 效果对比
在相同 prompt 上对比 SFT 和 DPO 模型的输出

用法:
    python dpo_vs_sft_comparison.py --sft_model ./sft_output --dpo_model ./dpo_output
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict


class ModelComparer:
    """模型对比器"""

    def __init__(self):
        self.models = {}

    def load_model(self, name: str, path: str):
        """加载模型"""
        print(f"加载 {name}: {path}")
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.models[name] = {"model": model, "tokenizer": tokenizer}

    def generate(self, name: str, prompt: str, max_new_tokens: int = 300) -> str:
        """用指定模型生成"""
        m = self.models[name]
        messages = [{"role": "user", "content": prompt}]
        text = m["tokenizer"].apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = m["tokenizer"](text, return_tensors="pt").to(m["model"].device)

        with torch.no_grad():
            outputs = m["model"].generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
            )

        return m["tokenizer"].decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )

    def compare(self, prompts: List[str]) -> List[Dict]:
        """对比多个 prompt 的输出"""
        results = []
        model_names = list(self.models.keys())

        for i, prompt in enumerate(prompts):
            result = {"prompt": prompt, "responses": {}}
            for name in model_names:
                resp = self.generate(name, prompt)
                result["responses"][name] = resp

            results.append(result)
            print(f"\n[{i+1}/{len(prompts)}] {prompt[:50]}...")
            for name in model_names:
                print(f"  [{name}]: {result['responses'][name][:100]}...")

        return results


def get_test_prompts() -> List[str]:
    """获取测试 prompts"""
    return [
        "什么是人工智能？请简要介绍。",
        "请帮我写一首关于春天的短诗。",
        "解释一下TCP/IP协议栈。",
        "如何评价三国演义这本书？",
        "用Python写一个快速排序算法。",
        "告诉我如何做番茄炒蛋。",
        "比较React和Vue的优缺点。",
        "如何面对工作中的压力？",
    ]


def main():
    parser = argparse.ArgumentParser(description="DPO vs SFT 对比")
    parser.add_argument("--sft_model", required=True, help="SFT 模型路径")
    parser.add_argument("--dpo_model", required=True, help="DPO 模型路径")
    parser.add_argument("--base_model", default=None, help="基座模型路径（可选）")
    parser.add_argument("--prompts_file", default=None)
    parser.add_argument("--output", default="comparison_results.json")
    args = parser.parse_args()

    comparer = ModelComparer()

    # 加载模型
    if args.base_model:
        comparer.load_model("Base", args.base_model)
    comparer.load_model("SFT", args.sft_model)
    comparer.load_model("DPO", args.dpo_model)

    # 加载 prompts
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts = [json.loads(line)["prompt"] for line in f if line.strip()]
    else:
        prompts = get_test_prompts()

    print(f"\n对比 {len(prompts)} 个 prompt")
    print(f"模型: {list(comparer.models.keys())}")

    # 对比
    results = comparer.compare(prompts)

    # 统计
    print("\n" + "=" * 60)
    print("对比统计")
    print("=" * 60)

    for name in comparer.models:
        lengths = [len(r["responses"][name]) for r in results]
        avg_len = sum(lengths) / len(lengths)
        print(f"  {name} 平均回答长度: {avg_len:.0f} 字符")

    # 保存
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n对比结果已保存到: {args.output}")

    # 打印详细对比
    print("\n" + "=" * 60)
    print("详细对比（前 3 个 prompt）")
    print("=" * 60)
    for r in results[:3]:
        print(f"\n问题: {r['prompt']}")
        print("-" * 40)
        for name, resp in r["responses"].items():
            print(f"[{name}]:")
            print(f"  {resp[:200]}...")
            print()


if __name__ == "__main__":
    main()
