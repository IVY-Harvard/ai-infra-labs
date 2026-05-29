"""
人工评估辅助工具：A/B 盲评界面
对比两个模型的输出质量，支持命令行评估

用法:
    python human_eval_tool.py --model_a ./sft_model --model_b ./dpo_model --num 20
    python human_eval_tool.py --results eval_log.json --analyze
"""

import argparse
import json
import random
import os
import torch
from datetime import datetime
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


class HumanEvalTool:
    """人工评估工具"""

    def __init__(self):
        self.models = {}
        self.results = []

    def load_model(self, name: str, path: str):
        """加载模型"""
        print(f"加载 {name}: {path}")
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.models[name] = {"model": model, "tokenizer": tokenizer}

    def generate(self, name: str, prompt: str) -> str:
        """生成回答"""
        m = self.models[name]
        messages = [{"role": "user", "content": prompt}]
        text = m["tokenizer"].apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = m["tokenizer"](text, return_tensors="pt").to(m["model"].device)
        with torch.no_grad():
            outputs = m["model"].generate(
                **inputs, max_new_tokens=500, temperature=0.7, do_sample=True,
            )
        return m["tokenizer"].decode(
            outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )

    def run_blind_eval(self, prompts: List[str], save_path: str = "eval_log.json"):
        """运行盲评"""
        model_names = list(self.models.keys())
        assert len(model_names) == 2, "需要恰好 2 个模型"

        print("\n" + "=" * 60)
        print("人工盲评开始")
        print("每次展示两个回答（随机顺序），请选择更好的")
        print("输入: A / B / tie / skip")
        print("输入 'quit' 退出")
        print("=" * 60)

        for i, prompt in enumerate(prompts):
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(prompts)}] 问题: {prompt}")
            print("=" * 60)

            # 生成回答
            resp_a = self.generate(model_names[0], prompt)
            resp_b = self.generate(model_names[1], prompt)

            # 随机化顺序（消除位置偏见）
            if random.random() < 0.5:
                display = [("A", resp_a, model_names[0]), ("B", resp_b, model_names[1])]
            else:
                display = [("A", resp_b, model_names[1]), ("B", resp_a, model_names[0])]

            # 展示（盲评，不显示模型名）
            print(f"\n回答 A:")
            print(f"  {display[0][1][:500]}")
            print(f"\n回答 B:")
            print(f"  {display[1][1][:500]}")

            # 获取评价
            while True:
                choice = input("\n你的选择 (A/B/tie/skip/quit): ").strip().lower()
                if choice in ("a", "b", "tie", "skip", "quit"):
                    break
                print("无效输入，请输入 A/B/tie/skip/quit")

            if choice == "quit":
                break

            if choice != "skip":
                # 记录结果
                winner = None
                if choice == "a":
                    winner = display[0][2]
                elif choice == "b":
                    winner = display[1][2]
                else:
                    winner = "tie"

                self.results.append({
                    "prompt": prompt,
                    "model_a": model_names[0],
                    "model_b": model_names[1],
                    "response_a": resp_a,
                    "response_b": resp_b,
                    "winner": winner,
                    "display_order": [display[0][2], display[1][2]],
                    "timestamp": datetime.now().isoformat(),
                })

                # 揭示模型
                print(f"  (A 是 {display[0][2]}, B 是 {display[1][2]})")

        # 保存结果
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"\n评估日志已保存到: {save_path}")

        # 统计
        self._print_statistics(model_names)

    def _print_statistics(self, model_names):
        """打印统计结果"""
        if not self.results:
            return

        print("\n" + "=" * 60)
        print("评估统计")
        print("=" * 60)

        wins = {name: 0 for name in model_names}
        wins["tie"] = 0

        for r in self.results:
            if r["winner"] in model_names:
                wins[r["winner"]] += 1
            else:
                wins["tie"] += 1

        total = len(self.results)
        for name, count in wins.items():
            print(f"  {name}: {count} ({100*count/total:.1f}%)")

        # 胜率
        if len(model_names) == 2:
            a, b = model_names
            a_wins = wins[a]
            b_wins = wins[b]
            print(f"\n  {a} vs {b}: {a_wins}W-{b_wins}L-{wins['tie']}T")
            if a_wins + b_wins > 0:
                win_rate = a_wins / (a_wins + b_wins)
                print(f"  {a} 胜率(不含平局): {win_rate:.1%}")


def analyze_results(results_file: str):
    """分析已有评估结果"""
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"分析 {len(results)} 条评估记录")

    # 统计
    model_names = set()
    wins = {}
    for r in results:
        model_names.add(r["model_a"])
        model_names.add(r["model_b"])
        winner = r["winner"]
        wins[winner] = wins.get(winner, 0) + 1

    print("\n胜负统计:")
    for name, count in sorted(wins.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count} ({100*count/len(results):.1f}%)")


def get_default_prompts() -> List[str]:
    """默认测试 prompts"""
    return [
        "什么是量子计算？请用简单的语言解释。",
        "帮我写一首关于秋天的诗。",
        "如何在工作中提高效率？",
        "解释什么是区块链技术。",
        "Python 和 Java 哪个更适合初学者？为什么？",
        "如何面对生活中的挫折？",
        "解释机器学习中的过拟合问题。",
        "给我推荐三本科幻小说并说明推荐理由。",
    ]


def main():
    parser = argparse.ArgumentParser(description="人工评估工具")
    parser.add_argument("--model_a", help="模型 A 路径")
    parser.add_argument("--model_b", help="模型 B 路径")
    parser.add_argument("--num", type=int, default=10, help="评估数量")
    parser.add_argument("--prompts_file", default=None)
    parser.add_argument("--results", default=None, help="分析已有结果")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--output", default="eval_log.json")
    args = parser.parse_args()

    if args.analyze and args.results:
        analyze_results(args.results)
        return

    if not args.model_a or not args.model_b:
        parser.error("需要 --model_a 和 --model_b")

    # 加载 prompts
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts = [json.loads(line)["prompt"] for line in f if line.strip()]
    else:
        prompts = get_default_prompts()

    prompts = prompts[:args.num]

    # 初始化
    tool = HumanEvalTool()
    tool.load_model("Model_A", args.model_a)
    tool.load_model("Model_B", args.model_b)

    # 运行盲评
    tool.run_blind_eval(prompts, save_path=args.output)


if __name__ == "__main__":
    main()
