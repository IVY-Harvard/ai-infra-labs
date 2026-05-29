"""
统一评估运行器：支持多种评估方法的统一接口

用法:
    python eval_runner.py --model ./my_model --benchmarks mmlu ceval
    python eval_runner.py --model ./my_model --custom_eval custom_test.jsonl
"""

import argparse
import json
import os
import time
import torch
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, field


@dataclass
class EvalConfig:
    """评估配置"""
    model_path: str
    benchmarks: List[str] = field(default_factory=list)
    custom_eval_file: Optional[str] = None
    batch_size: int = 8
    max_length: int = 2048
    num_few_shot: int = 5
    output_dir: str = "./eval_results"


class ModelEvaluator:
    """统一模型评估器"""

    def __init__(self, model_path: str, device: str = "auto"):
        print(f"加载评估模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """生成回答"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
            )
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )

    def get_choice_logprobs(self, prompt: str, choices: List[str]) -> Dict[str, float]:
        """获取各选项的 log probability（适用于多选题）"""
        logprobs = {}
        for choice in choices:
            full_text = prompt + choice
            inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # 取最后一个 token 的 logprob
            last_logit = outputs.logits[0, -1]
            choice_id = self.tokenizer.encode(choice, add_special_tokens=False)[0]
            logprobs[choice] = last_logit[choice_id].item()

        return logprobs

    def evaluate_multichoice(self, questions: List[Dict]) -> Dict:
        """评估多选题"""
        correct = 0
        total = 0
        results_by_subject = {}

        for q in questions:
            prompt = q["prompt"]
            choices = q.get("choices", ["A", "B", "C", "D"])
            answer = q["answer"]
            subject = q.get("subject", "general")

            # 方法 1: 生成后提取答案
            response = self.generate(prompt, max_new_tokens=10)
            predicted = self._extract_choice(response, choices)

            is_correct = predicted == answer
            if is_correct:
                correct += 1
            total += 1

            results_by_subject.setdefault(subject, []).append(is_correct)

        return {
            "accuracy": correct / max(total, 1),
            "correct": correct,
            "total": total,
            "by_subject": {
                s: sum(r) / len(r) for s, r in results_by_subject.items()
            }
        }

    def evaluate_generation(self, questions: List[Dict]) -> Dict:
        """评估生成任务（用于 GSM8K 等需要自由生成的任务）"""
        correct = 0
        total = 0

        for q in questions:
            prompt = q["prompt"]
            expected = q["answer"]

            response = self.generate(prompt, max_new_tokens=512)

            # 提取数字答案
            predicted = self._extract_number(response)
            expected_num = self._extract_number(str(expected))

            if predicted is not None and expected_num is not None:
                if abs(predicted - expected_num) < 1e-5:
                    correct += 1
            total += 1

        return {
            "accuracy": correct / max(total, 1),
            "correct": correct,
            "total": total,
        }

    def evaluate_custom(self, test_data: List[Dict]) -> Dict:
        """自定义评估"""
        results = []
        for item in test_data:
            prompt = item.get("prompt", item.get("instruction", ""))
            expected = item.get("expected", item.get("output", ""))

            response = self.generate(prompt)
            # 简单匹配
            score = 1.0 if expected.strip() in response else 0.0
            results.append({
                "prompt": prompt[:100],
                "expected": expected[:100],
                "response": response[:200],
                "score": score,
            })

        avg_score = sum(r["score"] for r in results) / max(len(results), 1)
        return {"score": avg_score, "details": results[:20]}

    def _extract_choice(self, text: str, choices: List[str]) -> str:
        """从生成文本中提取选择"""
        text = text.strip().upper()
        for c in choices:
            if c.upper() in text[:5]:
                return c
        return choices[0]

    def _extract_number(self, text: str) -> Optional[float]:
        """从文本中提取数字"""
        import re
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return float(numbers[-1]) if numbers else None


def run_evaluation(config: EvalConfig):
    """运行完整评估"""
    evaluator = ModelEvaluator(config.model_path)
    results = {}

    for benchmark in config.benchmarks:
        print(f"\n{'='*40}")
        print(f"评估: {benchmark}")
        print(f"{'='*40}")

        start_time = time.time()

        if benchmark == "mmlu":
            # 使用 lm-eval-harness 或自建简化版
            print("  建议使用 lm-eval: lm_eval --model hf --tasks mmlu")
            results[benchmark] = {"note": "请使用 lm-eval-harness"}

        elif benchmark == "custom" and config.custom_eval_file:
            with open(config.custom_eval_file, "r", encoding="utf-8") as f:
                test_data = [json.loads(line) for line in f if line.strip()]
            results[benchmark] = evaluator.evaluate_custom(test_data)

        else:
            print(f"  基准 {benchmark}: 建议使用 lm-eval-harness 运行")
            results[benchmark] = {"note": f"使用 lm_eval --tasks {benchmark}"}

        elapsed = time.time() - start_time
        results[benchmark]["time_sec"] = elapsed

    return results


def main():
    parser = argparse.ArgumentParser(description="统一模型评估器")
    parser.add_argument("--model", required=True, help="模型路径")
    parser.add_argument("--benchmarks", nargs="+", default=["mmlu", "ceval"],
                       help="要运行的基准测试")
    parser.add_argument("--custom_eval", default=None, help="自定义评测文件")
    parser.add_argument("--output_dir", default="./eval_results")
    args = parser.parse_args()

    config = EvalConfig(
        model_path=args.model,
        benchmarks=args.benchmarks,
        custom_eval_file=args.custom_eval,
        output_dir=args.output_dir,
    )

    results = run_evaluation(config)

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "eval_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n评估结果已保存到: {output_file}")

    # 打印摘要
    print("\n" + "=" * 60)
    print("评估摘要")
    print("=" * 60)
    for bench, result in results.items():
        if "accuracy" in result:
            print(f"  {bench}: {result['accuracy']:.2%}")
        elif "score" in result:
            print(f"  {bench}: {result['score']:.2%}")
        else:
            print(f"  {bench}: {result.get('note', 'N/A')}")


if __name__ == "__main__":
    main()
