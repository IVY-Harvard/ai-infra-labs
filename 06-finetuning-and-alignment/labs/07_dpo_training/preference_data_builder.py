"""
偏好数据构造工具：自动生成 DPO 训练用的偏好对
方法：对同一 prompt 用不同参数生成多个回答，然后自动/手动评分

用法:
    python preference_data_builder.py --model Qwen/Qwen2-7B-Instruct --num 100
    python preference_data_builder.py --model Qwen/Qwen2-7B-Instruct --method best_of_n --n 8
"""

import argparse
import json
import random
import torch
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


class PreferenceDataBuilder:
    """偏好数据构造器"""

    def __init__(self, model_name: str, device: str = "auto"):
        print(f"加载模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_response(self, prompt: str, temperature: float = 0.7,
                         max_new_tokens: int = 512) -> str:
        """生成单个回答"""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
            )

        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

    def best_of_n(self, prompt: str, n: int = 8) -> Dict:
        """Best-of-N 方法：生成 N 个回答，取最好/最差"""
        responses = []
        for i in range(n):
            # 使用不同温度增加多样性
            temp = random.uniform(0.5, 1.0)
            resp = self.generate_response(prompt, temperature=temp)
            responses.append(resp)

        # 用简单启发式评分（实际项目中用 RM 或 GPT-4）
        scored = [(resp, self._simple_score(prompt, resp)) for resp in responses]
        scored.sort(key=lambda x: x[1], reverse=True)

        best = scored[0]
        worst = scored[-1]

        # 只有分差够大才创建偏好对
        if best[1] - worst[1] >= 1.0:
            return {
                "prompt": prompt,
                "chosen": best[0],
                "rejected": worst[0],
                "chosen_score": best[1],
                "rejected_score": worst[1],
            }
        return None

    def contrast_temperatures(self, prompt: str) -> Optional[Dict]:
        """温度对比法：低温 vs 高温生成"""
        # 低温（更确定性、更保守）
        low_temp_resp = self.generate_response(prompt, temperature=0.3)
        # 高温（更随机、更冒险）
        high_temp_resp = self.generate_response(prompt, temperature=1.2)

        # 评分
        low_score = self._simple_score(prompt, low_temp_resp)
        high_score = self._simple_score(prompt, high_temp_resp)

        if low_score != high_score:
            chosen = low_temp_resp if low_score > high_score else high_temp_resp
            rejected = high_temp_resp if low_score > high_score else low_temp_resp
            return {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        return None

    def _simple_score(self, prompt: str, response: str) -> float:
        """简单的启发式评分（实际项目中替换为 RM 或 LLM-as-Judge）"""
        score = 0.0

        # 长度评分（太短扣分，适中加分）
        if len(response) < 20:
            score -= 2
        elif len(response) < 50:
            score -= 1
        elif 50 <= len(response) <= 500:
            score += 1
        elif len(response) > 2000:
            score -= 0.5

        # 结构评分（有列表/分点加分）
        if any(marker in response for marker in ["1.", "2.", "- ", "* ", "："]):
            score += 1

        # 相关性（包含 prompt 中关键词）
        prompt_words = set(prompt)
        response_words = set(response)
        overlap = len(prompt_words & response_words) / max(len(prompt_words), 1)
        score += overlap

        # 完整性（有结尾标点）
        if response.rstrip()[-1:] in ("。", "！", "？", ".", "!", "?"):
            score += 0.5

        return score

    def build_dataset(self, prompts: List[str], method: str = "best_of_n",
                      n: int = 4) -> List[Dict]:
        """批量构建偏好数据"""
        pairs = []
        for i, prompt in enumerate(prompts):
            if method == "best_of_n":
                pair = self.best_of_n(prompt, n=n)
            elif method == "contrast":
                pair = self.contrast_temperatures(prompt)
            else:
                raise ValueError(f"未知方法: {method}")

            if pair:
                pairs.append(pair)

            if (i + 1) % 10 == 0:
                print(f"  已处理 {i+1}/{len(prompts)} prompts, "
                      f"有效对: {len(pairs)}")

        return pairs


def get_demo_prompts(num: int = 100) -> List[str]:
    """获取演示用的 prompts"""
    base_prompts = [
        "什么是人工智能？",
        "如何学习编程？",
        "解释量子力学的基本概念",
        "写一封辞职信",
        "如何做好项目管理？",
        "解释什么是区块链",
        "给我推荐5本必读的编程书",
        "如何保持健康的生活方式？",
        "解释分布式系统的CAP定理",
        "如何准备技术面试？",
    ]
    random.seed(42)
    return [random.choice(base_prompts) for _ in range(num)]


def main():
    parser = argparse.ArgumentParser(description="偏好数据构造工具")
    parser.add_argument("--model", default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--method", choices=["best_of_n", "contrast"], default="best_of_n")
    parser.add_argument("--n", type=int, default=4, help="Best-of-N 的 N 值")
    parser.add_argument("--num", type=int, default=50, help="生成数量")
    parser.add_argument("--prompts_file", default=None)
    parser.add_argument("--output", default="./preference_data.jsonl")
    args = parser.parse_args()

    # 加载 prompts
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts = [json.loads(line)["prompt"] for line in f if line.strip()]
    else:
        prompts = get_demo_prompts(args.num)

    print(f"方法: {args.method}")
    print(f"Prompts: {len(prompts)} 条")

    # 构建偏好数据
    builder = PreferenceDataBuilder(args.model)
    pairs = builder.build_dataset(prompts[:args.num], method=args.method, n=args.n)

    # 保存
    with open(args.output, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n生成 {len(pairs)} 个偏好对，保存到: {args.output}")
    print(f"有效率: {len(pairs)/len(prompts[:args.num]):.1%}")


if __name__ == "__main__":
    main()
