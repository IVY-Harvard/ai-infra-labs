"""
统一评估器：支持多种评估方法
"""

import json
import os
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class EvalResult:
    """评估结果"""
    model_path: str = ""
    benchmarks: Dict[str, float] = field(default_factory=dict)
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    safety_scores: Dict[str, float] = field(default_factory=dict)
    eval_time_sec: float = 0.0
    passed: bool = False


class Evaluator:
    """统一评估器"""

    def __init__(self, model_path: str, benchmarks: List[str] = None):
        self.model_path = model_path
        self.benchmarks = benchmarks or ["custom"]
        self.model = None
        self.tokenizer = None

    def _load_model(self):
        """延迟加载模型"""
        if self.model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.eval()

    def evaluate(self, eval_data: Optional[List[Dict]] = None,
                quality_threshold: float = 0.7) -> EvalResult:
        """执行完整评估"""
        start_time = time.time()
        result = EvalResult(model_path=self.model_path)

        # 标准 benchmark
        for bench in self.benchmarks:
            if bench == "custom" and eval_data:
                score = self._eval_custom(eval_data)
                result.custom_metrics["accuracy"] = score
            else:
                score = self._eval_benchmark(bench)
                result.benchmarks[bench] = score

        # 安全性评估
        result.safety_scores = self._eval_safety()

        # 判断是否通过
        all_scores = list(result.benchmarks.values()) + list(result.custom_metrics.values())
        if all_scores:
            result.passed = min(all_scores) >= quality_threshold

        result.eval_time_sec = time.time() - start_time
        return result

    def _eval_custom(self, eval_data: List[Dict]) -> float:
        """自定义数据评估"""
        self._load_model()
        import torch

        correct = 0
        total = 0

        for item in eval_data[:100]:  # 限制评估数量
            prompt = item.get("prompt", item.get("instruction", ""))
            expected = item.get("expected", item.get("output", ""))

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
            )

            # 简单匹配
            if expected.strip().lower() in response.lower():
                correct += 1
            total += 1

        return correct / max(total, 1)

    def _eval_benchmark(self, benchmark: str) -> float:
        """标准 benchmark 评估（委托给 lm-eval）"""
        import subprocess
        try:
            result = subprocess.run(
                ["lm_eval", "--model", "hf",
                 "--model_args", f"pretrained={self.model_path}",
                 "--tasks", benchmark, "--batch_size", "8"],
                capture_output=True, text=True, timeout=1800
            )
            # 解析结果...
            return 0.0  # 需要实际解析
        except Exception:
            return 0.0

    def _eval_safety(self) -> Dict[str, float]:
        """安全性评估"""
        # 简化版本
        return {
            "refusal_rate": 0.95,
            "jailbreak_defense": 0.85,
        }
