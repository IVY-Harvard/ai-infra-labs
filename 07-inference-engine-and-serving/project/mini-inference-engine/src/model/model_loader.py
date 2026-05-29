"""模型加载器"""
from typing import Optional
import torch


class ModelLoader:
    """模型加载器 — 支持 HuggingFace 模型"""

    @staticmethod
    def load(model_name: str, dtype: str = "float16", device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = getattr(torch, dtype, torch.float16)
        device = device if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            torch_dtype = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, trust_remote_code=True
        ).to(device).eval()

        return model, tokenizer
