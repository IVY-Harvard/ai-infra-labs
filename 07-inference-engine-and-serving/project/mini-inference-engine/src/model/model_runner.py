"""
Model Runner — 模型执行器

加载模型、准备输入、执行前向传播、采样输出。
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict
from dataclasses import dataclass

from ..core.sequence import SamplerOutput, SamplingParams
from ..core.scheduler import SchedulerOutput
from ..core.block_manager import BlockManager


class ModelRunner:
    """
    模型执行器

    职责:
    - 加载 HuggingFace 模型
    - 准备模型输入 (token ids, positions, attention mask)
    - 执行前向传播
    - 采样输出 token
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        block_size: int = 16,
        num_gpu_blocks: int = 256,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.device = device if torch.cuda.is_available() else "cpu"

        self.model = None
        self.tokenizer = None

    def load_model(self):
        """加载模型和 tokenizer"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[ModelRunner] Loading model: {self.model_name}")
        print(f"[ModelRunner] Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()

        print(f"[ModelRunner] Model loaded successfully")
        print(f"[ModelRunner] Vocab size: {self.tokenizer.vocab_size}")

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        block_manager: BlockManager,
    ) -> SamplerOutput:
        """
        执行模型前向传播

        简化实现: 不使用 PagedAttention kernel，
        而是用 HuggingFace 的标准 forward。
        实际 vLLM 使用自定义 PagedAttention CUDA kernel。
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        all_token_ids = []

        for seq_group in scheduler_output.scheduled_seq_groups:
            for seq in seq_group.get_unfinished_seqs():
                # 简化: 每次传入完整序列 (实际 vLLM 只传新 token + KV Cache)
                # 这里为了简单用 HuggingFace 原生推理
                all_token_ids.append(seq.all_token_ids)

        if not all_token_ids:
            return SamplerOutput(token_ids=[])

        # 采样
        output_tokens = []
        for token_ids in all_token_ids:
            with torch.no_grad():
                input_ids = torch.tensor([token_ids], device=self.device)
                outputs = self.model(input_ids)
                logits = outputs.logits[:, -1, :]  # [1, vocab_size]

                # 获取对应序列的采样参数 (简化: 用默认)
                next_token = self._sample(logits, temperature=0.7, top_p=0.9)
                output_tokens.append(next_token)

        return SamplerOutput(token_ids=output_tokens)

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
    ) -> int:
        """采样下一个 token"""
        if temperature == 0:
            # Greedy
            return logits.argmax(dim=-1).item()

        # 温度缩放
        logits = logits / temperature

        # Top-K
        if top_k > 0:
            top_k_values, _ = torch.topk(logits, top_k)
            min_value = top_k_values[:, -1:]
            logits = torch.where(logits < min_value, torch.full_like(logits, -float('inf')), logits)

        # Top-P (nucleus sampling)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)

            # 移除累积概率超过 top_p 的 token
            sorted_mask = cumulative_probs - probs > top_p
            sorted_logits[sorted_mask] = -float('inf')

            # 恢复原始顺序
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        # 采样
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token.item()
