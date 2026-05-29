"""
DPO 训练器：直接偏好优化
"""

import torch
from dataclasses import dataclass
from typing import Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer as TRLDPOTrainer, DPOConfig

from .base_trainer import BaseTrainer, TrainConfig


@dataclass
class DPOTrainConfig(TrainConfig):
    """DPO 训练配置"""
    beta: float = 0.1
    loss_type: str = "sigmoid"  # sigmoid, ipo, hinge
    max_prompt_length: int = 1024
    ref_model_name: Optional[str] = None  # None = 使用同一个模型作为 ref


class DPOTrainerWrapper(BaseTrainer):
    """DPO 训练器"""

    def __init__(self, config: DPOTrainConfig):
        super().__init__(config)
        self.dpo_config = config
        self.ref_model = None

    def setup(self):
        """加载 policy 和 reference 模型"""
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Policy model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        # Reference model
        ref_name = self.dpo_config.ref_model_name or self.config.model_name
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            ref_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        self._configure_model()

    def _configure_model(self):
        """DPO 不需要额外的模型配置"""
        pass

    def _create_trainer(self, train_dataset, eval_dataset=None) -> Any:
        """创建 DPOTrainer"""
        dpo_config = DPOConfig(
            output_dir=self.config.output_dir,
            beta=self.dpo_config.beta,
            loss_type=self.dpo_config.loss_type,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler,
            warmup_ratio=self.config.warmup_ratio,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            num_train_epochs=self.config.num_epochs,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            logging_steps=self.config.logging_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            max_length=self.config.max_seq_length,
            max_prompt_length=self.dpo_config.max_prompt_length,
            seed=self.config.seed,
            report_to="none",
        )

        return TRLDPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=dpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
