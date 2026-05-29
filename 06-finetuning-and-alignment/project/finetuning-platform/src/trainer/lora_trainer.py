"""
LoRA 训练器：支持 LoRA 和 QLoRA 微调
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, List, Any
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import BitsAndBytesConfig
from trl import SFTTrainer

from .base_trainer import BaseTrainer, TrainConfig


@dataclass
class LoRATrainConfig(TrainConfig):
    """LoRA 训练配置"""
    # LoRA 参数
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_bias: str = "none"
    use_dora: bool = False

    # QLoRA
    use_qlora: bool = False
    quant_type: str = "nf4"
    use_double_quant: bool = True


class LoRATrainer(BaseTrainer):
    """LoRA/QLoRA 训练器"""

    def __init__(self, config: LoRATrainConfig):
        super().__init__(config)
        self.lora_config = config

    def setup(self):
        """加载模型，可选量化"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # QLoRA: 4-bit 量化加载
        if self.lora_config.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.lora_config.quant_type,
                bnb_4bit_use_double_quant=self.lora_config.use_double_quant,
                bnb_4bit_compute_dtype=torch_dtype,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )

        self._configure_model()

    def _configure_model(self):
        """添加 LoRA adapter"""
        lora_config = LoraConfig(
            r=self.lora_config.lora_r,
            lora_alpha=self.lora_config.lora_alpha,
            target_modules=self.lora_config.lora_target_modules,
            lora_dropout=self.lora_config.lora_dropout,
            bias=self.lora_config.lora_bias,
            use_dora=self.lora_config.use_dora,
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def _create_trainer(self, train_dataset, eval_dataset=None) -> Any:
        """创建 SFTTrainer"""
        training_args = self.get_training_args()

        # QLoRA 使用 paged optimizer
        if self.lora_config.use_qlora:
            training_args.optim = "paged_adamw_8bit"

        return SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_length,
            dataset_text_field="text",
        )

    def merge_and_save(self, output_dir: str):
        """合并 LoRA 权重并保存完整模型"""
        if self.lora_config.use_qlora:
            print("QLoRA 模型需要先反量化再合并")
            print("建议使用 merge_and_export.py 工具")
            return

        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(output_dir, safe_serialization=True)
        self.tokenizer.save_pretrained(output_dir)
        print(f"合并模型已保存到: {output_dir}")
