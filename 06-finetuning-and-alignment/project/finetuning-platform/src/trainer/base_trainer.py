"""
训练器基类：定义微调训练的统一接口
"""

import os
import json
import time
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Optional, List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments


@dataclass
class TrainConfig:
    """训练配置"""
    # 模型
    model_name: str = "Qwen/Qwen2-7B"
    model_revision: str = "main"
    torch_dtype: str = "bfloat16"

    # 数据
    train_data: str = ""
    eval_data: Optional[str] = None
    max_seq_length: int = 2048
    data_format: str = "messages"  # messages, sharegpt, alpaca

    # 训练
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42

    # 精度
    bf16: bool = True
    gradient_checkpointing: bool = True

    # 保存
    output_dir: str = "./output"
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 10

    # 分布式
    num_gpus: int = 1
    deepspeed_config: Optional[str] = None


@dataclass
class TrainResult:
    """训练结果"""
    job_id: str = ""
    status: str = "completed"  # completed, failed, cancelled
    model_path: str = ""
    train_loss: float = 0.0
    eval_loss: Optional[float] = None
    training_time_sec: float = 0.0
    peak_memory_gb: float = 0.0
    total_steps: int = 0
    config: Dict = field(default_factory=dict)
    metrics: Dict = field(default_factory=dict)
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class BaseTrainer(ABC):
    """训练器基类"""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self._start_time = None

    def setup(self):
        """初始化模型和 tokenizer"""
        print(f"加载模型: {self.config.model_name}")

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            revision=self.config.model_revision,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            revision=self.config.model_revision,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        # 子类实现具体的模型配置（如添加 LoRA）
        self._configure_model()

    @abstractmethod
    def _configure_model(self):
        """配置模型（由子类实现）"""
        pass

    @abstractmethod
    def _create_trainer(self, train_dataset, eval_dataset) -> Any:
        """创建 trainer（由子类实现）"""
        pass

    def get_training_args(self) -> TrainingArguments:
        """获取训练参数"""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            logging_steps=self.config.logging_steps,
            eval_strategy="steps" if self.config.eval_data else "no",
            eval_steps=self.config.save_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            deepspeed=self.config.deepspeed_config,
            seed=self.config.seed,
            report_to="none",
        )

    def train(self, train_dataset, eval_dataset=None) -> TrainResult:
        """执行训练"""
        self._start_time = time.time()
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # 设置模型
            if self.model is None:
                self.setup()

            # 创建 trainer
            self.trainer = self._create_trainer(train_dataset, eval_dataset)

            # 重置显存统计
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # 训练
            train_output = self.trainer.train()

            # 收集结果
            peak_memory = 0
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / 1e9

            training_time = time.time() - self._start_time

            # 保存
            self.save(self.config.output_dir)

            result = TrainResult(
                job_id=job_id,
                status="completed",
                model_path=self.config.output_dir,
                train_loss=train_output.training_loss,
                training_time_sec=training_time,
                peak_memory_gb=peak_memory,
                total_steps=train_output.global_step,
                config=asdict(self.config),
                metrics=train_output.metrics,
            )

        except Exception as e:
            result = TrainResult(
                job_id=job_id,
                status="failed",
                error=str(e),
                training_time_sec=time.time() - self._start_time,
                config=asdict(self.config),
            )

        # 保存结果
        result_path = os.path.join(self.config.output_dir, "train_result.json")
        os.makedirs(self.config.output_dir, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(asdict(result), f, indent=2)

        return result

    def save(self, output_dir: str):
        """保存模型"""
        os.makedirs(output_dir, exist_ok=True)
        if self.trainer:
            self.trainer.save_model(output_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)

    def get_trainable_params(self) -> Dict:
        """获取可训练参数统计"""
        if self.model is None:
            return {}
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return {
            "trainable": trainable,
            "total": total,
            "trainable_pct": 100 * trainable / total,
        }
