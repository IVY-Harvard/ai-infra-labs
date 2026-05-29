"""
分布式训练封装：支持 DDP 和 DeepSpeed
"""

import os
import json
from dataclasses import dataclass
from typing import Optional

from .base_trainer import BaseTrainer, TrainConfig


@dataclass
class DistributedConfig:
    """分布式训练配置"""
    strategy: str = "auto"  # auto, ddp, fsdp, deepspeed_z2, deepspeed_z3
    num_gpus: int = 8
    master_port: int = 29500


DEEPSPEED_Z2_CONFIG = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
}

DEEPSPEED_Z3_CONFIG = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_gather_16bit_weights_on_model_save": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
}


class DistributedTrainer:
    """分布式训练管理器"""

    def __init__(self, base_trainer: BaseTrainer, dist_config: DistributedConfig):
        self.base_trainer = base_trainer
        self.dist_config = dist_config

    def setup_distributed(self):
        """配置分布式环境"""
        strategy = self.dist_config.strategy

        if strategy == "auto":
            strategy = self._auto_detect_strategy()

        if strategy.startswith("deepspeed"):
            ds_config_path = self._create_deepspeed_config(strategy)
            self.base_trainer.config.deepspeed_config = ds_config_path

        self.base_trainer.config.num_gpus = self.dist_config.num_gpus

    def _auto_detect_strategy(self) -> str:
        """根据模型大小和 GPU 数自动选择策略"""
        num_gpus = self.dist_config.num_gpus

        if num_gpus == 1:
            return "single"
        elif num_gpus <= 4:
            return "ddp"
        else:
            return "deepspeed_z2"

    def _create_deepspeed_config(self, strategy: str) -> str:
        """创建 DeepSpeed 配置文件"""
        if "z3" in strategy:
            config = DEEPSPEED_Z3_CONFIG
        else:
            config = DEEPSPEED_Z2_CONFIG

        config_path = os.path.join(self.base_trainer.config.output_dir, "ds_config.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        return config_path

    def get_launch_command(self) -> str:
        """生成启动命令"""
        num_gpus = self.dist_config.num_gpus
        strategy = self.dist_config.strategy

        if strategy.startswith("deepspeed"):
            return f"deepspeed --num_gpus {num_gpus} train_script.py"
        else:
            return f"accelerate launch --num_processes {num_gpus} train_script.py"

    def train(self, train_dataset, eval_dataset=None):
        """执行分布式训练"""
        self.setup_distributed()
        return self.base_trainer.train(train_dataset, eval_dataset)
