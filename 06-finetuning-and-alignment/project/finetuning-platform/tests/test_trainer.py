"""
训练器单元测试
"""

import pytest
import os
import json
import tempfile
from unittest.mock import patch, MagicMock
from datasets import Dataset

# 将 src 添加到路径
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestTrainConfig:
    """测试训练配置"""

    def test_default_config(self):
        from src.trainer.base_trainer import TrainConfig
        config = TrainConfig()
        assert config.model_name == "Qwen/Qwen2-7B"
        assert config.num_epochs == 3
        assert config.batch_size == 4
        assert config.learning_rate == 2e-4
        assert config.bf16 is True

    def test_custom_config(self):
        from src.trainer.base_trainer import TrainConfig
        config = TrainConfig(
            model_name="meta-llama/Llama-3-8B",
            num_epochs=5,
            learning_rate=1e-4,
        )
        assert config.model_name == "meta-llama/Llama-3-8B"
        assert config.num_epochs == 5

    def test_lora_config(self):
        from src.trainer.lora_trainer import LoRATrainConfig
        config = LoRATrainConfig(lora_r=32, lora_alpha=64)
        assert config.lora_r == 32
        assert config.lora_alpha == 64
        assert config.use_qlora is False

    def test_dpo_config(self):
        from src.trainer.dpo_trainer import DPOTrainConfig
        config = DPOTrainConfig(beta=0.2, loss_type="ipo")
        assert config.beta == 0.2
        assert config.loss_type == "ipo"


class TestDataLoader:
    """测试数据加载器"""

    def test_detect_messages_format(self):
        from src.data.data_loader import DataLoader

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            data = {"messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"}
            ]}
            f.write(json.dumps(data) + "\n")
            f.flush()
            temp_path = f.name

        try:
            tokenizer = MagicMock()
            loader = DataLoader(tokenizer)
            detected = loader._detect_format(temp_path)
            assert detected == "messages"
        finally:
            os.unlink(temp_path)

    def test_detect_alpaca_format(self):
        from src.data.data_loader import DataLoader

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            data = {"instruction": "translate", "input": "hello", "output": "你好"}
            f.write(json.dumps(data) + "\n")
            f.flush()
            temp_path = f.name

        try:
            tokenizer = MagicMock()
            loader = DataLoader(tokenizer)
            detected = loader._detect_format(temp_path)
            assert detected == "alpaca"
        finally:
            os.unlink(temp_path)

    def test_detect_preference_format(self):
        from src.data.data_loader import DataLoader

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            data = {"prompt": "q", "chosen": "good", "rejected": "bad"}
            f.write(json.dumps(data) + "\n")
            f.flush()
            temp_path = f.name

        try:
            tokenizer = MagicMock()
            loader = DataLoader(tokenizer)
            detected = loader._detect_format(temp_path)
            assert detected == "preference"
        finally:
            os.unlink(temp_path)


class TestQualityFilter:
    """测试质量过滤器"""

    def test_filter_empty(self):
        from src.data.quality_filter import QualityFilter

        qf = QualityFilter()
        assert qf._filter_empty({"text": "hello"}) is True
        assert qf._filter_empty({"text": ""}) is False
        assert qf._filter_empty({"text": "   "}) is False

    def test_filter_length(self):
        from src.data.quality_filter import QualityFilter

        qf = QualityFilter({"min_length": 5, "max_length": 100})
        assert qf._filter_length({"text": "hello world"}) is True
        assert qf._filter_length({"text": "hi"}) is False
        assert qf._filter_length({"text": "x" * 200}) is False

    def test_filter_special_chars(self):
        from src.data.quality_filter import QualityFilter

        qf = QualityFilter()
        assert qf._filter_special_chars({"text": "normal text"}) is True
        assert qf._filter_special_chars({"text": "has\x00null"}) is False


class TestDistributedConfig:
    """测试分布式配置"""

    def test_auto_detect_single(self):
        from src.trainer.distributed_trainer import DistributedTrainer, DistributedConfig

        dist_config = DistributedConfig(strategy="auto", num_gpus=1)
        base_trainer = MagicMock()
        base_trainer.config = MagicMock()
        base_trainer.config.output_dir = "/tmp/test"

        dt = DistributedTrainer(base_trainer, dist_config)
        strategy = dt._auto_detect_strategy()
        assert strategy == "single"

    def test_auto_detect_ddp(self):
        from src.trainer.distributed_trainer import DistributedTrainer, DistributedConfig

        dist_config = DistributedConfig(strategy="auto", num_gpus=4)
        base_trainer = MagicMock()
        base_trainer.config = MagicMock()
        base_trainer.config.output_dir = "/tmp/test"

        dt = DistributedTrainer(base_trainer, dist_config)
        strategy = dt._auto_detect_strategy()
        assert strategy == "ddp"

    def test_auto_detect_deepspeed(self):
        from src.trainer.distributed_trainer import DistributedTrainer, DistributedConfig

        dist_config = DistributedConfig(strategy="auto", num_gpus=8)
        base_trainer = MagicMock()
        base_trainer.config = MagicMock()
        base_trainer.config.output_dir = "/tmp/test"

        dt = DistributedTrainer(base_trainer, dist_config)
        strategy = dt._auto_detect_strategy()
        assert strategy == "deepspeed_z2"


class TestEvalResult:
    """测试评估结果"""

    def test_eval_result_defaults(self):
        from src.evaluation.evaluator import EvalResult
        result = EvalResult()
        assert result.passed is False
        assert result.benchmarks == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
