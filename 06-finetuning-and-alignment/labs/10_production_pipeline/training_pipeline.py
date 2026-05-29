"""
端到端训练流水线：从数据到部署的完整流程自动化

用法:
    python training_pipeline.py --config pipeline_config.yaml
    python training_pipeline.py --model Qwen/Qwen2-7B --data ./data/train.jsonl
"""

import argparse
import json
import os
import time
import yaml
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Optional, List


@dataclass
class PipelineConfig:
    """流水线配置"""
    # 实验
    experiment_name: str = "finetune_experiment"
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    # 模型
    model_name: str = "Qwen/Qwen2-7B"
    method: str = "lora"  # lora, qlora, full

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128

    # 数据
    train_data: str = "./data/train.jsonl"
    eval_data: Optional[str] = None
    max_seq_length: int = 2048

    # 训练
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    num_gpus: int = 1

    # 评估
    eval_benchmarks: List[str] = field(default_factory=lambda: ["custom"])
    eval_threshold: float = 0.0  # 通过阈值

    # 输出
    output_dir: str = "./pipeline_output"


class TrainingPipeline:
    """端到端训练流水线"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.run_dir = os.path.join(config.output_dir, config.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.log_file = os.path.join(self.run_dir, "pipeline.log")
        self.metrics = {}

    def log(self, message: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, "a") as f:
            f.write(log_msg + "\n")

    def run(self):
        """执行完整流水线"""
        self.log("=" * 60)
        self.log(f"启动训练流水线: {self.config.experiment_name}")
        self.log(f"Run ID: {self.config.run_id}")
        self.log("=" * 60)

        start_time = time.time()
        success = True

        try:
            # Stage 1: 数据验证
            self.log("\n[Stage 1/5] 数据验证")
            self._validate_data()

            # Stage 2: 训练
            self.log("\n[Stage 2/5] 模型训练")
            model_path = self._train()

            # Stage 3: 评估
            self.log("\n[Stage 3/5] 模型评估")
            eval_results = self._evaluate(model_path)

            # Stage 4: 质量门控
            self.log("\n[Stage 4/5] 质量检查")
            passed = self._quality_gate(eval_results)

            # Stage 5: 注册/部署
            if passed:
                self.log("\n[Stage 5/5] 模型注册")
                self._register_model(model_path, eval_results)
            else:
                self.log("\n[Stage 5/5] 质量未达标，跳过部署")
                success = False

        except Exception as e:
            self.log(f"\n流水线失败: {e}")
            success = False

        # 总结
        elapsed = time.time() - start_time
        self.log(f"\n{'='*60}")
        self.log(f"流水线{'成功' if success else '失败'}")
        self.log(f"总耗时: {elapsed/60:.1f} 分钟")
        self.log(f"输出目录: {self.run_dir}")

        # 保存元数据
        self._save_metadata(success, elapsed)

        return success

    def _validate_data(self):
        """数据验证"""
        self.log(f"  检查训练数据: {self.config.train_data}")

        if not os.path.exists(self.config.train_data):
            # Demo 模式：创建模拟数据
            self.log("  数据文件不存在，创建 demo 数据...")
            self._create_demo_data()

        # 统计
        count = 0
        with open(self.config.train_data, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1

        self.log(f"  数据量: {count} 条")
        self.metrics["data_count"] = count

        if count < 10:
            raise ValueError(f"数据量过少: {count} (最少需要 10 条)")

    def _train(self):
        """执行训练"""
        model_output = os.path.join(self.run_dir, "model")

        self.log(f"  模型: {self.config.model_name}")
        self.log(f"  方法: {self.config.method}")
        self.log(f"  GPU: {self.config.num_gpus}")
        self.log(f"  输出: {model_output}")

        # 构建训练命令
        if self.config.method == "lora":
            self.log("  使用 LoRA 微调...")
            # 实际项目中调用训练脚本
            # subprocess.run(["python", "lora_train.py", ...])
            self.log("  (Demo 模式: 模拟训练完成)")
            os.makedirs(model_output, exist_ok=True)
            # 保存训练配置作为占位
            with open(os.path.join(model_output, "training_config.json"), "w") as f:
                json.dump(asdict(self.config), f, indent=2)

        self.metrics["model_path"] = model_output
        self.metrics["training_time"] = 0  # demo

        return model_output

    def _evaluate(self, model_path: str) -> Dict:
        """评估模型"""
        self.log(f"  评估基准: {self.config.eval_benchmarks}")

        # Demo: 模拟评估结果
        eval_results = {
            "custom_accuracy": 0.85,
            "format_compliance": 0.95,
            "safety_refusal_rate": 0.97,
        }

        self.log(f"  评估结果: {eval_results}")
        self.metrics["eval_results"] = eval_results

        return eval_results

    def _quality_gate(self, eval_results: Dict) -> bool:
        """质量门控"""
        threshold = self.config.eval_threshold

        checks = {
            "custom_accuracy >= 0.7": eval_results.get("custom_accuracy", 0) >= 0.7,
            "safety >= 0.95": eval_results.get("safety_refusal_rate", 0) >= 0.95,
        }

        all_passed = all(checks.values())

        for check, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            self.log(f"  [{status}] {check}")

        return all_passed

    def _register_model(self, model_path: str, eval_results: Dict):
        """注册模型"""
        registry_entry = {
            "version": self.config.run_id,
            "model_path": model_path,
            "base_model": self.config.model_name,
            "method": self.config.method,
            "eval_results": eval_results,
            "status": "registered",
            "created_at": datetime.now().isoformat(),
        }

        registry_file = os.path.join(self.config.output_dir, "model_registry.jsonl")
        with open(registry_file, "a") as f:
            f.write(json.dumps(registry_entry, ensure_ascii=False) + "\n")

        self.log(f"  已注册模型版本: {self.config.run_id}")
        self.log(f"  注册表: {registry_file}")

    def _save_metadata(self, success: bool, elapsed: float):
        """保存流水线元数据"""
        metadata = {
            "config": asdict(self.config),
            "metrics": self.metrics,
            "success": success,
            "elapsed_sec": elapsed,
            "timestamp": datetime.now().isoformat(),
        }

        with open(os.path.join(self.run_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _create_demo_data(self):
        """创建 demo 数据"""
        os.makedirs(os.path.dirname(self.config.train_data) or ".", exist_ok=True)
        with open(self.config.train_data, "w", encoding="utf-8") as f:
            for i in range(100):
                item = {
                    "messages": [
                        {"role": "user", "content": f"问题 {i}"},
                        {"role": "assistant", "content": f"回答 {i}"},
                    ]
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="训练流水线")
    parser.add_argument("--config", default=None, help="配置文件 (YAML)")
    parser.add_argument("--model", default="Qwen/Qwen2-7B")
    parser.add_argument("--data", default="./data/train.jsonl")
    parser.add_argument("--method", default="lora")
    parser.add_argument("--output_dir", default="./pipeline_output")
    args = parser.parse_args()

    # 加载配置
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config_dict = yaml.safe_load(f)
        config = PipelineConfig(**config_dict)
    else:
        config = PipelineConfig(
            model_name=args.model,
            train_data=args.data,
            method=args.method,
            output_dir=args.output_dir,
        )

    # 运行流水线
    pipeline = TrainingPipeline(config)
    success = pipeline.run()

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
