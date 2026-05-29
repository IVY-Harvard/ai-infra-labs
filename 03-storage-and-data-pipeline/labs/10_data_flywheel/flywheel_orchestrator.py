"""
数据飞轮编排器

编排完整的数据飞轮闭环：
1. 从反馈数据中提取候选训练样本
2. 通过标注流水线验证质量
3. 与现有训练集合并
4. 触发模型增量训练
5. 评估新模型
6. 部署更新

用法：
    python flywheel_orchestrator.py --feedback-dir /tmp/feedback --output-dir /tmp/flywheel
    python flywheel_orchestrator.py --demo
"""

import os
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict
from dataclasses import dataclass, asdict


@dataclass
class FlywheelIteration:
    """飞轮迭代记录"""
    iteration_id: int
    timestamp: str
    feedback_samples: int
    annotated_samples: int
    training_samples: int
    model_version: str
    eval_metrics: Dict
    status: str


class FlywheelOrchestrator:
    """数据飞轮编排器"""

    def __init__(self, feedback_dir: str, output_dir: str,
                 training_data_dir: str = None):
        self.feedback_dir = feedback_dir
        self.output_dir = output_dir
        self.training_data_dir = training_data_dir or os.path.join(
            output_dir, "training_data")
        self.iterations: List[FlywheelIteration] = []
        self.current_model_version = "v1.0"

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.training_data_dir, exist_ok=True)

    def collect_feedback(self) -> List[Dict]:
        """步骤 1: 收集反馈数据"""
        print("\n[Step 1] 收集反馈数据...")

        samples = []
        if os.path.exists(self.feedback_dir):
            for filename in sorted(os.listdir(self.feedback_dir)):
                if filename.endswith(".jsonl"):
                    filepath = os.path.join(self.feedback_dir, filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                samples.append(json.loads(line.strip()))
                            except json.JSONDecodeError:
                                continue

        print(f"  收集到 {len(samples)} 条反馈")
        return samples

    def filter_and_annotate(self, samples: List[Dict]) -> List[Dict]:
        """步骤 2: 过滤和标注"""
        print("\n[Step 2] 过滤和标注...")

        annotated = []
        for sample in samples:
            # 简单过滤规则
            instruction = sample.get("instruction", sample.get("user_input", ""))
            response = sample.get("chosen", sample.get("model_output", ""))

            if len(instruction) < 5 or len(response) < 10:
                continue

            # 根据反馈类型决定标签
            feedback_type = sample.get("feedback_type", sample.get("source", ""))

            if "positive" in feedback_type or "thumbs_up" in feedback_type:
                sample["label"] = "good"
                sample["use_for_training"] = True
            elif "edit" in feedback_type:
                sample["label"] = "improved"
                sample["use_for_training"] = True
            elif "negative" in feedback_type or "thumbs_down" in feedback_type:
                sample["label"] = "bad"
                sample["use_for_training"] = True  # 用作 DPO 负样本
            else:
                sample["label"] = "unknown"
                sample["use_for_training"] = False

            annotated.append(sample)

        trainable = [s for s in annotated if s.get("use_for_training")]
        print(f"  标注完成: {len(annotated)} 条, "
              f"可用于训练: {len(trainable)} 条")

        return trainable

    def merge_training_data(self, new_samples: List[Dict]) -> str:
        """步骤 3: 合并到训练集"""
        print("\n[Step 3] 合并训练数据...")

        iteration_id = len(self.iterations) + 1
        output_file = os.path.join(
            self.training_data_dir,
            f"iteration_{iteration_id:03d}.jsonl"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            for sample in new_samples:
                # 统一格式
                training_sample = {
                    "instruction": sample.get("instruction",
                                             sample.get("user_input", "")),
                    "chosen": sample.get("chosen",
                                        sample.get("edited_output",
                                                   sample.get("model_output", ""))),
                }
                if "rejected" in sample:
                    training_sample["rejected"] = sample["rejected"]

                f.write(json.dumps(training_sample, ensure_ascii=False) + "\n")

        print(f"  写入 {len(new_samples)} 条到 {output_file}")
        return output_file

    def trigger_training(self, training_file: str) -> str:
        """步骤 4: 触发训练（模拟）"""
        print("\n[Step 4] 触发模型训练...")

        # 在实际环境中，这里会：
        # 1. 提交训练任务到调度系统
        # 2. 等待训练完成
        # 3. 获取新模型路径

        time.sleep(0.5)  # 模拟训练时间
        new_version = f"v{len(self.iterations) + 1}.{1}"
        print(f"  训练完成，新版本: {new_version}")
        print(f"  (实际环境中此处应调用训练脚本)")

        return new_version

    def evaluate_model(self, model_version: str) -> Dict:
        """步骤 5: 评估模型（模拟）"""
        print("\n[Step 5] 评估新模型...")

        # 模拟评估指标
        import random
        metrics = {
            "accuracy": 0.85 + random.random() * 0.1,
            "user_satisfaction": 0.80 + random.random() * 0.15,
            "safety_score": 0.95 + random.random() * 0.05,
            "latency_p50_ms": 100 + random.random() * 50,
        }

        print(f"  准确率: {metrics['accuracy']:.3f}")
        print(f"  用户满意度: {metrics['user_satisfaction']:.3f}")
        print(f"  安全分: {metrics['safety_score']:.3f}")

        return metrics

    def deploy_decision(self, metrics: Dict,
                        min_accuracy: float = 0.85) -> bool:
        """步骤 6: 部署决策"""
        print("\n[Step 6] 部署决策...")

        should_deploy = (
            metrics["accuracy"] >= min_accuracy and
            metrics["safety_score"] >= 0.95
        )

        if should_deploy:
            print(f"  ✓ 满足部署条件，准备上线")
        else:
            print(f"  ✗ 未满足部署条件，保留当前版本")

        return should_deploy

    def run_iteration(self) -> FlywheelIteration:
        """运行一次飞轮迭代"""
        iteration_id = len(self.iterations) + 1
        print(f"\n{'#'*60}")
        print(f"# 飞轮迭代 #{iteration_id}")
        print(f"# 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*60}")

        # 执行各步骤
        feedback = self.collect_feedback()
        annotated = self.filter_and_annotate(feedback)
        training_file = self.merge_training_data(annotated)
        new_version = self.trigger_training(training_file)
        metrics = self.evaluate_model(new_version)
        deployed = self.deploy_decision(metrics)

        if deployed:
            self.current_model_version = new_version

        # 记录迭代
        iteration = FlywheelIteration(
            iteration_id=iteration_id,
            timestamp=datetime.now().isoformat(),
            feedback_samples=len(feedback),
            annotated_samples=len(annotated),
            training_samples=len(annotated),
            model_version=new_version,
            eval_metrics=metrics,
            status="deployed" if deployed else "rejected",
        )
        self.iterations.append(iteration)

        # 保存迭代记录
        history_path = os.path.join(self.output_dir, "flywheel_history.json")
        with open(history_path, "w") as f:
            json.dump([asdict(it) for it in self.iterations], f, indent=2)

        return iteration

    def print_history(self):
        """打印飞轮历史"""
        print(f"\n{'='*60}")
        print("飞轮迭代历史:")
        print(f"{'='*60}")
        print(f"{'#':<4} {'时间':<20} {'反馈':<6} {'训练':<6} "
              f"{'版本':<8} {'状态':<10}")
        print("-" * 60)

        for it in self.iterations:
            print(f"{it.iteration_id:<4} "
                  f"{it.timestamp[:19]:<20} "
                  f"{it.feedback_samples:<6} "
                  f"{it.training_samples:<6} "
                  f"{it.model_version:<8} "
                  f"{it.status:<10}")


def run_demo(output_dir: str):
    """运行演示"""
    feedback_dir = os.path.join(output_dir, "demo_feedback")
    os.makedirs(feedback_dir, exist_ok=True)

    # 创建模拟反馈数据
    demo_feedback = [
        {"user_input": "什么是深度学习？",
         "model_output": "深度学习是机器学习的一个分支，使用多层神经网络来学习数据的表示。",
         "feedback_type": "thumbs_up", "source": "feedback_positive"},
        {"user_input": "写一个快速排序",
         "model_output": "快排是一种排序。",
         "feedback_type": "thumbs_down", "source": "feedback_negative"},
        {"user_input": "解释 TCP 三次握手",
         "model_output": "TCP 握手。",
         "edited_output": "TCP 三次握手是建立连接的过程：客户端发送 SYN，服务端回复 SYN+ACK，客户端再发送 ACK，连接建立。",
         "feedback_type": "edit", "source": "feedback_edit"},
    ]

    with open(os.path.join(feedback_dir, "demo_feedback.jsonl"), "w") as f:
        for item in demo_feedback * 10:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 运行飞轮
    orchestrator = FlywheelOrchestrator(
        feedback_dir=feedback_dir,
        output_dir=output_dir,
    )

    # 运行 3 次迭代
    for _ in range(3):
        orchestrator.run_iteration()

    orchestrator.print_history()


def main():
    parser = argparse.ArgumentParser(description="数据飞轮编排器")
    parser.add_argument("--feedback-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="/tmp/flywheel")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        run_demo(args.output_dir)
    else:
        if not args.feedback_dir:
            print("请指定 --feedback-dir 或使用 --demo")
            return

        orchestrator = FlywheelOrchestrator(
            feedback_dir=args.feedback_dir,
            output_dir=args.output_dir,
        )
        orchestrator.run_iteration()
        orchestrator.print_history()


if __name__ == "__main__":
    main()
