"""
用户反馈收集器

收集模型推理过程中的用户反馈，转化为训练数据。

反馈类型：
- thumbs_up: 用户认可 → 正样本
- thumbs_down: 用户不满 → 负样本（DPO）
- edit: 用户修改了回答 → 修改后版本为正样本
- regenerate: 用户要求重新生成 → 原回答可能不佳

用法：
    python feedback_collector.py --output-dir /tmp/feedback --demo
"""

import os
import json
import time
import uuid
import argparse
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum


class FeedbackType(str, Enum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    EDIT = "edit"
    REGENERATE = "regenerate"
    REPORT = "report"


@dataclass
class FeedbackRecord:
    """反馈记录"""
    feedback_id: str
    request_id: str
    timestamp: str
    feedback_type: str
    user_input: str
    model_output: str
    model_version: str = "v1.0"
    edited_output: Optional[str] = None
    rating: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class FeedbackCollector:
    """反馈收集器"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.buffer: List[FeedbackRecord] = []
        self.buffer_size = 100
        self.stats = {
            "total": 0,
            "thumbs_up": 0,
            "thumbs_down": 0,
            "edit": 0,
            "regenerate": 0,
        }

    def collect(self, feedback_type: FeedbackType,
                user_input: str, model_output: str,
                edited_output: str = None,
                rating: int = None,
                tags: List[str] = None,
                model_version: str = "v1.0") -> FeedbackRecord:
        """收集一条反馈"""
        record = FeedbackRecord(
            feedback_id=str(uuid.uuid4()),
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            feedback_type=feedback_type.value,
            user_input=user_input,
            model_output=model_output,
            model_version=model_version,
            edited_output=edited_output,
            rating=rating,
            tags=tags or [],
        )

        self.buffer.append(record)
        self.stats["total"] += 1
        self.stats[feedback_type.value] = self.stats.get(
            feedback_type.value, 0) + 1

        # 缓冲区满则刷盘
        if len(self.buffer) >= self.buffer_size:
            self.flush()

        return record

    def flush(self):
        """将缓冲区写入文件"""
        if not self.buffer:
            return

        date_str = datetime.now().strftime("%Y%m%d")
        filepath = os.path.join(self.output_dir,
                                f"feedback_{date_str}.jsonl")

        with open(filepath, "a", encoding="utf-8") as f:
            for record in self.buffer:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

        count = len(self.buffer)
        self.buffer = []
        return count

    def to_training_samples(self, min_date: str = None) -> List[Dict]:
        """将反馈转换为训练样本

        转换规则：
        - thumbs_up → (instruction, response) 正样本
        - thumbs_down → (instruction, rejected_response) 负样本
        - edit → (instruction, edited_response) 正样本 + 原回答为负样本
        """
        self.flush()  # 确保所有数据已写入

        samples = []
        feedback_dir = self.output_dir

        for filepath in sorted(os.listdir(feedback_dir)):
            if not filepath.endswith(".jsonl"):
                continue

            full_path = os.path.join(feedback_dir, filepath)
            with open(full_path, "r") as f:
                for line in f:
                    record = json.loads(line.strip())
                    sample = self._convert_to_sample(record)
                    if sample:
                        samples.append(sample)

        return samples

    def _convert_to_sample(self, record: Dict) -> Optional[Dict]:
        """转换单条反馈为训练样本"""
        feedback_type = record["feedback_type"]

        if feedback_type == "thumbs_up":
            return {
                "instruction": record["user_input"],
                "chosen": record["model_output"],
                "source": "feedback_positive",
                "timestamp": record["timestamp"],
            }

        elif feedback_type == "edit" and record.get("edited_output"):
            return {
                "instruction": record["user_input"],
                "chosen": record["edited_output"],
                "rejected": record["model_output"],
                "source": "feedback_edit",
                "timestamp": record["timestamp"],
            }

        elif feedback_type == "thumbs_down":
            return {
                "instruction": record["user_input"],
                "rejected": record["model_output"],
                "source": "feedback_negative",
                "timestamp": record["timestamp"],
            }

        return None

    def print_stats(self):
        """打印统计"""
        print(f"\n{'='*50}")
        print(f"反馈收集统计:")
        print(f"  总计: {self.stats['total']}")
        for key, count in self.stats.items():
            if key != "total":
                pct = count / max(self.stats["total"], 1) * 100
                print(f"  {key}: {count} ({pct:.1f}%)")


def run_demo(output_dir: str):
    """运行演示"""
    collector = FeedbackCollector(output_dir)

    print("模拟收集用户反馈...")

    # 模拟反馈数据
    demo_interactions = [
        {
            "type": FeedbackType.THUMBS_UP,
            "input": "什么是梯度下降？",
            "output": "梯度下降是一种优化算法，通过计算损失函数的梯度来迭代更新模型参数，使损失函数值逐步减小。",
        },
        {
            "type": FeedbackType.THUMBS_DOWN,
            "input": "Python 中如何实现单例模式？",
            "output": "Python 没有单例模式。",
        },
        {
            "type": FeedbackType.EDIT,
            "input": "解释 ACID 特性",
            "output": "ACID 是数据库的四个特性。",
            "edited": "ACID 是数据库事务的四个特性：原子性(Atomicity)确保事务要么全部完成要么全部回滚；一致性(Consistency)确保事务将数据库从一个有效状态转换到另一个有效状态；隔离性(Isolation)确保并发事务互不干扰；持久性(Durability)确保已提交的事务永久保存。",
        },
        {
            "type": FeedbackType.REGENERATE,
            "input": "写一首关于编程的诗",
            "output": "编程很好。",
        },
    ]

    for interaction in demo_interactions * 5:
        collector.collect(
            feedback_type=interaction["type"],
            user_input=interaction["input"],
            model_output=interaction["output"],
            edited_output=interaction.get("edited"),
        )

    collector.flush()
    collector.print_stats()

    # 转换为训练样本
    samples = collector.to_training_samples()
    print(f"\n转换为训练样本: {len(samples)} 条")

    # 保存训练样本
    train_path = os.path.join(output_dir, "training_samples.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"训练样本保存到: {train_path}")


def main():
    parser = argparse.ArgumentParser(description="反馈收集器")
    parser.add_argument("--output-dir", type=str,
                       default="/tmp/feedback_data")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        run_demo(args.output_dir)
    else:
        print("使用 --demo 运行演示模式")
        print("生产环境中，集成到 FastAPI 服务中使用")


if __name__ == "__main__":
    main()
