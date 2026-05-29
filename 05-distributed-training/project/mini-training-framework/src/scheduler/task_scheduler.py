"""
训练任务调度器
===============
管理多步训练的调度逻辑。
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Callable

import torch


@dataclass
class TrainingState:
    """训练状态"""
    global_step: int = 0
    epoch: int = 0
    total_tokens: int = 0
    best_loss: float = float("inf")
    start_time: float = field(default_factory=time.time)


class TaskScheduler:
    """
    训练任务调度器。
    管理:
      - 学习率调度 (warmup + cosine decay)
      - Gradient accumulation
      - Checkpoint 触发时机
      - Early stopping
    """

    def __init__(
        self,
        max_steps: int = 1000,
        warmup_steps: int = 100,
        lr: float = 3e-4,
        min_lr: float = 3e-5,
        gradient_accumulation_steps: int = 1,
        checkpoint_interval: int = 100,
        log_interval: int = 10,
    ):
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.lr = lr
        self.min_lr = min_lr
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.checkpoint_interval = checkpoint_interval
        self.log_interval = log_interval
        self.state = TrainingState()

    def get_lr(self, step: Optional[int] = None) -> float:
        """计算当前学习率 (warmup + cosine decay)"""
        step = step or self.state.global_step

        if step < self.warmup_steps:
            # Linear warmup
            return self.lr * step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            import math
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            progress = min(1.0, progress)
            return self.min_lr + 0.5 * (self.lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def should_accumulate(self) -> bool:
        """是否应该累积梯度（不更新参数）"""
        return (self.state.global_step + 1) % self.gradient_accumulation_steps != 0

    def should_checkpoint(self) -> bool:
        """是否应该保存 checkpoint"""
        return self.state.global_step % self.checkpoint_interval == 0 and self.state.global_step > 0

    def should_log(self) -> bool:
        """是否应该打印日志"""
        return self.state.global_step % self.log_interval == 0

    def is_finished(self) -> bool:
        """训练是否结束"""
        return self.state.global_step >= self.max_steps

    def step(self, loss: float = 0.0, tokens: int = 0):
        """推进一步"""
        self.state.global_step += 1
        self.state.total_tokens += tokens
        if loss < self.state.best_loss:
            self.state.best_loss = loss

    def update_optimizer_lr(self, optimizer: torch.optim.Optimizer):
        """更新优化器的学习率"""
        lr = self.get_lr()
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
