"""
同步 Checkpoint 实现

演示最基础的 Checkpoint 保存方式及其对训练的阻塞影响。

用法：
    python sync_checkpoint.py --model-size 100 --save-dir /tmp/ckpt/sync --steps 10
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """简单模型用于 Checkpoint 测试"""

    def __init__(self, hidden_size: int = 4096, num_layers: int = 6):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            ])
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def param_size_mb(self) -> float:
        total = sum(p.numel() * p.element_size() for p in self.parameters())
        return total / 1024 / 1024


def save_checkpoint_sync(model, optimizer, step, save_dir):
    """同步 Checkpoint 保存

    特点：
    - 简单直接
    - 训练完全暂停直到写入完成
    - 使用临时文件 + 原子重命名防止损坏
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"checkpoint_step_{step}.pt")
    tmp_path = save_path + ".tmp"

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    t_start = time.perf_counter()
    torch.save(checkpoint, tmp_path)

    # fsync 确保数据落盘
    with open(tmp_path, "rb") as f:
        os.fsync(f.fileno())

    os.rename(tmp_path, save_path)
    save_time = time.perf_counter() - t_start

    size_mb = os.path.getsize(save_path) / 1024 / 1024
    print(f"  [Sync] Step {step}: saved {size_mb:.1f}MB in {save_time:.2f}s "
          f"({size_mb/save_time:.1f} MB/s)")

    return save_time


def simulate_training_step(model, optimizer, step_time: float = 0.5):
    """模拟一个训练步骤"""
    # 模拟前向+反向传播
    x = torch.randn(32, 4096)
    output = model(x)
    loss = output.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # 模拟计算时间
    time.sleep(step_time)


def main():
    parser = argparse.ArgumentParser(description="同步 Checkpoint 测试")
    parser.add_argument("--model-size", type=int, default=100,
                       help="模型大小(MB)")
    parser.add_argument("--save-dir", type=str, default="/tmp/ckpt/sync")
    parser.add_argument("--steps", type=int, default=10,
                       help="训练步数")
    parser.add_argument("--ckpt-interval", type=int, default=3,
                       help="每 N 步保存一次")
    args = parser.parse_args()

    # 创建模型（调整层数以接近目标大小）
    # 每层 Linear(4096,4096) ≈ 64MB
    num_layers = max(1, args.model_size // 64)
    model = SimpleModel(hidden_size=4096, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print(f"模型参数大小: {model.param_size_mb():.1f}MB")
    print(f"训练步数: {args.steps}, 每 {args.ckpt_interval} 步保存")
    print(f"保存目录: {args.save_dir}")
    print()

    total_train_time = 0
    total_save_time = 0

    for step in range(1, args.steps + 1):
        # 训练
        t0 = time.perf_counter()
        simulate_training_step(model, optimizer, step_time=0.2)
        train_time = time.perf_counter() - t0
        total_train_time += train_time

        # Checkpoint
        if step % args.ckpt_interval == 0:
            save_time = save_checkpoint_sync(model, optimizer, step, args.save_dir)
            total_save_time += save_time

    # 汇总
    print(f"\n{'='*50}")
    print(f"汇总:")
    print(f"  总训练时间: {total_train_time:.2f}s")
    print(f"  总保存时间: {total_save_time:.2f}s")
    print(f"  保存开销占比: {total_save_time/(total_train_time+total_save_time)*100:.1f}%")
    print(f"  GPU 利用率损失: {total_save_time/(total_train_time+total_save_time)*100:.1f}%")


if __name__ == "__main__":
    main()
