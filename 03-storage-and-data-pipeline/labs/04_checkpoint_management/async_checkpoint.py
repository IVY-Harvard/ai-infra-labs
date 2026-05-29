"""
异步 Checkpoint 实现

核心思想：将模型状态快速拷贝到 CPU 内存，然后后台线程写入磁盘。
训练暂停时间 = CPU 拷贝时间（秒级），而非完整的磁盘写入时间（分钟级）。

用法：
    python async_checkpoint.py --model-size 100 --save-dir /tmp/ckpt/async --steps 10
"""

import os
import time
import argparse
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from typing import Optional


class SimpleModel(nn.Module):
    """简单模型用于测试"""

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


class AsyncCheckpointer:
    """异步 Checkpoint 管理器

    工作流程：
    1. save() 被调用时，快速将 state_dict 拷贝到 CPU 内存
    2. 提交后台任务将 CPU 内存中的数据写入磁盘
    3. 训练进程立即继续，不等待磁盘 IO

    注意事项：
    - 需要额外内存（约等于一份模型大小）
    - 同时只允许一个后台写入（避免内存爆炸）
    - 训练结束前需要调用 wait() 确保最后一次写入完成
    """

    def __init__(self, save_dir: str, max_concurrent: int = 1):
        self.save_dir = save_dir
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.pending_future = None
        self.total_save_time = 0      # 阻塞时间（CPU 拷贝）
        self.total_bg_time = 0        # 后台写入时间
        self.save_count = 0
        os.makedirs(save_dir, exist_ok=True)

    def save(self, model, optimizer, step) -> float:
        """异步保存 Checkpoint

        Returns:
            阻塞时间（秒）— 即 CPU 拷贝耗时
        """
        # 等待上一次写入完成（防止内存爆炸）
        if self.pending_future and not self.pending_future.done():
            print(f"  [Async] 等待上一次写入完成...")
            self.pending_future.result()

        # Step 1: 快速拷贝到 CPU（阻塞的，但很快）
        t_copy_start = time.perf_counter()
        cpu_state = {
            "step": step,
            "model_state_dict": {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            },
            "optimizer_state_dict": self._deep_copy_optim(
                optimizer.state_dict()
            ),
        }
        copy_time = time.perf_counter() - t_copy_start
        self.total_save_time += copy_time

        # Step 2: 提交后台写入任务
        save_path = os.path.join(self.save_dir, f"checkpoint_step_{step}.pt")
        self.pending_future = self.executor.submit(
            self._write_checkpoint, cpu_state, save_path, step
        )

        self.save_count += 1
        print(f"  [Async] Step {step}: CPU copy {copy_time:.3f}s, "
              f"写入已提交到后台")

        return copy_time

    def _deep_copy_optim(self, optim_state: dict) -> dict:
        """深拷贝优化器状态到 CPU"""
        new_state = {
            "state": {},
            "param_groups": optim_state["param_groups"],
        }
        for k, v in optim_state["state"].items():
            new_state["state"][k] = {
                sk: sv.cpu().clone() if torch.is_tensor(sv) else sv
                for sk, sv in v.items()
            }
        return new_state

    def _write_checkpoint(self, state_dict: dict, path: str, step: int):
        """后台线程：写入磁盘"""
        tmp_path = path + ".tmp"
        t_start = time.perf_counter()

        torch.save(state_dict, tmp_path)
        with open(tmp_path, "rb") as f:
            os.fsync(f.fileno())
        os.rename(tmp_path, path)

        write_time = time.perf_counter() - t_start
        self.total_bg_time += write_time

        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  [Background] Step {step}: {size_mb:.1f}MB "
              f"written in {write_time:.2f}s "
              f"({size_mb/write_time:.1f} MB/s)")

    def wait(self):
        """等待所有挂起的写入完成"""
        if self.pending_future:
            self.pending_future.result()

    def stats(self) -> dict:
        """获取统计信息"""
        return {
            "save_count": self.save_count,
            "total_blocking_time": self.total_save_time,
            "total_background_time": self.total_bg_time,
            "avg_blocking_time": (self.total_save_time / self.save_count
                                  if self.save_count > 0 else 0),
        }


def simulate_training_step(model, optimizer, step_time: float = 0.2):
    """模拟训练步骤"""
    x = torch.randn(32, 4096)
    output = model(x)
    loss = output.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    time.sleep(step_time)


def main():
    parser = argparse.ArgumentParser(description="异步 Checkpoint 测试")
    parser.add_argument("--model-size", type=int, default=100)
    parser.add_argument("--save-dir", type=str, default="/tmp/ckpt/async")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--ckpt-interval", type=int, default=3)
    args = parser.parse_args()

    num_layers = max(1, args.model_size // 64)
    model = SimpleModel(hidden_size=4096, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    checkpointer = AsyncCheckpointer(args.save_dir)

    print(f"模型参数大小: {model.param_size_mb():.1f}MB")
    print(f"训练步数: {args.steps}, 每 {args.ckpt_interval} 步保存")
    print()

    total_train_time = 0
    total_block_time = 0

    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        simulate_training_step(model, optimizer)
        train_time = time.perf_counter() - t0
        total_train_time += train_time

        if step % args.ckpt_interval == 0:
            block_time = checkpointer.save(model, optimizer, step)
            total_block_time += block_time

    # 等待最后一次写入
    checkpointer.wait()

    # 汇总
    stats = checkpointer.stats()
    print(f"\n{'='*50}")
    print(f"异步 Checkpoint 汇总:")
    print(f"  总训练时间: {total_train_time:.2f}s")
    print(f"  总阻塞时间（CPU拷贝）: {total_block_time:.3f}s")
    print(f"  总后台写入时间: {stats['total_background_time']:.2f}s")
    print(f"  阻塞开销占比: "
          f"{total_block_time/(total_train_time+total_block_time)*100:.2f}%")
    print(f"  vs 同步方式省下: "
          f"{stats['total_background_time'] - total_block_time:.2f}s 训练时间")


if __name__ == "__main__":
    main()
