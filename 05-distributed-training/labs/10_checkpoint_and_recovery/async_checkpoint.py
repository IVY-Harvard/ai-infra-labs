"""
Lab 10 - 异步 Checkpoint
==========================
在后台线程保存 checkpoint，训练不中断。

关键技术:
  1. 快照 state_dict (CPU copy)
  2. 在后台线程写入磁盘
  3. 训练主线程继续下一步

注意: state_dict 必须 copy 到 CPU（避免 GPU tensor 被修改）

运行:
    torchrun --nproc_per_node=4 async_checkpoint.py
"""

import os
import time
import tempfile
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class AsyncCheckpointer:
    """
    异步 Checkpoint 管理器。
    使用后台线程池保存 checkpoint，不阻塞训练。
    """

    def __init__(self, save_dir: str, max_concurrent: int = 1):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.pending_futures = []
        self._lock = threading.Lock()

    def save_async(self, state_dict: dict, step: int, rank: int):
        """
        异步保存 checkpoint。

        关键: state_dict 中的 tensor 已经 copy 到 CPU，
              所以可以安全地在后台线程写入，不影响 GPU 训练。
        """
        save_path = os.path.join(self.save_dir, f"step_{step}_rank_{rank}.pt")

        future = self.executor.submit(self._save_worker, state_dict, save_path)
        with self._lock:
            self.pending_futures.append(future)

    @staticmethod
    def _save_worker(state_dict: dict, save_path: str):
        """后台线程的保存函数"""
        torch.save(state_dict, save_path)
        return save_path

    def wait_all(self):
        """等待所有异步保存完成"""
        with self._lock:
            for f in self.pending_futures:
                f.result()
            self.pending_futures.clear()

    def shutdown(self):
        self.wait_all()
        self.executor.shutdown()


def snapshot_state_dict(model, optimizer, step):
    """
    快照: 将 state_dict 中的 tensor copy 到 CPU。
    这是异步保存的关键 — 确保后台线程操作的数据不会被训练修改。
    """
    # CPU copy 模型参数
    model_state = {
        k: v.cpu().clone() for k, v in model.module.state_dict().items()
    }
    # CPU copy 优化器状态（包含 momentum 等）
    opt_state = {}
    raw_state = optimizer.state_dict()
    opt_state["param_groups"] = raw_state["param_groups"]
    opt_state["state"] = {}
    for k, v in raw_state["state"].items():
        opt_state["state"][k] = {
            sk: sv.cpu().clone() if isinstance(sv, torch.Tensor) else sv
            for sk, sv in v.items()
        }

    return {
        "step": step,
        "model_state_dict": model_state,
        "optimizer_state_dict": opt_state,
    }


class SimpleModel(nn.Module):
    def __init__(self, hidden_size=1024, num_layers=8):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    model = SimpleModel().to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    save_dir = tempfile.mkdtemp(prefix="async_ckpt_")
    checkpointer = AsyncCheckpointer(save_dir)

    if rank == 0:
        print("异步 Checkpoint 演示")
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  模型参数: {num_params/1e6:.1f}M")
        print(f"  保存目录: {save_dir}")

    total_steps = 20
    save_interval = 5
    x = torch.randn(8, 1024, device=device)

    # 对比同步和异步保存的训练阻塞时间
    sync_block_time = 0.0
    async_block_time = 0.0

    for step in range(total_steps):
        # 训练一步
        out = model(x)
        loss = out.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 保存 checkpoint
        if (step + 1) % save_interval == 0:
            # --- 异步保存 ---
            t0 = time.perf_counter()
            # 只需 CPU copy (快速)，后台线程做 I/O
            state = snapshot_state_dict(model, optimizer, step)
            checkpointer.save_async(state, step, rank)
            async_block_time += time.perf_counter() - t0

            # --- 同步保存 (对比) ---
            t0 = time.perf_counter()
            sync_path = os.path.join(save_dir, f"sync_step_{step}_rank_{rank}.pt")
            torch.save(snapshot_state_dict(model, optimizer, step), sync_path)
            sync_block_time += time.perf_counter() - t0

            if rank == 0:
                print(f"  Step {step+1}: 异步保存发起 (不阻塞训练)")

    # 等待所有异步保存完成
    checkpointer.wait_all()
    dist.barrier()

    if rank == 0:
        print(f"\n=== 结果 ===")
        print(f"  同步保存阻塞训练时间: {sync_block_time*1000:.1f} ms")
        print(f"  异步保存阻塞训练时间: {async_block_time*1000:.1f} ms (只有 CPU copy)")
        print(f"  训练阻塞减少: {(1-async_block_time/sync_block_time)*100:.0f}%")
        print(f"\n  关键: 异步保存只阻塞 CPU copy 时间 (~几 ms)")
        print(f"  磁盘 I/O (~几百 ms) 在后台完成，训练继续")

    checkpointer.shutdown()
    dist.barrier()
    if rank == 0:
        shutil.rmtree(save_dir, ignore_errors=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
