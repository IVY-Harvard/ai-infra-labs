"""
Lab 10 - 分布式 Checkpoint
============================
每个 rank 保存自己的模型/优化器分片，实现并行 I/O。

优势:
  1. 并行写入: 8 GPU 同时写 → 8× I/O 吞吐
  2. 不需要收集: 省去 AllGather 的通信开销
  3. 灵活恢复: 可以用不同数量的 GPU 加载（需要 resharding）

PyTorch 原生支持: torch.distributed.checkpoint (DCP)

运行:
    torchrun --nproc_per_node=4 distributed_checkpoint.py
"""

import os
import time
import tempfile
import shutil

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# 尝试使用 PyTorch Distributed Checkpoint
try:
    import torch.distributed.checkpoint as dcp
    HAS_DCP = True
except ImportError:
    HAS_DCP = False


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
        self.head = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return self.head(x)


def save_checkpoint_manual(model, optimizer, step, save_dir, rank, world_size):
    """
    手动分布式 checkpoint: 每个 rank 保存自己的状态。
    """
    os.makedirs(save_dir, exist_ok=True)

    # 每个 rank 保存自己的文件
    ckpt = {
        "step": step,
        "model_state_dict": model.module.state_dict(),  # DDP 模型
        "optimizer_state_dict": optimizer.state_dict(),
        "rank": rank,
        "world_size": world_size,
    }

    save_path = os.path.join(save_dir, f"rank_{rank}.pt")
    torch.save(ckpt, save_path)

    # 所有 rank 完成后，rank 0 保存元信息
    dist.barrier()
    if rank == 0:
        meta = {
            "step": step,
            "world_size": world_size,
            "files": [f"rank_{r}.pt" for r in range(world_size)],
        }
        torch.save(meta, os.path.join(save_dir, "meta.pt"))


def load_checkpoint_manual(model, optimizer, save_dir, rank, world_size):
    """加载分布式 checkpoint"""
    meta_path = os.path.join(save_dir, "meta.pt")
    if not os.path.exists(meta_path):
        return 0  # 无 checkpoint

    meta = torch.load(meta_path, map_location="cpu")
    saved_world_size = meta["world_size"]

    if saved_world_size == world_size:
        # 相同规模: 直接加载
        ckpt_path = os.path.join(save_dir, f"rank_{rank}.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.module.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt["step"]
    else:
        # 不同规模: 需要 resharding (简化: 只加载 rank 0 的模型参数)
        if rank == 0:
            print(f"  警告: checkpoint 保存时 {saved_world_size} GPU，当前 {world_size} GPU")
            print(f"  使用 rank 0 的参数 + broadcast")
        ckpt = torch.load(os.path.join(save_dir, "rank_0.pt"), map_location="cpu")
        model.module.load_state_dict(ckpt["model_state_dict"])
        # broadcast 参数到所有 rank
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        return ckpt["step"]


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

    # 临时目录存放 checkpoint
    save_dir = tempfile.mkdtemp(prefix="dist_ckpt_")
    if rank == 0:
        print(f"分布式 Checkpoint 演示")
        print(f"  保存目录: {save_dir}")
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  模型参数: {num_params/1e6:.1f}M")

    # 模拟几步训练
    x = torch.randn(8, 1024, device=device)
    for step in range(5):
        out = model(x)
        loss = out.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ====== 保存 Checkpoint ======
    dist.barrier()
    t0 = time.perf_counter()
    save_checkpoint_manual(model, optimizer, step=5, save_dir=save_dir,
                           rank=rank, world_size=world_size)
    dist.barrier()
    save_time = time.perf_counter() - t0

    if rank == 0:
        # 计算 checkpoint 大小
        total_size = sum(
            os.path.getsize(os.path.join(save_dir, f))
            for f in os.listdir(save_dir)
        )
        print(f"\n  保存完成:")
        print(f"    时间: {save_time:.3f}s")
        print(f"    总大小: {total_size/1e6:.1f} MB")
        print(f"    每 rank: {total_size/world_size/1e6:.1f} MB")

    # ====== 加载 Checkpoint ======
    # 创建新模型，验证加载
    model2 = SimpleModel().to(device)
    model2 = DDP(model2, device_ids=[local_rank])
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)

    t0 = time.perf_counter()
    loaded_step = load_checkpoint_manual(model2, optimizer2, save_dir, rank, world_size)
    dist.barrier()
    load_time = time.perf_counter() - t0

    # 验证
    params_match = all(
        torch.allclose(p1, p2) for p1, p2 in
        zip(model.parameters(), model2.parameters())
    )

    if rank == 0:
        print(f"\n  加载完成:")
        print(f"    时间: {load_time:.3f}s")
        print(f"    恢复到 step: {loaded_step}")
        print(f"    参数一致: {params_match}")

    # 清理
    dist.barrier()
    if rank == 0:
        shutil.rmtree(save_dir, ignore_errors=True)
        print(f"\n  分布式 Checkpoint 演示完成！")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
