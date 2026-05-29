"""
Lab 10 - 弹性训练 (Elastic Training)
======================================
处理训练过程中 GPU 节点的动态变化:
  - 节点故障 → 自动从 checkpoint 恢复
  - 节点加入 → 重新分配数据

PyTorch Elastic (torchrun) 提供了基础设施:
  - 监控 worker 健康状态
  - Worker 失败时重启所有 worker
  - 支持 min/max worker 数量

运行:
    torchrun --nproc_per_node=4 \
             --rdzv_backend=c10d \
             --rdzv_endpoint=localhost:29500 \
             elastic_training_demo.py

模拟故障:
    在训练过程中 kill 一个进程，观察恢复行为
"""

import os
import signal
import time
import tempfile

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class SimpleModel(nn.Module):
    def __init__(self, hidden_size=512, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class SyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, hidden_size=512):
        self.num_samples = num_samples
        self.hidden_size = hidden_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.randn(self.hidden_size), torch.randn(self.hidden_size)


class ElasticTrainer:
    """
    弹性训练管理器。
    核心能力:
      1. 定期保存 checkpoint
      2. 启动时检测并加载最新 checkpoint
      3. 适配当前 world_size 进行数据分配
    """

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def get_latest_checkpoint(self):
        """找到最新的 checkpoint"""
        ckpts = [f for f in os.listdir(self.checkpoint_dir) if f.startswith("step_")]
        if not ckpts:
            return None, 0

        # 按 step 排序
        steps = [int(f.split("_")[1].split(".")[0]) for f in ckpts]
        latest_step = max(steps)
        latest_file = os.path.join(self.checkpoint_dir, f"step_{latest_step}.pt")
        return latest_file, latest_step

    def save_checkpoint(self, model, optimizer, step, rank):
        """只有 rank 0 保存（简化版，生产中可分布式保存）"""
        if rank != 0:
            return

        ckpt = {
            "step": step,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        save_path = os.path.join(self.checkpoint_dir, f"step_{step}.pt")
        torch.save(ckpt, save_path)

        # 保留最近 3 个 checkpoint
        ckpts = sorted(
            [f for f in os.listdir(self.checkpoint_dir) if f.startswith("step_")],
            key=lambda f: int(f.split("_")[1].split(".")[0]),
        )
        while len(ckpts) > 3:
            os.remove(os.path.join(self.checkpoint_dir, ckpts.pop(0)))

    def load_checkpoint(self, model, optimizer, device):
        """加载最新 checkpoint"""
        ckpt_path, step = self.get_latest_checkpoint()
        if ckpt_path is None:
            return 0

        ckpt = torch.load(ckpt_path, map_location=device)
        model.module.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt["step"]


def main():
    # 初始化（torchrun 提供环境变量）
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"弹性训练演示 | world_size={world_size}")
        print(f"  如果某个 worker 失败，torchrun 会重启所有 worker")
        print(f"  重启后从最新 checkpoint 恢复\n")

    # 模型和优化器
    model = SimpleModel().to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 数据
    dataset = SyntheticDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)

    # 弹性训练管理器
    ckpt_dir = tempfile.mkdtemp(prefix="elastic_ckpt_")
    trainer = ElasticTrainer(ckpt_dir)

    # 尝试从 checkpoint 恢复
    start_step = trainer.load_checkpoint(model, optimizer, device)
    if rank == 0:
        if start_step > 0:
            print(f"  从 step {start_step} 恢复训练")
        else:
            print(f"  从头开始训练")

    # 训练循环
    total_steps = 50
    save_interval = 10
    global_step = start_step

    for epoch in range(3):
        sampler.set_epoch(epoch)
        for batch_idx, (x, target) in enumerate(dataloader):
            if global_step >= total_steps:
                break

            x = x.to(device)
            target = target.to(device)

            out = model(x)
            loss = (out - target).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            if rank == 0 and global_step % 10 == 0:
                print(f"  Step {global_step}/{total_steps} | Loss {loss.item():.4f}")

            # 定期保存 checkpoint
            if global_step % save_interval == 0:
                trainer.save_checkpoint(model, optimizer, global_step, rank)
                dist.barrier()
                if rank == 0:
                    print(f"    Checkpoint saved at step {global_step}")

    if rank == 0:
        print(f"\n训练完成！最终 step: {global_step}")
        print(f"\n弹性训练要点:")
        print(f"  1. 定期保存 checkpoint (每 N 步)")
        print(f"  2. 启动时检查并恢复最新 checkpoint")
        print(f"  3. 使用 DistributedSampler 适配变化的 world_size")
        print(f"  4. torchrun 自动处理 worker 故障和重启")
        print(f"  5. 使用 --rdzv_backend=c10d 进行服务发现")

    dist.destroy_process_group()

    # 清理
    import shutil
    if rank == 0:
        shutil.rmtree(ckpt_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
