"""
Lab 01 - DDP 多卡训练
=====================
在 single_gpu.py 基础上，最少改动迁移到 DDP。
标注了每处 DDP 特有代码，方便与单卡版本 diff 对比。

运行方式:
    torchrun --nproc_per_node=8 ddp_train.py --epochs 3 --batch-size 32

关键改动点（搜索 "# DDP"）:
    1. dist.init_process_group
    2. DistributedSampler
    3. DDP(model)
    4. 只在 rank 0 打印日志
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


# ---------------------------------------------------------------------------
# 1. 模型（与 single_gpu.py 完全相同）
# ---------------------------------------------------------------------------

class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        self.drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        x = self.drop(x)
        mask = nn.Transformer.generate_square_subsequent_mask(S, device=input_ids.device)
        x = self.encoder(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# 2. 数据集（与 single_gpu.py 完全相同）
# ---------------------------------------------------------------------------

class SyntheticTextDataset(Dataset):
    def __init__(self, num_samples=10000, seq_len=512, vocab_size=32000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        tokens = torch.randint(0, self.vocab_size, (self.seq_len,))
        return tokens[:-1], tokens[1:]


# ---------------------------------------------------------------------------
# 3. DDP 训练循环
# ---------------------------------------------------------------------------

def train(args):
    # ====== DDP 改动 1: 初始化进程组 ======
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"[DDP] 启动 {world_size} 个进程")

    # ---------- 模型 ----------
    model = SimpleTransformer(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.seq_len,
    ).to(device)

    # ====== DDP 改动 2: 用 DDP 包裹模型 ======
    # DDP 构造时会 broadcast rank 0 的参数到所有 rank
    # bucket_cap_mb 控制梯度桶大小，默认 25MB
    model = DDP(model, device_ids=[local_rank], bucket_cap_mb=25)

    num_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"  模型参数: {num_params / 1e6:.1f}M")

    # ---------- 数据 ----------
    dataset = SyntheticTextDataset(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
    )

    # ====== DDP 改动 3: 使用 DistributedSampler ======
    # 保证每个 rank 拿到不同的数据子集
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,  # 这是 per-GPU batch size
        sampler=sampler,             # 不能再用 shuffle=True
        num_workers=2,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # ---------- 训练 ----------
    model.train()
    for epoch in range(args.epochs):
        # ====== DDP 改动 4: 每个 epoch 设置 sampler 的 epoch ======
        # 确保每个 epoch 的 shuffle 不同
        sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.perf_counter()

        for step, (input_ids, targets) in enumerate(dataloader):
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, args.vocab_size), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            # DDP 的 AllReduce 在 backward() 内部自动触发
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_tokens = input_ids.numel()
            epoch_tokens += batch_tokens
            epoch_loss += loss.item() * batch_tokens

            # ====== DDP 改动 5: 只在 rank 0 打印 ======
            if rank == 0 and step % 50 == 0:
                mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
                print(
                    f"  Epoch {epoch} Step {step:4d} | "
                    f"Loss {loss.item():.4f} | "
                    f"Memory {mem_gb:.2f} GB"
                )

        elapsed = time.perf_counter() - t0

        # 汇总所有 rank 的吞吐量
        total_tokens_tensor = torch.tensor([epoch_tokens], device=device)
        dist.all_reduce(total_tokens_tensor, op=dist.ReduceOp.SUM)
        global_tokens = total_tokens_tensor.item()
        global_throughput = global_tokens / elapsed

        if rank == 0:
            avg_loss = epoch_loss / epoch_tokens
            print(
                f"Epoch {epoch} 完成 | "
                f"Loss {avg_loss:.4f} | "
                f"Global Throughput {global_throughput:.0f} tokens/sec | "
                f"Time {elapsed:.1f}s | "
                f"GPUs {world_size}"
            )

    # ---------- 总结 ----------
    if rank == 0:
        peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"\n=== DDP {world_size} GPU 总结 ===")
        print(f"  模型参数: {num_params / 1e6:.1f}M")
        print(f"  峰值显存/卡: {peak_mem:.2f} GB")
        print(f"  全局 batch size: {args.batch_size * world_size}")

    # ====== DDP 改动 6: 清理进程组 ======
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="DDP 多卡训练")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--num-samples", type=int, default=10000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
