"""
Lab 02 - FSDP 训练实践
======================
使用 PyTorch FSDP 训练 Transformer 模型，支持多种切分策略。

运行方式:
    torchrun --nproc_per_node=8 fsdp_train.py --sharding full
    torchrun --nproc_per_node=8 fsdp_train.py --sharding grad_op
    torchrun --nproc_per_node=8 fsdp_train.py --sharding no_shard

关键概念:
    - FULL_SHARD (ZeRO-3): 参数+梯度+优化器全切分
    - SHARD_GRAD_OP (ZeRO-2): 梯度+优化器切分
    - NO_SHARD: 等效 DDP
"""

import argparse
import os
import time
import functools

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)


# ---------------------------------------------------------------------------
# 1. 模型 — 更大一点以凸显 FSDP 优势（~350M 参数）
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """单个 Transformer 块，作为 FSDP wrap 的基本单元"""

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        # Pre-LN Transformer
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """
    Transformer 语言模型
    hidden=1024, layers=24, heads=16 → ~350M params
    """

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=1024,
        num_layers=24,
        num_heads=16,
        max_seq_len=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, input_ids):
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)
        mask = nn.Transformer.generate_square_subsequent_mask(S, device=input_ids.device)
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.ln_f(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# 2. 数据
# ---------------------------------------------------------------------------

class SyntheticDataset(Dataset):
    def __init__(self, num_samples=8000, seq_len=1024, vocab_size=32000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        tokens = torch.randint(0, self.vocab_size, (self.seq_len,))
        return tokens[:-1], tokens[1:]


# ---------------------------------------------------------------------------
# 3. FSDP 配置
# ---------------------------------------------------------------------------

def get_sharding_strategy(name: str) -> ShardingStrategy:
    strategies = {
        "full": ShardingStrategy.FULL_SHARD,        # ZeRO-3
        "grad_op": ShardingStrategy.SHARD_GRAD_OP,  # ZeRO-2
        "no_shard": ShardingStrategy.NO_SHARD,      # DDP
        "hybrid": ShardingStrategy.HYBRID_SHARD,    # 节点内 FULL，节点间 DDP
    }
    return strategies[name]


def get_mixed_precision():
    """BF16 混合精度配置"""
    return MixedPrecision(
        param_dtype=torch.bfloat16,    # 参数用 BF16
        reduce_dtype=torch.bfloat16,   # 通信用 BF16
        buffer_dtype=torch.bfloat16,
    )


# ---------------------------------------------------------------------------
# 4. 训练
# ---------------------------------------------------------------------------

def train(args):
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"[FSDP] 启动 {world_size} 进程, sharding={args.sharding}")

    # ---------- 模型 ----------
    model = TransformerLM(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.seq_len,
    )

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  模型参数: {num_params / 1e6:.1f}M")

    # ---------- FSDP 封装 ----------
    # auto_wrap_policy: 将每个 TransformerBlock 作为一个 FSDP 单元
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    model = FSDP(
        model,
        sharding_strategy=get_sharding_strategy(args.sharding),
        mixed_precision=get_mixed_precision(),
        auto_wrap_policy=wrap_policy,
        device_id=device,
        # limit_all_gathers=True,  # 限制同时进行的 AllGather 数量
    )

    # ---------- 数据 ----------
    dataset = SyntheticDataset(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=2, pin_memory=True,
    )

    # ---------- 优化器 ----------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # ---------- 训练循环 ----------
    model.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.perf_counter()

        for step, (input_ids, targets) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, args.vocab_size), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            # FSDP 中梯度裁剪需要使用 clip_grad_norm_
            model.clip_grad_norm_(max_norm=1.0)
            optimizer.step()

            batch_tokens = input_ids.numel()
            epoch_tokens += batch_tokens
            epoch_loss += loss.item() * batch_tokens

            if rank == 0 and step % 20 == 0:
                mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
                print(
                    f"  Epoch {epoch} Step {step:3d} | "
                    f"Loss {loss.item():.4f} | "
                    f"Mem {mem_gb:.2f} GB"
                )

        elapsed = time.perf_counter() - t0
        total_tokens = torch.tensor([epoch_tokens], device=device)
        dist.all_reduce(total_tokens)

        if rank == 0:
            throughput = total_tokens.item() / elapsed
            avg_loss = epoch_loss / epoch_tokens
            print(
                f"Epoch {epoch} | Loss {avg_loss:.4f} | "
                f"Throughput {throughput:.0f} tok/s | Time {elapsed:.1f}s"
            )

    # ---------- 总结 ----------
    if rank == 0:
        peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"\n=== FSDP ({args.sharding}) 总结 ===")
        print(f"  Sharding: {args.sharding}")
        print(f"  峰值显存: {peak_mem:.2f} GB")

    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sharding", choices=["full", "grad_op", "no_shard", "hybrid"],
                        default="full")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=8000)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
