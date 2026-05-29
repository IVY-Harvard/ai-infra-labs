"""
Lab 09 - BF16 混合精度训练
===========================
BF16 训练比 FP16 简单: 不需要 Loss Scaling！

BF16 的指数位与 FP32 相同 (8 bit)，范围足够大，不会梯度下溢。
精度低于 FP16 (7 vs 10 mantissa bits)，但对训练影响很小。

H20 原生支持 BF16，是推荐的训练精度。

运行:
    torchrun --nproc_per_node=4 bf16_training.py
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
from torch.cuda.amp import autocast


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=32000, hidden_size=768, num_layers=12,
                 num_heads=12, max_seq_len=512):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, input_ids):
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(S, device=input_ids.device)
        x = self.encoder(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        return self.lm_head(x)


class SyntheticDataset(Dataset):
    def __init__(self, num_samples=8000, seq_len=512, vocab_size=32000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        tokens = torch.randint(0, self.vocab_size, (self.seq_len,))
        return tokens[:-1], tokens[1:]


def train(args):
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    model = SimpleTransformer().to(device)
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # BF16: 不需要 GradScaler！
    # 只需要 autocast(dtype=torch.bfloat16)

    dataset = SyntheticDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    model.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        t0 = time.perf_counter()

        for step, (input_ids, targets) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # ====== BF16 autocast: 简单干净 ======
            with autocast(dtype=torch.bfloat16):
                logits = model(input_ids)
                loss = loss_fn(logits.view(-1, 32000), targets.view(-1))

            # 直接 backward，不需要 scaler
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if rank == 0 and step % 50 == 0:
                mem = torch.cuda.max_memory_allocated(device) / 1e9
                print(f"  Step {step:4d} | Loss {loss.item():.4f} | Mem {mem:.2f} GB")

        if rank == 0:
            elapsed = time.perf_counter() - t0
            print(f"Epoch {epoch} | Time {elapsed:.1f}s")

    if rank == 0:
        print(f"\n=== BF16 训练完成 | 峰值显存: {torch.cuda.max_memory_allocated(device)/1e9:.2f} GB ===")
        print(f"  BF16 优势: 不需要 Loss Scaling，代码更简洁，训练更稳定")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    train(parser.parse_args())
