"""
Lab 01 - 单卡训练基线
=====================
用一个简化的 Transformer 模型演示单 GPU 训练循环。
记录吞吐量、显存占用和 loss 曲线，作为后续 DDP 对比的基线。

运行方式:
    python single_gpu.py --epochs 3 --batch-size 32

目标环境: 单张 H20 GPU (96 GB HBM3)
"""

import argparse
import time
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# 1. 模型定义 — 简化 Transformer（~125M 参数，方便快速实验）
# ---------------------------------------------------------------------------

class SimpleTransformer(nn.Module):
    """
    一个小型 Transformer 语言模型：
    - vocab_size=32000, hidden=768, layers=12, heads=12
    - 参数量 ≈ 125M（BF16 ≈ 250 MB）
    """

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

        # Token + Position Embedding
        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        self.drop = nn.Dropout(dropout)

        # Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN，更稳定
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # LM Head
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # 权重共享 (Weight Tying)
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        """参数初始化（GPT-2 风格）"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)

        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        x = self.drop(x)

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(S, device=input_ids.device)
        x = self.encoder(x, mask=mask, is_causal=True)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


# ---------------------------------------------------------------------------
# 2. 合成数据集
# ---------------------------------------------------------------------------

class SyntheticTextDataset(Dataset):
    """
    合成随机 token 序列，模拟语言建模任务。
    生产中替换为真实数据即可，训练循环不需要改动。
    """

    def __init__(self, num_samples: int = 10000, seq_len: int = 512, vocab_size: int = 32000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 随机 token 序列
        tokens = torch.randint(0, self.vocab_size, (self.seq_len,))
        # 语言建模：input = tokens[:-1], target = tokens[1:]
        return tokens[:-1], tokens[1:]


# ---------------------------------------------------------------------------
# 3. 训练循环
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda:0")

    # ---------- 模型 ----------
    model = SimpleTransformer(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.seq_len,
    ).to(device)

    # 统计参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[单卡训练] 模型参数量: {num_params / 1e6:.1f}M")

    # ---------- 数据 ----------
    dataset = SyntheticTextDataset(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # ---------- 优化器 ----------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # ---------- 训练 ----------
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.perf_counter()

        for step, (input_ids, targets) in enumerate(dataloader):
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, args.vocab_size), targets.view(-1))

            # Backward
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪（生产必备）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_tokens = input_ids.numel()
            epoch_tokens += batch_tokens
            epoch_loss += loss.item() * batch_tokens

            if step % 50 == 0:
                mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
                print(
                    f"  Epoch {epoch} Step {step:4d} | "
                    f"Loss {loss.item():.4f} | "
                    f"Memory {mem_gb:.2f} GB"
                )

        elapsed = time.perf_counter() - t0
        avg_loss = epoch_loss / epoch_tokens
        throughput = epoch_tokens / elapsed

        print(
            f"Epoch {epoch} 完成 | "
            f"Avg Loss {avg_loss:.4f} | "
            f"Throughput {throughput:.0f} tokens/sec | "
            f"Time {elapsed:.1f}s"
        )

    # ---------- 显存统计 ----------
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"\n=== 单卡训练总结 ===")
    print(f"  模型参数: {num_params / 1e6:.1f}M")
    print(f"  峰值显存: {peak_mem:.2f} GB")
    print(f"  最终 loss: {avg_loss:.4f}")


# ---------------------------------------------------------------------------
# 4. 入口
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="单卡训练基线")
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
