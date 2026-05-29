"""
Lab 06 - DeepSpeed ZeRO Stage 1
=================================
优化器状态切分：每 GPU 只存 1/N 的 optimizer state。
参数和梯度仍然完整。

运行:
    deepspeed --num_gpus=8 zero1_train.py

显存节省原理:
  DDP: 2Φ + 2Φ + 12Φ = 16Φ per GPU
  Stage 1: 2Φ + 2Φ + 12Φ/N = 4Φ + 12Φ/N per GPU
  8 GPU: 4×14 + 12×14/8 = 56 + 21 = 77 GB (vs DDP 112 GB)
"""

import argparse
import torch
import torch.nn as nn
import deepspeed
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

class TransformerLM(nn.Module):
    def __init__(self, vocab_size=32000, hidden_size=1024, num_layers=24,
                 num_heads=16, max_seq_len=1024):
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

    def forward(self, input_ids, labels=None):
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(S, device=input_ids.device)
        x = self.encoder(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        return loss, logits


class SyntheticDataset(Dataset):
    def __init__(self, num_samples=5000, seq_len=512, vocab_size=32000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        tokens = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {"input_ids": tokens[:-1], "labels": tokens[1:]}


# ---------------------------------------------------------------------------
# DeepSpeed 配置
# ---------------------------------------------------------------------------

def get_ds_config():
    """ZeRO Stage 1 配置"""
    return {
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 4,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 3e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
            }
        },
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 1,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
        "wall_clock_breakdown": True,
    }


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=2)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    model = TransformerLM()
    dataset = SyntheticDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # DeepSpeed 初始化
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        config=get_ds_config(),
        training_data=dataset,
    )

    device = model_engine.local_rank

    if model_engine.local_rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"[ZeRO Stage 1] 模型参数: {num_params/1e6:.1f}M")
        print(f"  预期显存: 参数 {num_params*2/1e9:.1f}GB + 梯度 {num_params*2/1e9:.1f}GB + "
              f"优化器 {num_params*12/8/1e9:.1f}GB")

    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(model_engine.device)
            labels = batch["labels"].to(model_engine.device)

            loss, _ = model_engine(input_ids, labels=labels)
            model_engine.backward(loss)
            model_engine.step()

            if step % 20 == 0 and model_engine.local_rank == 0:
                mem_gb = torch.cuda.max_memory_allocated() / 1e9
                print(f"  Epoch {epoch} Step {step} | Loss {loss.item():.4f} | Mem {mem_gb:.2f} GB")

    if model_engine.local_rank == 0:
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n=== ZeRO Stage 1 完成 ===")
        print(f"  峰值显存: {peak_mem:.2f} GB")


if __name__ == "__main__":
    train()
