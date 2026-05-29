"""
Lab 06 - DeepSpeed ZeRO Stage 3
=================================
参数 + 梯度 + 优化器全切分。等效 FSDP FULL_SHARD。

运行:
    deepspeed --num_gpus=8 zero3_train.py

训练流程:
  Forward: 每层 AllGather 参数 → 计算 → 释放参数
  Backward: 每层 AllGather 参数 → 反向 → ReduceScatter 梯度 → 释放
  Optimizer: 只更新本地分片

显存: 16Φ/N per GPU (极致节省)
通信: 3M (比 DDP 多 50%)
"""

import argparse
import torch
import torch.nn as nn
import deepspeed
from torch.utils.data import Dataset, DataLoader


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
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
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


def get_ds_config():
    """ZeRO Stage 3 配置"""
    return {
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 4,
        "optimizer": {
            "type": "Adam",
            "params": {"lr": 3e-4, "betas": [0.9, 0.999], "weight_decay": 0.01}
        },
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "reduce_bucket_size": 5e8,
            "stage3_prefetch_bucket_size": 5e8,
            # 小参数不切分（如 LayerNorm），避免通信 overhead > 显存节省
            "stage3_param_persistence_threshold": 1e6,
            # 控制同时 AllGather 的参数量
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            # 保存 checkpoint 时收集完整 FP16 权重
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
    }


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=2)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    model = TransformerLM()
    dataset = SyntheticDataset()

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=model, config=get_ds_config(), training_data=dataset
    )

    if model_engine.local_rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        world_size = model_engine.world_size
        print(f"[ZeRO Stage 3] 模型参数: {num_params/1e6:.1f}M")
        print(f"  每 GPU 显存 (稳态): {num_params*16/world_size/1e9:.2f} GB")
        print(f"  每 GPU 显存 (峰值): ~{num_params*16/world_size/1e9 + num_params*2/1e9:.2f} GB")
        print(f"  通信量: 3 × {num_params*2/1e9:.1f} GB = {num_params*6/1e9:.1f} GB per step")

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

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
        print(f"\n=== ZeRO Stage 3 完成 | 峰值显存: {peak_mem:.2f} GB ===")


if __name__ == "__main__":
    train()
