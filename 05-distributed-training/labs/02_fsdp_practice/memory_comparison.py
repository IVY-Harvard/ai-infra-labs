"""
Lab 02 - DDP vs FSDP 显存对比
==============================
在相同模型上对比 DDP、FSDP SHARD_GRAD_OP、FSDP FULL_SHARD 的显存占用。

运行方式:
    torchrun --nproc_per_node=8 memory_comparison.py

输出: 一张显存对比表格
"""

import os
import functools
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


# ---------------------------------------------------------------------------
# 模型（与 fsdp_train.py 相同）
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=16, dropout=0.1):
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
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size=32000, hidden_size=1024, num_layers=24,
                 num_heads=16, max_seq_len=1024):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, input_ids):
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(S, device=input_ids.device)
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.ln_f(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# 显存测量
# ---------------------------------------------------------------------------

def measure_memory(model, device, batch_size=4, seq_len=512, vocab_size=32000):
    """运行一步训练，返回峰值显存 (GB)"""
    torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Forward + Backward + Step
    logits = model(input_ids)
    loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize(device)
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
    return peak_mem


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    results = {}
    batch_size = 4
    seq_len = 512

    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )
    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # --- DDP ---
    torch.cuda.reset_peak_memory_stats(device)
    model_ddp = TransformerLM().to(device)
    model_ddp = DDP(model_ddp, device_ids=[local_rank])
    results["DDP"] = measure_memory(model_ddp, device, batch_size, seq_len)
    del model_ddp
    torch.cuda.empty_cache()

    # --- FSDP NO_SHARD (应该接近 DDP) ---
    torch.cuda.reset_peak_memory_stats(device)
    model_ns = TransformerLM()
    model_ns = FSDP(model_ns, sharding_strategy=ShardingStrategy.NO_SHARD,
                    mixed_precision=mp, auto_wrap_policy=wrap_policy, device_id=device)
    results["FSDP NO_SHARD"] = measure_memory(model_ns, device, batch_size, seq_len)
    del model_ns
    torch.cuda.empty_cache()

    # --- FSDP SHARD_GRAD_OP (ZeRO-2) ---
    torch.cuda.reset_peak_memory_stats(device)
    model_go = TransformerLM()
    model_go = FSDP(model_go, sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
                    mixed_precision=mp, auto_wrap_policy=wrap_policy, device_id=device)
    results["FSDP SHARD_GRAD_OP"] = measure_memory(model_go, device, batch_size, seq_len)
    del model_go
    torch.cuda.empty_cache()

    # --- FSDP FULL_SHARD (ZeRO-3) ---
    torch.cuda.reset_peak_memory_stats(device)
    model_fs = TransformerLM()
    model_fs = FSDP(model_fs, sharding_strategy=ShardingStrategy.FULL_SHARD,
                    mixed_precision=mp, auto_wrap_policy=wrap_policy, device_id=device)
    results["FSDP FULL_SHARD"] = measure_memory(model_fs, device, batch_size, seq_len)
    del model_fs
    torch.cuda.empty_cache()

    # --- 打印结果 ---
    if rank == 0:
        print("\n" + "=" * 60)
        print("显存对比 (350M 模型, 8×H20)")
        print("=" * 60)
        print(f"{'策略':<25} {'峰值显存 (GB)':<15} {'相比 DDP':<10}")
        print("-" * 60)
        ddp_mem = results["DDP"]
        for name, mem in results.items():
            ratio = mem / ddp_mem * 100
            print(f"{name:<25} {mem:<15.2f} {ratio:.0f}%")
        print("-" * 60)
        print(f"\n理论预期:")
        print(f"  DDP:            ~100% (完整模型+梯度+优化器)")
        print(f"  SHARD_GRAD_OP:  ~60-70% (梯度+优化器切分)")
        print(f"  FULL_SHARD:     ~25-35% (全切分，峰值含 AllGather 的完整参数)")
        print(f"\n注: BF16 混合精度下优化器状态占主要显存")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
