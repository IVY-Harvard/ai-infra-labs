"""
朴素 KV Cache 管理实现

演示传统的连续内存分配方案及其问题:
- 预分配 max_seq_len 的空间
- 内部碎片 (浪费预分配但未使用的空间)
- 外部碎片 (空闲空间不连续)

通过对比，理解为什么需要 PagedAttention。
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import time


@dataclass
class NaiveKVCache:
    """
    朴素 KV Cache 管理器

    为每个序列预分配 max_seq_len 的连续显存空间。
    这是 PagedAttention 之前的传统方案。
    """
    num_layers: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    max_batch_size: int
    dtype: torch.dtype = torch.float16
    device: str = "cuda"

    # 状态
    allocated_sequences: Dict[int, dict] = field(default_factory=dict)

    def __post_init__(self):
        """预分配整块 KV Cache 显存"""
        # 预分配: [max_batch, max_seq_len, num_kv_heads, head_dim] per layer
        self.k_cache = []
        self.v_cache = []

        for _ in range(self.num_layers):
            k = torch.zeros(
                self.max_batch_size, self.max_seq_len, self.num_kv_heads, self.head_dim,
                dtype=self.dtype, device=self.device
            )
            v = torch.zeros(
                self.max_batch_size, self.max_seq_len, self.num_kv_heads, self.head_dim,
                dtype=self.dtype, device=self.device
            )
            self.k_cache.append(k)
            self.v_cache.append(v)

        # 跟踪哪些 slot 被占用
        self.slot_used = [False] * self.max_batch_size

        # 计算总分配量
        per_element = 2 if self.dtype == torch.float16 else 4
        self.total_allocated_bytes = (
            2 * self.num_layers * self.max_batch_size * self.max_seq_len
            * self.num_kv_heads * self.head_dim * per_element
        )

        print(f"[NaiveKVCache] Pre-allocated {self.total_allocated_bytes / 1024**3:.2f} GB")
        print(f"  Config: {self.num_layers} layers, {self.num_kv_heads} kv_heads, "
              f"head_dim={self.head_dim}")
        print(f"  Capacity: {self.max_batch_size} sequences × {self.max_seq_len} tokens")

    def allocate(self, seq_id: int) -> Optional[int]:
        """为新序列分配一个 slot (预分配 max_seq_len 空间)"""
        # 找一个空闲 slot
        for i, used in enumerate(self.slot_used):
            if not used:
                self.slot_used[i] = True
                self.allocated_sequences[seq_id] = {
                    "slot": i,
                    "current_len": 0,
                    "allocated_len": self.max_seq_len,  # 始终分配 max_seq_len!
                }
                return i
        return None  # 没有空闲 slot

    def append_token(self, seq_id: int, layer_idx: int,
                     k: torch.Tensor, v: torch.Tensor) -> bool:
        """向序列追加一个 token 的 KV"""
        if seq_id not in self.allocated_sequences:
            return False

        info = self.allocated_sequences[seq_id]
        slot = info["slot"]
        pos = info["current_len"]

        if pos >= self.max_seq_len:
            return False  # 超出预分配长度

        # 写入 KV Cache
        self.k_cache[layer_idx][slot, pos] = k
        self.v_cache[layer_idx][slot, pos] = v

        if layer_idx == self.num_layers - 1:
            info["current_len"] = pos + 1

        return True

    def get_kv(self, seq_id: int, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """获取序列的 KV Cache (用于 attention 计算)"""
        if seq_id not in self.allocated_sequences:
            return None

        info = self.allocated_sequences[seq_id]
        slot = info["slot"]
        current_len = info["current_len"]

        k = self.k_cache[layer_idx][slot, :current_len]
        v = self.v_cache[layer_idx][slot, :current_len]
        return k, v

    def free(self, seq_id: int):
        """释放序列的 KV Cache"""
        if seq_id in self.allocated_sequences:
            slot = self.allocated_sequences[seq_id]["slot"]
            self.slot_used[slot] = False
            # 清零 (可选, 但有助于观察)
            for layer_idx in range(self.num_layers):
                self.k_cache[layer_idx][slot].zero_()
                self.v_cache[layer_idx][slot].zero_()
            del self.allocated_sequences[seq_id]

    def get_utilization(self) -> dict:
        """计算显存利用率"""
        total_slots = self.max_batch_size
        used_slots = sum(self.slot_used)

        # 实际使用的 token 数
        actual_tokens = sum(
            info["current_len"] for info in self.allocated_sequences.values()
        )
        # 已分配的 token 数 (预分配)
        allocated_tokens = used_slots * self.max_seq_len
        # 总容量
        total_tokens = total_slots * self.max_seq_len

        return {
            "slot_utilization": used_slots / total_slots,
            "token_utilization": actual_tokens / allocated_tokens if allocated_tokens > 0 else 0,
            "memory_efficiency": actual_tokens / total_tokens if total_tokens > 0 else 0,
            "internal_fragmentation": 1 - (actual_tokens / allocated_tokens) if allocated_tokens > 0 else 0,
            "used_slots": used_slots,
            "total_slots": total_slots,
            "actual_tokens": actual_tokens,
            "allocated_tokens": allocated_tokens,
            "wasted_tokens": allocated_tokens - actual_tokens,
        }


def demonstrate_fragmentation():
    """演示朴素 KV Cache 的碎片问题"""

    print("\n" + "=" * 70)
    print("  Demonstration: Naive KV Cache Fragmentation")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建一个小规模的 KV Cache (用于演示)
    cache = NaiveKVCache(
        num_layers=4,
        num_kv_heads=8,
        head_dim=64,
        max_seq_len=1024,  # 预分配 1024 tokens
        max_batch_size=8,
        device=device,
    )

    # 模拟多个请求，实际序列长度差异很大
    requests = [
        (0, 100),   # 请求 0: 实际只用 100 tokens
        (1, 50),    # 请求 1: 实际只用 50 tokens
        (2, 800),   # 请求 2: 实际用 800 tokens
        (3, 30),    # 请求 3: 实际只用 30 tokens
        (4, 200),   # 请求 4: 实际用 200 tokens
    ]

    print(f"\n  Scenario: 5 requests with varying actual lengths")
    print(f"  Max sequence length (pre-allocated): {cache.max_seq_len}")
    print(f"\n  {'Seq ID':<8} {'Actual Len':<12} {'Allocated':<12} {'Waste %':<10}")
    print(f"  {'-'*42}")

    for seq_id, actual_len in requests:
        cache.allocate(seq_id)
        # 模拟写入 actual_len 个 token
        dummy_kv = torch.randn(cache.num_kv_heads, cache.head_dim,
                              dtype=cache.dtype, device=device)
        for pos in range(actual_len):
            for layer in range(cache.num_layers):
                cache.append_token(seq_id, layer, dummy_kv, dummy_kv)

        waste = (cache.max_seq_len - actual_len) / cache.max_seq_len * 100
        print(f"  {seq_id:<8} {actual_len:<12} {cache.max_seq_len:<12} {waste:.1f}%")

    # 打印利用率
    util = cache.get_utilization()
    print(f"\n  Overall Statistics:")
    print(f"  {'─'*50}")
    print(f"  Slot Utilization:        {util['slot_utilization']*100:.1f}% ({util['used_slots']}/{util['total_slots']})")
    print(f"  Token Utilization:       {util['token_utilization']*100:.1f}%")
    print(f"  Memory Efficiency:       {util['memory_efficiency']*100:.1f}%")
    print(f"  Internal Fragmentation:  {util['internal_fragmentation']*100:.1f}%")
    print(f"  Wasted Tokens:           {util['wasted_tokens']:,} / {util['allocated_tokens']:,}")

    # 演示外部碎片
    print(f"\n  --- External Fragmentation Demo ---")
    print(f"\n  Freeing requests 1 and 3 (non-adjacent slots)...")
    cache.free(1)
    cache.free(3)

    util_after = cache.get_utilization()
    print(f"  After freeing: {util_after['used_slots']}/{util_after['total_slots']} slots used")
    print(f"  Free slots: {util_after['total_slots'] - util_after['used_slots']}")
    print(f"  But free slots might not be contiguous! (external fragmentation)")
    print(f"  → PagedAttention solves this by allowing non-contiguous allocation")

    # 清理
    for seq_id, _ in requests:
        if seq_id in cache.allocated_sequences:
            cache.free(seq_id)

    return cache


def compare_with_ideal():
    """对比朴素方案和理想方案的显存效率"""

    print("\n" + "=" * 70)
    print("  Comparison: Naive vs Ideal (PagedAttention-like) Memory Usage")
    print("=" * 70)

    # 模拟真实场景: 100 个请求, 实际长度服从某分布
    import random
    random.seed(42)

    max_seq_len = 4096
    num_requests = 100

    # 实际长度: 大部分短, 少数长 (typical production distribution)
    actual_lengths = [min(int(random.expovariate(1/500)), max_seq_len)
                     for _ in range(num_requests)]
    actual_lengths = [max(l, 10) for l in actual_lengths]  # 最少 10 tokens

    avg_len = sum(actual_lengths) / len(actual_lengths)
    max_actual = max(actual_lengths)

    # 朴素方案: 每个请求预分配 max_seq_len
    naive_allocated = num_requests * max_seq_len
    naive_used = sum(actual_lengths)
    naive_efficiency = naive_used / naive_allocated

    # 理想方案 (PagedAttention): 只分配实际使用量 + 小量碎片
    block_size = 16
    paged_allocated = sum((l + block_size - 1) // block_size * block_size
                         for l in actual_lengths)
    paged_efficiency = naive_used / paged_allocated

    print(f"\n  Workload: {num_requests} requests")
    print(f"  Max sequence length (config): {max_seq_len}")
    print(f"  Actual lengths: avg={avg_len:.0f}, max={max_actual}, min={min(actual_lengths)}")

    print(f"\n  {'Scheme':<20} {'Allocated (tokens)':<22} {'Efficiency':<12} {'Waste':<15}")
    print(f"  {'-'*69}")
    print(f"  {'Naive (pre-alloc)':<20} {naive_allocated:<22,} {naive_efficiency*100:.1f}%       "
          f"{(1-naive_efficiency)*100:.1f}%")
    print(f"  {'PagedAttention':<20} {paged_allocated:<22,} {paged_efficiency*100:.1f}%       "
          f"{(1-paged_efficiency)*100:.1f}%")

    # 转换为 GB (假设 LLaMA-70B per-token KV = 320KB)
    per_token_bytes = 320 * 1024  # 320 KB per token for 70B GQA model (approximate)
    naive_gb = naive_allocated * per_token_bytes / (1024**3)
    paged_gb = paged_allocated * per_token_bytes / (1024**3)
    actual_gb = naive_used * per_token_bytes / (1024**3)

    print(f"\n  In GB (assuming LLaMA-70B-like model, ~320KB/token KV):")
    print(f"  {'Naive':<20} {naive_gb:.1f} GB allocated, {actual_gb:.1f} GB actually used")
    print(f"  {'PagedAttention':<20} {paged_gb:.1f} GB allocated, {actual_gb:.1f} GB actually used")
    print(f"\n  Savings: {naive_gb - paged_gb:.1f} GB freed by PagedAttention")
    print(f"  → Can serve {(naive_gb - paged_gb) / (avg_len * per_token_bytes / 1024**3):.0f} more requests!")


if __name__ == "__main__":
    demonstrate_fragmentation()
    compare_with_ideal()

    print("\n" + "=" * 70)
    print("  Conclusion:")
    print("  ─────────────────────────────────────────────")
    print("  Naive KV Cache pre-allocation wastes 60-90% of memory.")
    print("  PagedAttention reduces waste to <4% by:")
    print("    1. Allocating blocks on-demand (no pre-allocation)")
    print("    2. Using fixed-size blocks (no external fragmentation)")
    print("    3. Block table mapping (non-contiguous is OK)")
    print("=" * 70)
