"""
分页 KV Cache 实现

基于 Block Manager 的 KV Cache，支持:
- 按需分配 Block
- 非连续物理存储 + 逻辑连续访问
- PagedAttention 计算
"""

import torch
from typing import Dict, List, Optional, Tuple
from block_manager import BlockManager


class PagedKVCache:
    """
    分页 KV Cache

    KV 数据存储在物理 Block 中，通过 Block Table 映射。
    支持按需增长，无碎片浪费。
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.dtype = dtype
        self.device = device

        # Block Manager 管理分配逻辑
        self.block_manager = BlockManager(
            num_blocks=num_blocks,
            block_size=block_size,
            num_cpu_blocks=num_blocks // 4,
        )

        # 物理 KV Cache 存储
        # Shape: [num_blocks, block_size, num_kv_heads, head_dim]
        self.k_cache = torch.zeros(
            num_layers, num_blocks, block_size, num_kv_heads, head_dim,
            dtype=dtype, device=device,
        )
        self.v_cache = torch.zeros(
            num_layers, num_blocks, block_size, num_kv_heads, head_dim,
            dtype=dtype, device=device,
        )

        # 序列到 token position 的追踪
        self.seq_lengths: Dict[int, int] = {}

        total_bytes = 2 * num_layers * num_blocks * block_size * num_kv_heads * head_dim * 2
        print(f"[PagedKVCache] Initialized:")
        print(f"  Total KV Cache pool: {total_bytes / 1024**3:.2f} GB")
        print(f"  Blocks: {num_blocks} × {block_size} tokens")
        print(f"  Layers: {num_layers}, KV Heads: {num_kv_heads}, Head Dim: {head_dim}")

    def allocate_sequence(self, seq_id: int, num_tokens: int) -> bool:
        """为新序列分配 Block"""
        success = self.block_manager.allocate(seq_id, num_tokens)
        if success:
            self.seq_lengths[seq_id] = num_tokens
        return success

    def write_prefill_kv(
        self,
        seq_id: int,
        layer_idx: int,
        k: torch.Tensor,  # [seq_len, num_kv_heads, head_dim]
        v: torch.Tensor,  # [seq_len, num_kv_heads, head_dim]
    ):
        """
        写入 Prefill 阶段的 KV Cache

        将连续的 KV tensor 写入分散的物理 Block。
        """
        block_table = self.block_manager.block_tables[seq_id]
        seq_len = k.shape[0]
        offset = 0

        for logical_block in block_table:
            block_id = logical_block.physical_block.block_id
            tokens_in_block = logical_block.num_tokens

            # 写入这个 Block
            self.k_cache[layer_idx, block_id, :tokens_in_block] = k[offset:offset + tokens_in_block]
            self.v_cache[layer_idx, block_id, :tokens_in_block] = v[offset:offset + tokens_in_block]
            offset += tokens_in_block

    def append_token_kv(
        self,
        seq_id: int,
        layer_idx: int,
        k: torch.Tensor,  # [1, num_kv_heads, head_dim]
        v: torch.Tensor,  # [1, num_kv_heads, head_dim]
    ) -> bool:
        """
        追加一个 token 的 KV (Decode 阶段)
        """
        # 确保有 slot
        if layer_idx == 0:  # 只在第一层时分配 (所有层共享 block 结构)
            if not self.block_manager.append_slot(seq_id):
                return False
            self.seq_lengths[seq_id] = self.seq_lengths.get(seq_id, 0) + 1

        # 找到写入位置
        block_table = self.block_manager.block_tables[seq_id]
        last_block = block_table[-1]
        block_id = last_block.physical_block.block_id
        pos_in_block = last_block.num_tokens - 1  # append_slot 已经 +1 了

        self.k_cache[layer_idx, block_id, pos_in_block] = k.squeeze(0)
        self.v_cache[layer_idx, block_id, pos_in_block] = v.squeeze(0)
        return True

    def read_kv(
        self,
        seq_id: int,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        读取序列的完整 KV Cache (用于 attention 计算)

        从分散的物理 Block 中收集数据，拼接成连续 tensor。

        注意: 实际 PagedAttention kernel 不需要拼接！
        它直接读取分散的 Block (通过 Block Table)。
        这里拼接只是为了演示和验证。
        """
        block_table = self.block_manager.block_tables[seq_id]
        seq_len = self.seq_lengths[seq_id]

        k_out = torch.zeros(seq_len, self.num_kv_heads, self.head_dim,
                           dtype=self.dtype, device=self.device)
        v_out = torch.zeros(seq_len, self.num_kv_heads, self.head_dim,
                           dtype=self.dtype, device=self.device)

        offset = 0
        for logical_block in block_table:
            block_id = logical_block.physical_block.block_id
            tokens_in_block = logical_block.num_tokens

            k_out[offset:offset + tokens_in_block] = self.k_cache[layer_idx, block_id, :tokens_in_block]
            v_out[offset:offset + tokens_in_block] = self.v_cache[layer_idx, block_id, :tokens_in_block]
            offset += tokens_in_block

        return k_out, v_out

    def paged_attention(
        self,
        seq_id: int,
        layer_idx: int,
        query: torch.Tensor,  # [1, num_heads, head_dim] (decode) or [seq, num_heads, head_dim] (prefill)
    ) -> torch.Tensor:
        """
        简化版 PagedAttention 计算

        实际的 vLLM PagedAttention kernel 直接在 Block 上操作，
        不需要拼接 KV。这里为了清晰展示原理而先拼接。
        """
        k, v = self.read_kv(seq_id, layer_idx)

        # Standard attention: Q × K^T / sqrt(d) → softmax → × V
        # query: [q_len, num_heads, head_dim]
        # k: [kv_len, num_kv_heads, head_dim]
        # v: [kv_len, num_kv_heads, head_dim]

        # GQA: 扩展 KV heads 到 match Q heads
        num_q_heads = query.shape[1]
        num_kv_heads = k.shape[1]
        if num_q_heads != num_kv_heads:
            repeat_factor = num_q_heads // num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # Compute attention
        scale = self.head_dim ** -0.5
        # [q_len, num_heads, head_dim] × [kv_len, num_heads, head_dim]^T
        scores = torch.einsum("qhd,khd->hqk", query, k) * scale
        weights = torch.softmax(scores, dim=-1)
        output = torch.einsum("hqk,khd->qhd", weights, v)

        return output

    def free_sequence(self, seq_id: int):
        """释放序列"""
        self.block_manager.free(seq_id)
        if seq_id in self.seq_lengths:
            del self.seq_lengths[seq_id]

    def get_stats(self) -> dict:
        """获取统计信息"""
        bm_stats = self.block_manager.get_stats()
        bm_stats["active_sequences"] = len(self.seq_lengths)
        bm_stats["total_cached_tokens"] = sum(self.seq_lengths.values())
        return bm_stats


def test_paged_kv_cache():
    """测试分页 KV Cache"""
    print("\n" + "=" * 70)
    print("  Test: Paged KV Cache")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache = PagedKVCache(
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        num_blocks=32,
        block_size=8,
        device=device,
    )

    # 模拟 Prefill
    seq_id = 0
    prompt_len = 20
    print(f"\n  [1] Prefill: seq_id={seq_id}, prompt_len={prompt_len}")

    cache.allocate_sequence(seq_id, prompt_len)

    # 生成 dummy KV
    for layer in range(cache.num_layers):
        k = torch.randn(prompt_len, cache.num_kv_heads, cache.head_dim,
                       dtype=cache.dtype, device=device)
        v = torch.randn(prompt_len, cache.num_kv_heads, cache.head_dim,
                       dtype=cache.dtype, device=device)
        cache.write_prefill_kv(seq_id, layer, k, v)

    print(f"  Block table: {cache.block_manager.get_block_table(seq_id)}")
    print(f"  Blocks allocated: {len(cache.block_manager.block_tables[seq_id])}")

    # 模拟 Decode
    print(f"\n  [2] Decode: generating 15 tokens...")
    for step in range(15):
        for layer in range(cache.num_layers):
            k_new = torch.randn(1, cache.num_kv_heads, cache.head_dim,
                               dtype=cache.dtype, device=device)
            v_new = torch.randn(1, cache.num_kv_heads, cache.head_dim,
                               dtype=cache.dtype, device=device)
            cache.append_token_kv(seq_id, layer, k_new, v_new)

    print(f"  Block table after decode: {cache.block_manager.get_block_table(seq_id)}")
    print(f"  Total tokens cached: {cache.seq_lengths[seq_id]}")

    # 验证 Attention 计算
    print(f"\n  [3] Attention computation...")
    query = torch.randn(1, cache.num_kv_heads, cache.head_dim,
                       dtype=cache.dtype, device=device)
    output = cache.paged_attention(seq_id, layer_idx=0, query=query)
    print(f"  Query shape: {query.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output norm: {output.norm().item():.4f}")

    # 统计
    stats = cache.get_stats()
    print(f"\n  Stats:")
    print(f"  GPU blocks used: {stats['used_gpu_blocks']}/{stats['total_gpu_blocks']}")
    print(f"  Memory efficiency: {stats['memory_efficiency']:.2%}")
    print(f"  Tokens cached: {stats['total_cached_tokens']}")

    cache.block_manager.print_memory_map()

    # 清理
    cache.free_sequence(seq_id)
    print(f"\n  After freeing: {cache.block_manager.num_free_gpu_blocks} blocks free")


if __name__ == "__main__":
    test_paged_kv_cache()
