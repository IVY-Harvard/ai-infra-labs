"""PagedAttention 计算逻辑 (简化版)"""
import torch
import torch.nn.functional as F
from typing import List, Optional


def paged_attention_forward(
    query: torch.Tensor,       # [num_tokens, num_heads, head_dim]
    key_cache: torch.Tensor,   # [num_blocks, block_size, num_kv_heads, head_dim]
    value_cache: torch.Tensor, # [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables: List[List[int]],  # [num_seqs, max_num_blocks]
    context_lens: List[int],   # [num_seqs]
    scale: float,
) -> torch.Tensor:
    """
    简化版 PagedAttention

    实际 vLLM 使用高度优化的 CUDA kernel。
    这里用 PyTorch 展示核心逻辑。
    """
    num_tokens = query.shape[0]
    num_heads = query.shape[1]
    head_dim = query.shape[2]

    outputs = []

    for seq_idx in range(len(block_tables)):
        q = query[seq_idx:seq_idx+1]  # [1, num_heads, head_dim]
        ctx_len = context_lens[seq_idx]
        blocks = block_tables[seq_idx]

        # 收集 KV from blocks
        k_list = []
        v_list = []
        tokens_collected = 0

        for block_id in blocks:
            if tokens_collected >= ctx_len:
                break
            tokens_in_block = min(key_cache.shape[1], ctx_len - tokens_collected)
            k_list.append(key_cache[block_id, :tokens_in_block])
            v_list.append(value_cache[block_id, :tokens_in_block])
            tokens_collected += tokens_in_block

        k = torch.cat(k_list, dim=0)  # [ctx_len, num_kv_heads, head_dim]
        v = torch.cat(v_list, dim=0)

        # Standard attention
        scores = torch.einsum("qhd,khd->hqk", q, k) * scale
        weights = F.softmax(scores, dim=-1)
        output = torch.einsum("hqk,khd->qhd", weights, v)
        outputs.append(output)

    return torch.cat(outputs, dim=0)
