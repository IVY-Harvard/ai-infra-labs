"""KV Cache 引擎 — 管理 KV Cache 的物理存储"""
import torch
from typing import List, Tuple, Optional


class CacheEngine:
    """KV Cache 物理存储管理"""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        block_size: int,
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.block_size = block_size

        device = "cuda" if torch.cuda.is_available() else "cpu"
        storage_dtype = dtype if device == "cuda" else torch.float32

        # GPU KV Cache: [num_blocks, block_size, num_kv_heads, head_dim]
        self.gpu_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(num_layers):
            k = torch.zeros(num_gpu_blocks, block_size, num_kv_heads, head_dim,
                          dtype=storage_dtype, device=device)
            v = torch.zeros(num_gpu_blocks, block_size, num_kv_heads, head_dim,
                          dtype=storage_dtype, device=device)
            self.gpu_cache.append((k, v))

        # CPU KV Cache (for swap)
        self.cpu_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(num_layers):
            k = torch.zeros(num_cpu_blocks, block_size, num_kv_heads, head_dim,
                          dtype=torch.float32, device="cpu")
            v = torch.zeros(num_cpu_blocks, block_size, num_kv_heads, head_dim,
                          dtype=torch.float32, device="cpu")
            self.cpu_cache.append((k, v))

    def swap_out(self, src_gpu_blocks: List[int], dst_cpu_blocks: List[int]):
        """GPU → CPU"""
        for layer_idx in range(self.num_layers):
            gpu_k, gpu_v = self.gpu_cache[layer_idx]
            cpu_k, cpu_v = self.cpu_cache[layer_idx]
            for src, dst in zip(src_gpu_blocks, dst_cpu_blocks):
                cpu_k[dst].copy_(gpu_k[src].cpu())
                cpu_v[dst].copy_(gpu_v[src].cpu())

    def swap_in(self, src_cpu_blocks: List[int], dst_gpu_blocks: List[int]):
        """CPU → GPU"""
        device = self.gpu_cache[0][0].device
        for layer_idx in range(self.num_layers):
            gpu_k, gpu_v = self.gpu_cache[layer_idx]
            cpu_k, cpu_v = self.cpu_cache[layer_idx]
            for src, dst in zip(src_cpu_blocks, dst_gpu_blocks):
                gpu_k[dst].copy_(cpu_k[src].to(device))
                gpu_v[dst].copy_(cpu_v[src].to(device))
