"""
Lab 05: 朴素 Attention 实现

实现标准的 Scaled Dot-Product Attention，
然后分析其内存访问模式和性能瓶颈。

Usage: python naive_attention.py
"""

import torch
import torch.nn.functional as F
import time
import math


def naive_attention_manual(Q, K, V, mask=None):
    """
    手动实现的 Attention，拆解每一步。

    Q, K, V: [batch, heads, seq_len, head_dim]

    这个实现是为了学习，展示每步的内存开销。
    """
    batch, heads, seq_len, head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)

    # Step 1: 计算注意力分数 S = Q @ K^T
    # S shape: [batch, heads, seq_len, seq_len]
    # 内存开销: batch × heads × seq_len² × 4 bytes (FP32)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale

    s_memory_mb = S.nelement() * S.element_size() / 1e6
    print(f"  S 矩阵大小: {list(S.shape)}, 显存: {s_memory_mb:.1f} MB")

    # Step 2: Causal Mask（可选）
    if mask is not None:
        S = S.masked_fill(mask == 0, float('-inf'))

    # Step 3: Softmax
    # P shape: 同 S，[batch, heads, seq_len, seq_len]
    # 又一个 seq_len² 的矩阵！
    P = F.softmax(S, dim=-1)

    p_memory_mb = P.nelement() * P.element_size() / 1e6
    print(f"  P 矩阵大小: {list(P.shape)}, 显存: {p_memory_mb:.1f} MB")

    # Step 4: 加权求和 O = P @ V
    # O shape: [batch, heads, seq_len, head_dim]
    O = torch.matmul(P, V)

    return O


def pytorch_sdpa_attention(Q, K, V, is_causal=False):
    """
    PyTorch 内置的 Scaled Dot-Product Attention。
    PyTorch 2.0+ 会自动选择最优后端（FlashAttention、Memory-Efficient 等）。
    """
    return F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)


def benchmark_attention(func, Q, K, V, warmup=5, runs=20, **kwargs):
    """计时工具"""
    # Warmup
    for _ in range(warmup):
        _ = func(Q, K, V, **kwargs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        _ = func(Q, K, V, **kwargs)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / runs * 1000  # ms


def analyze_memory_scaling():
    """分析 Attention 显存随序列长度的 O(N²) 增长"""
    print("=" * 60)
    print("Attention 显存分析: S 和 P 矩阵的 O(N²) 问题")
    print("=" * 60)

    batch, heads, head_dim = 1, 32, 128

    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]

    print(f"\n{'seq_len':>8} | {'S矩阵大小':>12} | {'显存(MB)':>10} | {'显存(GB)':>10}")
    print("-" * 55)

    for seq_len in seq_lengths:
        # S 矩阵: [batch, heads, seq_len, seq_len] 的 FP32
        s_elements = batch * heads * seq_len * seq_len
        s_memory_bytes = s_elements * 4  # FP32 = 4 bytes
        s_memory_mb = s_memory_bytes / 1e6
        s_memory_gb = s_memory_bytes / 1e9

        # 注意: 实际上 S 和 P 都需要存，所以内存是 2x
        total_gb = s_memory_gb * 2  # S + P

        print(f"{seq_len:>8} | {s_elements:>12,} | {s_memory_mb:>10.1f} | {total_gb:>10.2f}")

    print(f"\n关键观察:")
    print(f"  - seq_len=2048:  S+P 约 1 GB —— 可以接受")
    print(f"  - seq_len=8192:  S+P 约 16 GB —— 单卡放不下")
    print(f"  - seq_len=32768: S+P 约 256 GB —— 完全不可行")
    print(f"  - 这就是为什么需要 FlashAttention: 不存储完整的 N×N 矩阵")


def performance_comparison():
    """性能对比: 朴素 vs PyTorch SDPA"""
    print("\n" + "=" * 60)
    print("性能对比: 朴素 Attention vs PyTorch SDPA")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA 不可用，跳过 GPU 测试")
        return

    device = torch.device("cuda:0")
    batch, heads, head_dim = 4, 32, 128

    seq_lengths = [512, 1024, 2048, 4096]

    print(f"\nbatch={batch}, heads={heads}, head_dim={head_dim}")
    print(f"{'seq_len':>8} | {'朴素(ms)':>10} | {'SDPA(ms)':>10} | {'加速比':>8}")
    print("-" * 50)

    for seq_len in seq_lengths:
        Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
        K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
        V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)

        # 朴素实现
        try:
            naive_ms = benchmark_attention(
                lambda q, k, v: torch.matmul(F.softmax(torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(head_dim), dim=-1), v),
                Q, K, V
            )
        except torch.cuda.OutOfMemoryError:
            naive_ms = float('inf')

        # PyTorch SDPA（会自动选择 FlashAttention 等后端）
        sdpa_ms = benchmark_attention(pytorch_sdpa_attention, Q, K, V)

        speedup = naive_ms / sdpa_ms if naive_ms != float('inf') else float('inf')

        naive_str = f"{naive_ms:.2f}" if naive_ms != float('inf') else "OOM"
        print(f"{seq_len:>8} | {naive_str:>10} | {sdpa_ms:>10.2f} | {speedup:>7.1f}x")

        del Q, K, V
        torch.cuda.empty_cache()

    print(f"\n关键观察:")
    print(f"  - PyTorch SDPA 自动使用 FlashAttention（如可用）")
    print(f"  - 长序列加速更明显，因为 O(N²) 的 HBM 访问被消除")
    print(f"  - 朴素实现在长序列上可能 OOM，而 FlashAttention 不会")


def main():
    print("Lab 05: 朴素 Attention 实现与分析\n")

    # Part 1: 显示手动实现的每步开销
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("(在 CPU 上运行，性能数据仅供参考)\n")

    batch, heads, seq_len, head_dim = 2, 8, 1024, 64
    Q = torch.randn(batch, heads, seq_len, head_dim, device=device)
    K = torch.randn(batch, heads, seq_len, head_dim, device=device)
    V = torch.randn(batch, heads, seq_len, head_dim, device=device)

    print(f"输入: batch={batch}, heads={heads}, seq_len={seq_len}, head_dim={head_dim}")
    print(f"Q/K/V 每个: {Q.nelement() * Q.element_size() / 1e6:.1f} MB\n")

    print("朴素 Attention 各步骤:")
    O = naive_attention_manual(Q, K, V)
    print(f"  输出大小: {list(O.shape)}, 显存: {O.nelement() * O.element_size() / 1e6:.1f} MB")

    # Part 2: 显存 scaling 分析
    analyze_memory_scaling()

    # Part 3: 性能对比
    performance_comparison()


if __name__ == "__main__":
    main()
