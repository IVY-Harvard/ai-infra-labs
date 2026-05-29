"""
Lab 05: FlashAttention 原理分析

不是实现 FlashAttention（那需要 Triton/CUDA），而是：
1. 用 Python 模拟 FlashAttention 的分块计算逻辑
2. 统计 HBM 访问次数的差异
3. 验证在线 Softmax 的正确性

Usage: python flash_attention_analysis.py
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple


def standard_attention_with_io_count(Q, K, V):
    """
    标准 Attention + 统计 HBM 读写量。

    Q, K, V: [seq_len, head_dim]

    返回: (output, hbm_reads_bytes, hbm_writes_bytes)
    """
    N, d = Q.shape
    bytes_per_elem = Q.element_size()

    hbm_reads = 0
    hbm_writes = 0

    # S = Q @ K^T : 读 Q(Nd) + K(Nd), 写 S(N²)
    S = Q @ K.T / math.sqrt(d)
    hbm_reads += (N * d + N * d) * bytes_per_elem
    hbm_writes += (N * N) * bytes_per_elem

    # P = softmax(S) : 读 S(N²), 写 P(N²)
    P = F.softmax(S, dim=-1)
    hbm_reads += (N * N) * bytes_per_elem
    hbm_writes += (N * N) * bytes_per_elem

    # O = P @ V : 读 P(N²) + V(Nd), 写 O(Nd)
    O = P @ V
    hbm_reads += (N * N + N * d) * bytes_per_elem
    hbm_writes += (N * d) * bytes_per_elem

    return O, hbm_reads, hbm_writes


def flash_attention_simulation(Q, K, V, block_size=64):
    """
    模拟 FlashAttention 的分块计算 + 在线 Softmax。

    核心思想：
    - 不存储完整的 N×N 注意力矩阵
    - 分块计算，每块只处理 block_size × block_size 的子矩阵
    - 用在线 Softmax 算法递推更新

    Q, K, V: [seq_len, head_dim]
    返回: (output, hbm_reads_bytes, hbm_writes_bytes)
    """
    N, d = Q.shape
    bytes_per_elem = Q.element_size()
    Br = block_size  # Q 的块大小
    Bc = block_size  # K/V 的块大小

    hbm_reads = 0
    hbm_writes = 0

    # 输出和辅助变量
    O = torch.zeros(N, d, dtype=Q.dtype, device=Q.device)
    l = torch.zeros(N, dtype=Q.dtype, device=Q.device)  # softmax 分母
    m = torch.full((N,), float('-inf'), dtype=Q.dtype, device=Q.device)  # 行最大值

    # 外层循环: 遍历 K/V 的块
    num_blocks_kv = math.ceil(N / Bc)
    num_blocks_q = math.ceil(N / Br)

    for j in range(num_blocks_kv):
        # 加载 K_j 和 V_j 到 SRAM (实际在共享内存中)
        kv_start = j * Bc
        kv_end = min(kv_start + Bc, N)
        K_j = K[kv_start:kv_end]  # [Bc, d]
        V_j = V[kv_start:kv_end]  # [Bc, d]
        hbm_reads += 2 * (kv_end - kv_start) * d * bytes_per_elem

        # 内层循环: 遍历 Q 的块
        for i in range(num_blocks_q):
            q_start = i * Br
            q_end = min(q_start + Br, N)
            Q_i = Q[q_start:q_end]  # [Br, d]
            hbm_reads += (q_end - q_start) * d * bytes_per_elem

            # 读取当前的 O_i, l_i, m_i
            O_i = O[q_start:q_end].clone()
            l_i = l[q_start:q_end].clone()
            m_i = m[q_start:q_end].clone()
            hbm_reads += ((q_end-q_start) * d + 2 * (q_end-q_start)) * bytes_per_elem

            # ---- 以下全在 SRAM 中计算（不产生 HBM 访问） ----

            # 计算局部注意力分数 S_ij = Q_i @ K_j^T
            S_ij = Q_i @ K_j.T / math.sqrt(d)  # [Br, Bc] — 在 SRAM 中

            # 在线 Softmax: 更新 max 和 sum
            m_ij = S_ij.max(dim=-1).values  # 当前块的行最大值
            m_new = torch.maximum(m_i, m_ij)  # 全局行最大值更新

            # 计算 exp
            exp_S = torch.exp(S_ij - m_new.unsqueeze(-1))
            exp_old = torch.exp(m_i - m_new)

            # 更新 softmax 分母
            l_new = exp_old * l_i + exp_S.sum(dim=-1)

            # 更新输出
            # 关键公式: O_new = (exp_old * l_i * O_old + exp_S @ V_j) / l_new
            O_new = (exp_old.unsqueeze(-1) * l_i.unsqueeze(-1) * O_i + exp_S @ V_j) / l_new.unsqueeze(-1)

            # ---- 写回 HBM ----
            O[q_start:q_end] = O_new
            l[q_start:q_end] = l_new
            m[q_start:q_end] = m_new
            hbm_writes += ((q_end-q_start) * d + 2 * (q_end-q_start)) * bytes_per_elem

    return O, hbm_reads, hbm_writes


def io_complexity_analysis():
    """分析标准 Attention vs FlashAttention 的 IO 复杂度"""
    print("=" * 60)
    print("HBM IO 复杂度分析: Standard vs FlashAttention")
    print("=" * 60)

    head_dim = 128
    sram_size = 228 * 1024  # 228 KB (H20 的共享内存)

    # FlashAttention 的最优 block size
    # Bc = ceil(M / (4d)), 其中 M = SRAM size
    block_size = min(128, sram_size // (4 * head_dim * 4))  # 4 bytes per float
    print(f"\nhead_dim={head_dim}, SRAM={sram_size/1024:.0f}KB, block_size={block_size}")

    seq_lengths = [512, 1024, 2048, 4096, 8192]
    print(f"\n{'seq_len':>8} | {'标准IO(MB)':>12} | {'Flash IO(MB)':>12} | {'IO节省':>8}")
    print("-" * 55)

    for N in seq_lengths:
        # 标准 Attention IO: O(N²d + N²)
        # 具体: 读 Q,K(2Nd) + 写S(N²) + 读S(N²) + 写P(N²) + 读P,V(N²+Nd) + 写O(Nd)
        standard_io = (2*N*head_dim + 3*N*N + N*head_dim + N*head_dim) * 4  # bytes

        # FlashAttention IO: O(N²d² / M)
        # 具体分析: T_c = ceil(N/Bc) 个 KV 块
        # 每个 KV 块: 读 KV(2·Bc·d) + T_r 次 (读Q_i,O_i,l_i,m_i + 写O_i,l_i,m_i)
        Bc = block_size
        Br = block_size
        T_c = math.ceil(N / Bc)
        T_r = math.ceil(N / Br)
        flash_io = T_c * (2 * Bc * head_dim * 4  # 读 K_j, V_j
                         + T_r * (Br * head_dim * 4  # 读 Q_i
                                 + 2 * Br * head_dim * 4  # 读写 O_i
                                 + 4 * Br * 4))  # 读写 l_i, m_i

        standard_mb = standard_io / 1e6
        flash_mb = flash_io / 1e6
        saving = (1 - flash_io / standard_io) * 100

        print(f"{N:>8} | {standard_mb:>12.1f} | {flash_mb:>12.1f} | {saving:>7.1f}%")

    print(f"\n关键观察:")
    print(f"  - FlashAttention 的 IO 不包含 N×N 矩阵的完整读写")
    print(f"  - 序列越长，IO 节省越显著")
    print(f"  - IO 节省 ≈ 计算速度提升（因为 Attention 是 memory-bound）")


def verify_correctness():
    """验证 FlashAttention 模拟的正确性"""
    print("\n" + "=" * 60)
    print("正确性验证: Flash Attention 分块计算 vs 标准实现")
    print("=" * 60)

    torch.manual_seed(42)
    N, d = 256, 64  # 用小矩阵验证
    Q = torch.randn(N, d)
    K = torch.randn(N, d)
    V = torch.randn(N, d)

    # 标准实现
    O_standard, std_reads, std_writes = standard_attention_with_io_count(Q, K, V)

    # FlashAttention 模拟
    O_flash, flash_reads, flash_writes = flash_attention_simulation(Q, K, V, block_size=32)

    # 检查误差
    max_diff = (O_standard - O_flash).abs().max().item()
    mean_diff = (O_standard - O_flash).abs().mean().item()

    print(f"\n  N={N}, d={d}, block_size=32")
    print(f"  最大绝对误差: {max_diff:.2e}")
    print(f"  平均绝对误差: {mean_diff:.2e}")
    print(f"  正确性: {'PASS' if max_diff < 1e-4 else 'FAIL'}")

    print(f"\n  HBM 读写量对比:")
    print(f"    标准:  读 {std_reads/1e6:.1f} MB, 写 {std_writes/1e6:.1f} MB, 总 {(std_reads+std_writes)/1e6:.1f} MB")
    print(f"    Flash: 读 {flash_reads/1e6:.1f} MB, 写 {flash_writes/1e6:.1f} MB, 总 {(flash_reads+flash_writes)/1e6:.1f} MB")
    print(f"    节省: {(1 - (flash_reads+flash_writes)/(std_reads+std_writes))*100:.1f}%")


def online_softmax_demo():
    """演示在线 Softmax 算法"""
    print("\n" + "=" * 60)
    print("在线 Softmax 算法演示")
    print("=" * 60)

    # 标准 softmax 需要两遍扫描: 先找 max，再算 exp/sum
    # 在线 softmax 只需要一遍扫描，边扫描边更新

    x = torch.randn(16)
    print(f"\n  输入 x: {x[:8].tolist()[:4]}... (长度 {len(x)})")

    # 标准 softmax
    standard_result = F.softmax(x, dim=0)

    # 在线 softmax（模拟分块处理）
    block_size = 4
    m = float('-inf')  # running max
    l = 0.0            # running sum of exp

    # 存储中间结果以展示递推过程
    print(f"\n  分块处理 (block_size={block_size}):")
    for i in range(0, len(x), block_size):
        block = x[i:i+block_size]
        m_block = block.max().item()
        m_new = max(m, m_block)

        # 更新: l = l * exp(m_old - m_new) + sum(exp(block - m_new))
        l = l * math.exp(m - m_new) + torch.exp(block - m_new).sum().item()
        m = m_new
        print(f"    Block {i//block_size}: m={m:.4f}, l={l:.4f}")

    # 最终计算
    online_result = torch.exp(x - m) / l
    max_error = (standard_result - online_result).abs().max().item()
    print(f"\n  在线 Softmax 误差: {max_error:.2e}")
    print(f"  正确性: {'PASS' if max_error < 1e-6 else 'FAIL'}")
    print(f"\n  核心思想: 通过维护 running max 和 running sum，")
    print(f"  可以在不知道全局 max 的情况下逐块计算 softmax。")
    print(f"  这使得 FlashAttention 可以逐块处理而不需要存储完整 N×N 矩阵。")


def main():
    print("Lab 05: FlashAttention 原理分析\n")

    # 1. 验证分块计算的正确性
    verify_correctness()

    # 2. 在线 Softmax 演示
    online_softmax_demo()

    # 3. IO 复杂度分析
    io_complexity_analysis()

    print("\n" + "=" * 60)
    print("总结:")
    print("  1. FlashAttention 不减少 FLOPs，只减少 HBM 访问")
    print("  2. 核心技巧: 在线 Softmax + 分块计算")
    print("  3. Attention 是 memory-bound → 减少 IO = 减少时间")
    print("  4. 额外好处: 不需要 O(N²) 显存，支持超长序列")
    print("=" * 60)


if __name__ == "__main__":
    main()
