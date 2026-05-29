"""
Lab 08: Unfused vs Fused 算子对比

直观对比独立执行 vs 融合执行的性能差异。
理解为什么"少访问 HBM"如此重要。

Usage: python unfused_vs_fused.py
"""

import torch
import torch.nn.functional as F
import time


def benchmark(fn, *args, warmup=20, runs=100, label=""):
    """计时工具"""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        fn(*args)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / runs * 1000
    print(f"  {label}: {elapsed:.4f} ms")
    return elapsed


# ============================================================
# 实验 1: Pointwise 融合
# ============================================================

def pointwise_unfused(x):
    """5 个独立的 pointwise kernel"""
    x = x * 2.0         # Kernel 1: 读 x, 写 temp1
    x = x + 1.0         # Kernel 2: 读 temp1, 写 temp2
    x = F.relu(x)       # Kernel 3: 读 temp2, 写 temp3
    x = x * 0.5         # Kernel 4: 读 temp3, 写 temp4
    x = torch.sigmoid(x)  # Kernel 5: 读 temp4, 写 output
    return x
    # 总 HBM 访问: 10 次读写 (2 per kernel × 5)


def pointwise_fused_compile(x):
    """用 torch.compile 自动融合"""
    x = x * 2.0
    x = x + 1.0
    x = F.relu(x)
    x = x * 0.5
    x = torch.sigmoid(x)
    return x
    # 融合后 HBM 访问: 2 次读写 (1 read + 1 write)


# ============================================================
# 实验 2: Residual + LayerNorm + Dropout 融合
# ============================================================

def residual_ln_dropout_unfused(x, residual, weight, bias, eps=1e-5, p=0.1):
    """3 个独立操作"""
    # Kernel 1: Residual Add
    x = x + residual  # 读 x + residual, 写 temp1

    # Kernel 2: LayerNorm (内部可能是多个 kernel)
    x = F.layer_norm(x, x.shape[-1:], weight, bias, eps)

    # Kernel 3: Dropout
    x = F.dropout(x, p=p, training=True)

    return x


# ============================================================
# 实验 3: GEMM + Bias + ReLU 融合
# ============================================================

def gemm_bias_relu_unfused(x, weight, bias):
    """3 步分开执行"""
    x = F.linear(x, weight)  # Kernel 1: GEMM
    x = x + bias             # Kernel 2: Bias add
    x = F.relu(x)            # Kernel 3: ReLU
    return x


def gemm_bias_relu_onestep(x, weight, bias):
    """理想情况: GEMM epilogue 融合了 bias + relu"""
    # PyTorch 的 F.linear 在某些情况下会自动做 epilogue fusion
    x = F.linear(x, weight, bias)
    x = F.relu(x)
    return x


def experiment_pointwise():
    """实验 1: Pointwise 融合"""
    print("=" * 60)
    print("实验 1: Pointwise 操作融合")
    print("=" * 60)

    sizes = [(1024, 1024), (4096, 4096), (8192, 4096)]

    for H, W in sizes:
        print(f"\n  矩阵大小: ({H}, {W}), 数据量: {H*W*4/1e6:.1f} MB")
        x = torch.randn(H, W, device='cuda')

        # Unfused (eager)
        ms_unfused = benchmark(pointwise_unfused, x, label="Unfused (5 kernels)")

        # Fused (torch.compile)
        compiled_fn = torch.compile(pointwise_fused_compile)
        ms_fused = benchmark(compiled_fn, x, label="Fused (1 kernel)   ")

        speedup = ms_unfused / ms_fused
        print(f"  加速比: {speedup:.2f}x")

        # 分析
        bytes_per_elem = 4  # FP32
        n_elements = H * W
        unfused_hbm = n_elements * bytes_per_elem * 10  # 5 kernels × 2 accesses
        fused_hbm = n_elements * bytes_per_elem * 2     # 1 read + 1 write
        print(f"  HBM 访问: unfused={unfused_hbm/1e6:.0f}MB, fused={fused_hbm/1e6:.0f}MB")
        print(f"  HBM 节省: {(1 - fused_hbm/unfused_hbm)*100:.0f}%")


def experiment_residual_ln():
    """实验 2: Residual + LN + Dropout"""
    print("\n" + "=" * 60)
    print("实验 2: Residual + LayerNorm + Dropout 融合")
    print("=" * 60)

    batch, seq_len, hidden = 32, 512, 4096
    print(f"  形状: ({batch}, {seq_len}, {hidden})")

    x = torch.randn(batch, seq_len, hidden, device='cuda')
    residual = torch.randn_like(x)
    weight = torch.ones(hidden, device='cuda')
    bias = torch.zeros(hidden, device='cuda')

    # Unfused
    ms_unfused = benchmark(residual_ln_dropout_unfused, x, residual, weight, bias,
                           label="Unfused (separate ops)")

    # Fused with torch.compile
    compiled_fn = torch.compile(residual_ln_dropout_unfused)
    ms_fused = benchmark(compiled_fn, x, residual, weight, bias,
                        label="Fused (torch.compile)")

    print(f"  加速比: {ms_unfused / ms_fused:.2f}x")


def experiment_gemm_epilogue():
    """实验 3: GEMM + Bias + ReLU"""
    print("\n" + "=" * 60)
    print("实验 3: GEMM + Bias + ReLU (Epilogue Fusion)")
    print("=" * 60)

    M, K, N = 4096, 4096, 4096
    print(f"  GEMM: ({M}, {K}) × ({K}, {N})")

    x = torch.randn(M, K, device='cuda', dtype=torch.float16)
    weight = torch.randn(N, K, device='cuda', dtype=torch.float16)
    bias = torch.randn(N, device='cuda', dtype=torch.float16)

    ms_unfused = benchmark(gemm_bias_relu_unfused, x, weight, bias,
                           label="Unfused (GEMM + Bias + ReLU)")

    compiled_fn = torch.compile(gemm_bias_relu_onestep)
    ms_fused = benchmark(compiled_fn, x, weight, bias,
                        label="Compiled (epilogue fusion) ")

    print(f"  加速比: {ms_unfused / ms_fused:.2f}x")
    print(f"  注: GEMM 本身是 compute-bound, epilogue fusion 节省有限")
    print(f"  但对于小 GEMM (decode阶段), 节省的 kernel launch 很重要")


def main():
    if not torch.cuda.is_available():
        print("需要 CUDA GPU")
        return

    experiment_pointwise()
    experiment_residual_ln()
    experiment_gemm_epilogue()

    print("\n" + "=" * 60)
    print("总结:")
    print("  1. Pointwise 融合效果最好 (2-5x), 因为它们纯 memory-bound")
    print("  2. 融合的本质: 减少 HBM 读写次数")
    print("  3. Compute-bound 操作 (如 GEMM) 融合收益较小")
    print("  4. torch.compile 自动完成大部分融合, 无需手动")
    print("=" * 60)


if __name__ == "__main__":
    main()
