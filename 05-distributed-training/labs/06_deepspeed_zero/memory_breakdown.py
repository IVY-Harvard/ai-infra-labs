"""
Lab 06 - ZeRO 各 Stage 显存拆解
=================================
理论计算 + 实际测量各 Stage 的显存分布。

运行:
    python memory_breakdown.py

无需多卡，纯理论计算 + 公式验证。
"""

import torch


def memory_breakdown(model_params_b: float, num_gpus: int, dtype_bytes: int = 2):
    """
    计算 DDP / ZeRO Stage 1/2/3 的显存分布。

    参数:
        model_params_b: 模型参数量 (十亿)
        num_gpus: GPU 数量
        dtype_bytes: 参数数据类型字节数 (BF16=2)
    """
    N = num_gpus
    phi = model_params_b * 1e9  # 参数个数

    # 各组件大小 (bytes)
    param_size = phi * dtype_bytes              # BF16 参数
    grad_size = phi * dtype_bytes               # BF16 梯度
    # Adam 优化器: FP32 master weight + FP32 momentum + FP32 variance
    opt_size = phi * 4 * 3                      # 12Φ bytes

    print(f"\n{'='*60}")
    print(f"显存拆解: {model_params_b}B 模型, {N} GPU, BF16 混合精度")
    print(f"{'='*60}")
    print(f"\n组件大小:")
    print(f"  参数 (BF16):     {param_size/1e9:.1f} GB")
    print(f"  梯度 (BF16):     {grad_size/1e9:.1f} GB")
    print(f"  优化器 (FP32):   {opt_size/1e9:.1f} GB")
    print(f"  总计 (不含激活): {(param_size+grad_size+opt_size)/1e9:.1f} GB")

    print(f"\n{'方案':<20} {'参数':<10} {'梯度':<10} {'优化器':<12} {'总计/卡':<12} {'通信量':<8}")
    print(f"{'-'*72}")

    # DDP
    ddp_total = (param_size + grad_size + opt_size) / 1e9
    print(f"{'DDP':<20} {param_size/1e9:<10.1f} {grad_size/1e9:<10.1f} "
          f"{opt_size/1e9:<12.1f} {ddp_total:<12.1f} {'2M':<8}")

    # ZeRO Stage 1
    z1_opt = opt_size / N
    z1_total = (param_size + grad_size + z1_opt) / 1e9
    print(f"{'ZeRO Stage 1':<20} {param_size/1e9:<10.1f} {grad_size/1e9:<10.1f} "
          f"{z1_opt/1e9:<12.1f} {z1_total:<12.1f} {'3M':<8}")

    # ZeRO Stage 2
    z2_grad = grad_size / N
    z2_opt = opt_size / N
    z2_total = (param_size + z2_grad + z2_opt) / 1e9
    print(f"{'ZeRO Stage 2':<20} {param_size/1e9:<10.1f} {z2_grad/1e9:<10.1f} "
          f"{z2_opt/1e9:<12.1f} {z2_total:<12.1f} {'2M':<8}")

    # ZeRO Stage 3
    z3_param = param_size / N
    z3_grad = grad_size / N
    z3_opt = opt_size / N
    z3_total = (z3_param + z3_grad + z3_opt) / 1e9
    z3_peak = z3_total + param_size / 1e9  # AllGather 时需要一层完整参数
    print(f"{'ZeRO Stage 3':<20} {z3_param/1e9:<10.1f} {z3_grad/1e9:<10.1f} "
          f"{z3_opt/1e9:<12.1f} {z3_total:<12.1f} {'3M':<8}")
    print(f"{'ZeRO Stage 3 (峰值)':<20} {'':10} {'':10} {'':12} {z3_peak:<12.1f}")

    print(f"\n节省比例 (相比 DDP):")
    print(f"  Stage 1: {(1-z1_total/ddp_total)*100:.0f}%")
    print(f"  Stage 2: {(1-z2_total/ddp_total)*100:.0f}%")
    print(f"  Stage 3: {(1-z3_total/ddp_total)*100:.0f}% (峰值 {(1-z3_peak/ddp_total)*100:.0f}%)")

    # H20 可行性分析
    h20_mem = 96  # GB
    print(f"\n{'='*60}")
    print(f"H20 (96 GB) 可行性分析 (预留 10 GB 给激活值和碎片)")
    print(f"{'='*60}")
    budget = h20_mem - 10
    results = [
        ("DDP", ddp_total),
        ("ZeRO Stage 1", z1_total),
        ("ZeRO Stage 2", z2_total),
        ("ZeRO Stage 3", z3_peak),
    ]
    for name, mem in results:
        status = "OK" if mem <= budget else "OOM"
        print(f"  {name:<20} {mem:.1f} GB  [{status}]")


def main():
    print("=" * 60)
    print("ZeRO 显存拆解分析")
    print("=" * 60)

    # 不同模型规模
    for model_b in [1.3, 7, 13, 70]:
        memory_breakdown(model_b, num_gpus=8)

    # 通信量对比
    print(f"\n\n{'='*60}")
    print("通信量分析 (7B 模型, BF16)")
    print("='*60")
    model_size_gb = 7e9 * 2 / 1e9  # BF16

    print(f"  DDP AllReduce:          2 × {model_size_gb:.1f} = {2*model_size_gb:.1f} GB/step")
    print(f"  Stage 1 (AR + AG):      3 × {model_size_gb:.1f} = {3*model_size_gb:.1f} GB/step")
    print(f"  Stage 2 (RS + AG):      2 × {model_size_gb:.1f} = {2*model_size_gb:.1f} GB/step")
    print(f"  Stage 3 (2AG + RS):     3 × {model_size_gb:.1f} = {3*model_size_gb:.1f} GB/step")
    print(f"\n  关键洞察: Stage 2 通信量 = DDP，但显存大幅节省！")
    print(f"  Stage 3 多 50% 通信但显存最优，适合模型过大的场景。")


if __name__ == "__main__":
    main()
