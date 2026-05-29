"""
Lab 09 - 混合精度显存节省分析
===============================
对比 FP32 / FP16 / BF16 混合精度的显存占用。

运行:
    python memory_saving_analysis.py
"""

import torch


def analyze_memory(model_params_b: float, batch_size: int = 4,
                   seq_len: int = 2048, hidden_size: int = 4096,
                   num_layers: int = 32):
    """分析不同精度下的显存占用"""

    phi = model_params_b * 1e9

    print(f"\n{'='*70}")
    print(f"{model_params_b}B 模型 | B={batch_size}, S={seq_len}, H={hidden_size}, L={num_layers}")
    print(f"{'='*70}")

    # --- FP32 训练 ---
    fp32_params = phi * 4
    fp32_grads = phi * 4
    fp32_adam = phi * 4 * 2  # momentum + variance
    fp32_total = (fp32_params + fp32_grads + fp32_adam) / 1e9

    # --- FP16/BF16 混合精度 ---
    # 参数: FP16/BF16 (2 bytes)
    # 梯度: FP16/BF16 (2 bytes)
    # Adam: FP32 master weight (4) + momentum (4) + variance (4) = 12
    mp_params = phi * 2
    mp_grads = phi * 2
    mp_adam = phi * 12  # master_weight + momentum + variance (all FP32)
    mp_total = (mp_params + mp_grads + mp_adam) / 1e9

    # --- 激活值 ---
    # 粗略估计: 每层约 2*B*S*H*dtype_bytes * 几个 tensor
    fp32_act_per_layer = 2 * batch_size * seq_len * hidden_size * 4 * 4 / 1e9  # ~4 个中间 tensor
    mp_act_per_layer = 2 * batch_size * seq_len * hidden_size * 2 * 4 / 1e9
    fp32_act = fp32_act_per_layer * num_layers
    mp_act = mp_act_per_layer * num_layers

    print(f"\n  {'组件':<20} {'FP32 (GB)':<15} {'混合精度 (GB)':<15} {'节省':<10}")
    print(f"  {'-'*60}")
    print(f"  {'参数':<20} {fp32_params/1e9:<15.1f} {mp_params/1e9:<15.1f} {(1-mp_params/fp32_params)*100:.0f}%")
    print(f"  {'梯度':<20} {fp32_grads/1e9:<15.1f} {mp_grads/1e9:<15.1f} {(1-mp_grads/fp32_grads)*100:.0f}%")
    print(f"  {'优化器':<20} {fp32_adam/1e9:<15.1f} {mp_adam/1e9:<15.1f} {(1-mp_adam/fp32_adam)*100:.0f}%")
    print(f"  {'激活值 (估算)':<20} {fp32_act:<15.1f} {mp_act:<15.1f} {(1-mp_act/fp32_act)*100:.0f}%")
    print(f"  {'-'*60}")
    fp32_total_with_act = fp32_total + fp32_act
    mp_total_with_act = mp_total + mp_act
    print(f"  {'总计':<20} {fp32_total_with_act:<15.1f} {mp_total_with_act:<15.1f} "
          f"{(1-mp_total_with_act/fp32_total_with_act)*100:.0f}%")

    print(f"\n  注意: 混合精度的优化器状态（12Φ bytes）占大头！")
    print(f"  这就是 ZeRO 主要切分的对象。")

    return fp32_total_with_act, mp_total_with_act


def throughput_analysis():
    """混合精度对吞吐量的影响"""
    print(f"\n\n{'='*70}")
    print("混合精度吞吐量分析")
    print("='*70")

    print(f"\n  H20 GPU 算力:")
    print(f"    FP32:  74 TFLOPS")
    print(f"    BF16: 148 TFLOPS (2x FP32)")
    print(f"    FP16: 148 TFLOPS")
    print(f"    TF32: 148 TFLOPS")
    print(f"\n  混合精度 → GEMM 用 BF16/FP16 → 算力翻倍!")
    print(f"  加上显存节省 → 可以用更大 batch → 更高吞吐")

    # 理论吞吐量对比
    print(f"\n  7B 模型训练吞吐量估算 (8×H20, TP=4, DP=2):")
    for precision, tflops, mfu in [("FP32", 74, 0.45), ("BF16", 148, 0.45)]:
        per_gpu = tflops * mfu
        total = per_gpu * 8
        flops_per_token = 6 * 7e9
        tps = total * 1e12 / flops_per_token
        print(f"    {precision}: {per_gpu:.0f} effective TFLOPS/GPU → {tps:.0f} tokens/sec total")


def main():
    for model_b in [1.3, 7, 13]:
        analyze_memory(model_b)

    throughput_analysis()

    print(f"\n\n{'='*70}")
    print("总结: 为什么推荐 BF16 混合精度")
    print("='*70")
    print("""
  1. 显存节省 ~40%（参数+梯度减半）
  2. 算力翻倍（BF16 TFLOPS = 2× FP32）
  3. 不需要 Loss Scaling（与 FP32 范围相同）
  4. 对模型质量几乎无影响
  5. H20 原生支持，性能稳定
    """)


if __name__ == "__main__":
    main()
