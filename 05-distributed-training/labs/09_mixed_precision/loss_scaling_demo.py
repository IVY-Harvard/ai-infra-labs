"""
Lab 09 - Loss Scaling 原理演示
================================
深入理解为什么 FP16 需要 Loss Scaling。

FP16 最小正规化数: 2^(-14) ≈ 6.1e-5
如果梯度 < 6.1e-5，在 FP16 中会变成 0 (下溢)

Loss Scaling 解决方案:
  1. 将 loss 乘以 scale (如 1024)
  2. 反向传播得到放大的梯度 (不会下溢)
  3. optimizer step 前除以 scale 恢复
  4. 如果梯度中有 inf → scale 减半; 否则 → scale 加倍

运行:
    python loss_scaling_demo.py
"""

import torch
import torch.nn as nn


def demo_underflow():
    """演示 FP16 梯度下溢"""
    print("=" * 60)
    print("FP16 梯度下溢演示")
    print("=" * 60)

    # 模拟一个小梯度
    small_grads = [1e-5, 1e-6, 1e-7, 1e-8]

    print(f"  {'FP32 值':<12} {'FP16 值':<12} {'下溢?':<8}")
    print(f"  {'-'*32}")
    for g in small_grads:
        fp32_val = torch.tensor(g, dtype=torch.float32)
        fp16_val = fp32_val.half()
        underflow = "是" if fp16_val.item() == 0 else "否"
        print(f"  {g:<12.1e} {fp16_val.item():<12.1e} {underflow:<8}")

    print(f"\n  FP16 最小正规化数: {torch.finfo(torch.float16).tiny:.2e}")
    print(f"  FP16 最大值:       {torch.finfo(torch.float16).max:.2e}")
    print(f"  BF16 最小正规化数: {torch.finfo(torch.bfloat16).tiny:.2e}")
    print(f"  BF16 最大值:       {torch.finfo(torch.bfloat16).max:.2e}")


def demo_loss_scaling():
    """演示 Loss Scaling 如何防止下溢"""
    print(f"\n{'='*60}")
    print("Loss Scaling 机制")
    print("=" * 60)

    grad_value = 1e-7  # 这个值在 FP16 中会下溢

    print(f"\n  原始梯度: {grad_value:.1e}")
    print(f"  FP16(原始): {torch.tensor(grad_value).half().item():.1e} (下溢!)")

    for scale in [256, 1024, 65536]:
        scaled = grad_value * scale
        fp16_scaled = torch.tensor(scaled).half().item()
        recovered = fp16_scaled / scale
        print(f"\n  Scale={scale:>6}:")
        print(f"    缩放后:   {scaled:.1e}")
        print(f"    FP16:     {fp16_scaled:.1e}")
        print(f"    恢复后:   {recovered:.1e}")
        print(f"    误差:     {abs(recovered - grad_value) / grad_value * 100:.1f}%")


def demo_dynamic_scaling():
    """演示动态 Loss Scaling"""
    print(f"\n{'='*60}")
    print("动态 Loss Scaling 模拟")
    print("=" * 60)

    scale = 65536.0  # 初始 scale
    growth_factor = 2.0
    backoff_factor = 0.5
    growth_interval = 5  # 连续 N 步无 overflow → 增大 scale

    steps_since_growth = 0

    print(f"\n  {'Step':<6} {'Scale':<10} {'Overflow':<10} {'Action':<15}")
    print(f"  {'-'*41}")

    for step in range(20):
        # 模拟: 某些步骤梯度会 overflow
        has_overflow = (step in [3, 4, 12])

        if has_overflow:
            scale *= backoff_factor
            steps_since_growth = 0
            action = "scale /= 2"
        else:
            steps_since_growth += 1
            if steps_since_growth >= growth_interval:
                scale *= growth_factor
                steps_since_growth = 0
                action = "scale *= 2"
            else:
                action = ""

        overflow_str = "YES" if has_overflow else ""
        print(f"  {step:<6} {scale:<10.0f} {overflow_str:<10} {action:<15}")

    print(f"\n  关键: Scale 在 overflow 时缩小，稳定后逐渐增大")
    print(f"  BF16 不需要这个机制，因为其指数范围与 FP32 相同")


def demo_fp32_vs_bf16_vs_fp16():
    """数值格式对比"""
    print(f"\n{'='*60}")
    print("数值格式对比")
    print("=" * 60)

    formats = {
        "FP32": torch.float32,
        "FP16": torch.float16,
        "BF16": torch.bfloat16,
    }

    print(f"\n  {'格式':<8} {'Sign':<6} {'Exp':<6} {'Mantissa':<10} {'Max':<12} {'Min (正规)':<12} {'精度 (eps)':<12}")
    print(f"  {'-'*66}")

    layout = {
        "FP32": (1, 8, 23),
        "FP16": (1, 5, 10),
        "BF16": (1, 8, 7),
    }

    for name, dtype in formats.items():
        info = torch.finfo(dtype)
        s, e, m = layout[name]
        print(f"  {name:<8} {s:<6} {e:<6} {m:<10} {info.max:<12.2e} {info.tiny:<12.2e} {info.eps:<12.2e}")

    print(f"\n  推荐: H20 使用 BF16")
    print(f"    - 范围与 FP32 相同 → 不需要 loss scaling")
    print(f"    - 精度略低 → 对 LLM 训练几乎无影响")
    print(f"    - 代码简洁 → 减少 bug 来源")


if __name__ == "__main__":
    demo_underflow()
    demo_loss_scaling()
    demo_dynamic_scaling()
    demo_fp32_vs_bf16_vs_fp16()
