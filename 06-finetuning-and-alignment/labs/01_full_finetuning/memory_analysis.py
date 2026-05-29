"""
显存占用分析工具：详细分析 LLM 训练各组件的显存使用
帮助理解全量微调为什么需要那么多显存

用法:
    python memory_analysis.py --model_name Qwen/Qwen2-1.5B
    python memory_analysis.py --model_name Qwen/Qwen2-7B --estimate_only
"""

import argparse
import torch
import gc
from typing import Dict


def estimate_memory(num_params_billion: float, seq_len: int = 2048, batch_size: int = 4) -> Dict:
    """
    根据参数量估算训练显存需求（不实际加载模型）

    显存组成:
    1. 模型参数 (FP16/BF16): 2 bytes/param
    2. 梯度 (FP16/BF16): 2 bytes/param
    3. 优化器状态 (AdamW):
       - FP32 参数副本: 4 bytes/param
       - 一阶矩 (m): 4 bytes/param
       - 二阶矩 (v): 4 bytes/param
    4. 激活值: 取决于 seq_len, batch_size, hidden_dim, num_layers
    """
    num_params = num_params_billion * 1e9

    # 固定部分
    model_params_gb = num_params * 2 / 1e9       # FP16 参数
    gradients_gb = num_params * 2 / 1e9           # FP16 梯度
    optimizer_states_gb = num_params * 12 / 1e9   # AdamW: FP32 copy + m + v

    # 激活值估算 (粗略)
    # 近似公式: activations ≈ seq_len * batch_size * hidden_dim * num_layers * 2 bytes * factor
    # factor 考虑了 attention, MLP, LayerNorm 等中间结果
    if num_params_billion <= 2:
        hidden_dim, num_layers = 2048, 24
    elif num_params_billion <= 8:
        hidden_dim, num_layers = 4096, 32
    elif num_params_billion <= 15:
        hidden_dim, num_layers = 5120, 40
    elif num_params_billion <= 35:
        hidden_dim, num_layers = 6656, 60
    else:
        hidden_dim, num_layers = 8192, 80

    # 无 gradient checkpointing
    activations_no_gc = (
        seq_len * batch_size * hidden_dim * num_layers * 34 / 1e9
    )

    # 有 gradient checkpointing (约节省 60-70%)
    activations_with_gc = activations_no_gc * 0.35

    return {
        "num_params_B": num_params_billion,
        "model_params_GB": model_params_gb,
        "gradients_GB": gradients_gb,
        "optimizer_states_GB": optimizer_states_gb,
        "activations_no_gc_GB": activations_no_gc,
        "activations_with_gc_GB": activations_with_gc,
        "total_no_gc_GB": model_params_gb + gradients_gb + optimizer_states_gb + activations_no_gc,
        "total_with_gc_GB": model_params_gb + gradients_gb + optimizer_states_gb + activations_with_gc,
        "config": {
            "seq_len": seq_len,
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
        }
    }


def actual_memory_profile(model_name: str, seq_len: int = 512, batch_size: int = 2):
    """
    实际加载模型并测量显存（需要 GPU）
    使用较小的 seq_len 和 batch_size 以避免 OOM
    """
    if not torch.cuda.is_available():
        print("CUDA 不可用，跳过实际测量")
        return None

    from transformers import AutoModelForCausalLM, AutoTokenizer

    results = {}
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    baseline = torch.cuda.memory_allocated() / 1e9

    # Step 1: 加载模型
    print(f"  加载模型 {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    results["after_model_load_GB"] = torch.cuda.memory_allocated() / 1e9 - baseline
    print(f"  模型加载后显存: {results['after_model_load_GB']:.2f} GB")

    # Step 2: 创建优化器
    print("  创建优化器...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    results["after_optimizer_GB"] = torch.cuda.memory_allocated() / 1e9 - baseline
    print(f"  优化器创建后显存: {results['after_optimizer_GB']:.2f} GB")

    # Step 3: 前向传播
    print(f"  前向传播 (seq_len={seq_len}, batch={batch_size})...")
    dummy_input = torch.randint(0, 1000, (batch_size, seq_len), device="cuda:0")
    outputs = model(dummy_input, labels=dummy_input)
    results["after_forward_GB"] = torch.cuda.memory_allocated() / 1e9 - baseline
    results["forward_peak_GB"] = torch.cuda.max_memory_allocated() / 1e9 - baseline
    print(f"  前向传播后显存: {results['after_forward_GB']:.2f} GB")

    # Step 4: 反向传播
    print("  反向传播...")
    torch.cuda.reset_peak_memory_stats()
    outputs.loss.backward()
    results["after_backward_GB"] = torch.cuda.memory_allocated() / 1e9 - baseline
    results["backward_peak_GB"] = torch.cuda.max_memory_allocated() / 1e9 - baseline
    print(f"  反向传播后显存: {results['after_backward_GB']:.2f} GB")
    print(f"  峰值显存: {results['backward_peak_GB']:.2f} GB")

    # Step 5: 优化器更新
    optimizer.step()
    results["after_step_GB"] = torch.cuda.memory_allocated() / 1e9 - baseline
    print(f"  优化器更新后显存: {results['after_step_GB']:.2f} GB")

    # 清理
    del model, optimizer, outputs, dummy_input
    torch.cuda.empty_cache()
    gc.collect()

    return results


def print_comparison_table():
    """打印不同方法的显存对比表"""
    print("\n" + "=" * 80)
    print("不同微调方法的显存对比 (7B 模型, seq_len=2048, batch=4)")
    print("=" * 80)

    methods = [
        {
            "method": "Full FT (FP16)",
            "params_GB": 14, "grad_GB": 14, "optim_GB": 56,
            "act_GB": 30, "total_GB": 114, "notes": "需要 2×H20"
        },
        {
            "method": "Full FT (FP16+GC)",
            "params_GB": 14, "grad_GB": 14, "optim_GB": 56,
            "act_GB": 10, "total_GB": 94, "notes": "1×H20 紧凑可行"
        },
        {
            "method": "LoRA (r=64)",
            "params_GB": 14, "grad_GB": 0.3, "optim_GB": 1.9,
            "act_GB": 12, "total_GB": 28, "notes": "1×H20 轻松"
        },
        {
            "method": "QLoRA (4bit+r=64)",
            "params_GB": 3.5, "grad_GB": 0.3, "optim_GB": 1.9,
            "act_GB": 8, "total_GB": 14, "notes": "消费级 GPU"
        },
    ]

    header = f"{'方法':<20} {'参数':>8} {'梯度':>8} {'优化器':>8} {'激活值':>8} {'总计':>8}  {'备注'}"
    print(header)
    print("-" * 80)
    for m in methods:
        print(f"{m['method']:<20} {m['params_GB']:>7.1f}G {m['grad_GB']:>7.1f}G "
              f"{m['optim_GB']:>7.1f}G {m['act_GB']:>7.1f}G {m['total_GB']:>7.1f}G  {m['notes']}")


def main():
    parser = argparse.ArgumentParser(description="LLM 训练显存分析")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-1.5B",
                       help="模型名称或路径")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--estimate_only", action="store_true",
                       help="只进行理论估算，不实际加载模型")
    args = parser.parse_args()

    # 理论估算
    print("=" * 60)
    print("理论显存估算")
    print("=" * 60)

    for size in [1.5, 7, 13, 30, 70]:
        est = estimate_memory(size, args.seq_len, args.batch_size)
        print(f"\n--- {size}B 模型 (seq={args.seq_len}, batch={args.batch_size}) ---")
        print(f"  模型参数:     {est['model_params_GB']:>8.1f} GB")
        print(f"  梯度:         {est['gradients_GB']:>8.1f} GB")
        print(f"  优化器状态:   {est['optimizer_states_GB']:>8.1f} GB")
        print(f"  激活值(无GC): {est['activations_no_gc_GB']:>8.1f} GB")
        print(f"  激活值(有GC): {est['activations_with_gc_GB']:>8.1f} GB")
        print(f"  总计(无GC):   {est['total_no_gc_GB']:>8.1f} GB")
        print(f"  总计(有GC):   {est['total_with_gc_GB']:>8.1f} GB")

        # H20 适配建议
        total = est['total_with_gc_GB']
        if total <= 96:
            gpus = 1
        elif total <= 192:
            gpus = 2
        elif total <= 384:
            gpus = 4
        else:
            gpus = 8
        print(f"  → 建议 H20 卡数: {gpus} (使用 GC)")

    # 对比表
    print_comparison_table()

    # 实际测量
    if not args.estimate_only:
        print("\n" + "=" * 60)
        print(f"实际显存测量: {args.model_name}")
        print("=" * 60)

        results = actual_memory_profile(
            args.model_name,
            seq_len=min(args.seq_len, 512),  # 限制测量时的 seq_len
            batch_size=min(args.batch_size, 2),
        )

        if results:
            print("\n--- 显存分解 ---")
            for key, value in results.items():
                print(f"  {key}: {value:.2f} GB")


if __name__ == "__main__":
    main()
