"""
Lab 07: torch.compile 实战

展示 torch.compile 的用法、效果和注意事项。

Usage: python compile_demo.py

设置 TORCH_LOGS="output_code" 可以看到 Inductor 生成的 Triton 代码:
    TORCH_LOGS="output_code" python compile_demo.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os


# ============================================================
# 示例 1: 基本用法 — 简单函数编译
# ============================================================

def simple_function(x, y):
    """一系列 pointwise 操作，非常适合融合"""
    z = x + y
    z = z * 2.0
    z = F.relu(z)
    z = z - 1.0
    z = torch.sigmoid(z)
    return z


# ============================================================
# 示例 2: 模型编译
# ============================================================

class SimpleTransformerBlock(nn.Module):
    """简化版 Transformer Block"""
    def __init__(self, hidden_size=1024, num_heads=16, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.ffn_up = nn.Linear(hidden_size, 4 * hidden_size)
        self.ffn_down = nn.Linear(4 * hidden_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-Attention
        residual = x
        x = self.ln1(x)
        B, T, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 使用 SDPA（自动选择后端）
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, C)
        x = residual + self.dropout(self.out_proj(attn_out))

        # FFN
        residual = x
        x = self.ln2(x)
        x = self.ffn_up(x)
        x = F.gelu(x)
        x = self.ffn_down(x)
        x = residual + self.dropout(x)

        return x


# ============================================================
# 示例 3: Graph Break 演示
# ============================================================

def function_with_graph_break(x):
    """
    这个函数包含会导致 graph break 的代码。
    graph break = Dynamo 无法追踪的操作，导致图被拆分。
    """
    x = x * 2
    x = F.relu(x)

    # Graph Break! print 是 Python 副作用，Dynamo 无法追踪
    # print(f"intermediate shape: {x.shape}")

    # Graph Break! 将 tensor 转为 Python 数据类型
    # if x.sum().item() > 0:
    #     x = x * 3

    # 安全做法: 用 tensor 操作代替 Python 控制流
    mask = (x.sum() > 0).float()
    x = x * (2 * mask + 1)  # 等效但不会 break

    x = torch.sigmoid(x)
    return x


def benchmark_function(fn, *args, warmup=10, runs=100, label=""):
    """通用计时工具"""
    # Warmup
    for _ in range(warmup):
        _ = fn(*args)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(runs):
        _ = fn(*args)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / runs * 1000

    print(f"  {label}: {elapsed:.3f} ms")
    return elapsed


def demo_basic_compile():
    """Demo 1: 基本函数编译"""
    print("=" * 60)
    print("Demo 1: 基本函数编译")
    print("=" * 60)

    x = torch.randn(4096, 4096, device='cuda')
    y = torch.randn(4096, 4096, device='cuda')

    # 编译函数
    compiled_fn = torch.compile(simple_function)

    # 对比性能
    eager_ms = benchmark_function(simple_function, x, y, label="Eager mode")
    compiled_ms = benchmark_function(compiled_fn, x, y, label="Compiled  ")
    print(f"  加速比: {eager_ms / compiled_ms:.2f}x")
    print(f"  原因: 5 个 pointwise op 融合成 1 个 kernel\n")


def demo_model_compile():
    """Demo 2: 模型编译"""
    print("=" * 60)
    print("Demo 2: Transformer Block 编译")
    print("=" * 60)

    device = torch.device('cuda')
    model = SimpleTransformerBlock(hidden_size=1024, num_heads=16).to(device).half()
    model.eval()

    x = torch.randn(4, 512, 1024, device=device, dtype=torch.float16)

    # 不同的编译模式
    modes = {
        "default": torch.compile(model),
        "reduce-overhead": torch.compile(model, mode="reduce-overhead"),
        "max-autotune": torch.compile(model, mode="max-autotune"),
    }

    with torch.no_grad():
        eager_ms = benchmark_function(model, x, warmup=20, runs=50, label="Eager        ")

        for mode_name, compiled_model in modes.items():
            ms = benchmark_function(compiled_model, x, warmup=20, runs=50,
                                   label=f"Compiled ({mode_name:>16})")
            print(f"    → 加速: {eager_ms / ms:.2f}x")

    print(f"\n  模式说明:")
    print(f"  - default: 标准编译，平衡编译时间和运行时性能")
    print(f"  - reduce-overhead: 使用 CUDA Graphs 减少 launch overhead")
    print(f"  - max-autotune: 尝试更多的 kernel 配置，编译更慢但可能更快\n")


def demo_compile_modes():
    """Demo 3: 查看生成的代码"""
    print("=" * 60)
    print("Demo 3: 查看 Inductor 输出")
    print("=" * 60)

    print("  运行以下命令可以看到 Inductor 生成的 Triton kernel:")
    print('  TORCH_LOGS="output_code" python -c "')
    print("    import torch")
    print("    @torch.compile")
    print("    def fn(x):")
    print("        return torch.relu(x * 2 + 1)")
    print("    fn(torch.randn(1024, device='cuda'))")
    print('  "')

    print(f"\n  生成的代码大致是这样的 (示意):")
    print(f"  @triton.jit")
    print(f"  def triton_kernel(in_ptr, out_ptr, n, BLOCK: tl.constexpr):")
    print(f"      pid = tl.program_id(0)")
    print(f"      offsets = pid * BLOCK + tl.arange(0, BLOCK)")
    print(f"      x = tl.load(in_ptr + offsets)")
    print(f"      # x*2+1 和 relu 融合在一起:")
    print(f"      result = tl.maximum(x * 2.0 + 1.0, 0.0)")
    print(f"      tl.store(out_ptr + offsets, result)")


def demo_graph_break():
    """Demo 4: Graph Break"""
    print("\n" + "=" * 60)
    print("Demo 4: Graph Break 分析")
    print("=" * 60)

    # 正常函数 — 无 graph break
    def good_function(x):
        x = x * 2
        x = F.relu(x)
        x = torch.sigmoid(x)
        return x

    # 有 graph break 的函数
    def bad_function(x):
        x = x * 2
        x = F.relu(x)
        # .item() 导致 graph break: tensor → Python scalar
        if x.sum().item() > 0:
            x = x * 3
        x = torch.sigmoid(x)
        return x

    x = torch.randn(2048, 2048, device='cuda')

    compiled_good = torch.compile(good_function)
    compiled_bad = torch.compile(bad_function)

    ms_good_eager = benchmark_function(good_function, x, label="Good (eager)   ")
    ms_good_compile = benchmark_function(compiled_good, x, label="Good (compiled)")
    print(f"  加速比: {ms_good_eager / ms_good_compile:.2f}x")

    print()

    ms_bad_eager = benchmark_function(bad_function, x, label="Bad (eager)    ")
    ms_bad_compile = benchmark_function(compiled_bad, x, label="Bad (compiled) ")
    print(f"  加速比: {ms_bad_eager / ms_bad_compile:.2f}x")

    print(f"\n  观察: graph break 导致编译收益降低")
    print(f"  常见 graph break 原因:")
    print(f"    - tensor.item() / tensor.tolist()")
    print(f"    - print() / logging")
    print(f"    - 基于 tensor 值的 Python if/for")
    print(f"    - 不支持的第三方库调用")


def main():
    if not torch.cuda.is_available():
        print("需要 CUDA GPU")
        return

    demo_basic_compile()
    demo_model_compile()
    demo_compile_modes()
    demo_graph_break()

    print("\n" + "=" * 60)
    print("总结:")
    print("  1. torch.compile 对 pointwise 融合效果最好")
    print("  2. reduce-overhead 模式适合推理（CUDA Graphs）")
    print("  3. 避免 graph break = 避免 Python 副作用")
    print("  4. 设置 TORCH_LOGS='output_code' 查看生成的代码")
    print("=" * 60)


if __name__ == "__main__":
    main()
