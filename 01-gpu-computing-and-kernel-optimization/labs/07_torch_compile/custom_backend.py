"""
Lab 07: 自定义 torch.compile 后端

理解 TorchDynamo 的后端机制:
- Dynamo 追踪得到 FX Graph
- 后端负责将 FX Graph 编译为可执行代码
- 我们可以注册自定义后端来理解编译流程

Usage: python custom_backend.py
"""

import torch
import torch.nn.functional as F
from torch._dynamo import register_backend
from torch.fx import GraphModule
import time


def inspect_backend(gm: GraphModule, example_inputs):
    """
    自定义后端: 只打印计算图信息，不做优化。

    TorchDynamo 会把追踪到的计算图传给后端。
    后端的职责是返回一个可调用对象（优化后的函数）。
    这个"后端"只是打印图信息，然后原样返回。
    """
    print("\n" + "=" * 50)
    print("自定义后端收到的 FX Graph:")
    print("=" * 50)

    # 打印图的节点
    print("\n节点列表:")
    for node in gm.graph.nodes:
        print(f"  {node.op:>15} | {str(node.name):>20} | target: {node.target}")

    # 打印可读的代码表示
    print(f"\n生成的 Python 代码:")
    print(gm.code)

    # 打印输入信息
    print(f"输入数量: {len(example_inputs)}")
    for i, inp in enumerate(example_inputs):
        if isinstance(inp, torch.Tensor):
            print(f"  input[{i}]: shape={inp.shape}, dtype={inp.dtype}, device={inp.device}")

    # 返回原始的 GraphModule 作为"编译后"的可执行对象
    return gm.forward


def counting_backend(gm: GraphModule, example_inputs):
    """
    统计后端: 统计各类操作的数量。

    帮助你理解一个模型/函数中有多少操作，
    哪些可以融合，哪些是关键瓶颈。
    """
    op_counts = {}
    total_nodes = 0

    for node in gm.graph.nodes:
        if node.op == 'call_function':
            op_name = str(node.target).split('.')[-1]
            op_counts[op_name] = op_counts.get(op_name, 0) + 1
            total_nodes += 1

    print(f"\n算子统计 (共 {total_nodes} 个函数调用):")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"  {op}: {count}")

    return gm.forward


def demo_inspect():
    """使用 inspect 后端查看计算图"""
    print("=" * 60)
    print("Demo 1: 查看计算图结构")
    print("=" * 60)

    def my_function(x, y):
        z = x + y
        z = z * 2
        z = F.relu(z)
        return z.sum()

    # 使用自定义后端编译
    compiled_fn = torch.compile(my_function, backend=inspect_backend)

    x = torch.randn(4, 4, device='cuda')
    y = torch.randn(4, 4, device='cuda')

    # 第一次调用时触发编译
    result = compiled_fn(x, y)
    print(f"\n结果: {result.item():.4f}")


def demo_counting():
    """统计模型中的操作类型"""
    print("\n" + "=" * 60)
    print("Demo 2: 统计模型操作类型")
    print("=" * 60)

    class MiniModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(256, 512)
            self.linear2 = torch.nn.Linear(512, 256)
            self.ln = torch.nn.LayerNorm(256)

        def forward(self, x):
            residual = x
            x = self.linear1(x)
            x = F.gelu(x)
            x = self.linear2(x)
            x = F.dropout(x, 0.1, training=self.training)
            x = self.ln(x + residual)
            return x

    model = MiniModel().cuda().half()
    model.eval()

    compiled_model = torch.compile(model, backend=counting_backend)

    with torch.no_grad():
        x = torch.randn(32, 128, 256, device='cuda', dtype=torch.float16)
        _ = compiled_model(x)


def demo_compare_backends():
    """对比不同编译模式的效果"""
    print("\n" + "=" * 60)
    print("Demo 3: 对比编译后端效果")
    print("=" * 60)

    def compute_heavy(x):
        """多个 pointwise 操作的链式调用"""
        x = x * 2.0
        x = x + 1.0
        x = torch.sigmoid(x)
        x = x * 3.0
        x = F.relu(x)
        x = x - 0.5
        x = torch.tanh(x)
        x = x ** 2
        return x

    x = torch.randn(4096, 4096, device='cuda')

    # Eager
    eager_times = []
    for _ in range(50):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = compute_heavy(x)
        torch.cuda.synchronize()
        eager_times.append(time.perf_counter() - start)
    eager_ms = sum(eager_times[10:]) / 40 * 1000  # 跳过 warmup

    # Compiled (default)
    compiled_fn = torch.compile(compute_heavy)
    compiled_times = []
    for _ in range(50):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = compiled_fn(x)
        torch.cuda.synchronize()
        compiled_times.append(time.perf_counter() - start)
    compiled_ms = sum(compiled_times[10:]) / 40 * 1000

    print(f"  8 个 pointwise 操作链:")
    print(f"  Eager:    {eager_ms:.3f} ms (8 个 kernel)")
    print(f"  Compiled: {compiled_ms:.3f} ms (融合为 1 个 kernel)")
    print(f"  加速比: {eager_ms / compiled_ms:.2f}x")
    print(f"\n  Inductor 将 8 个 kernel 融合为 1 个 Triton kernel")
    print(f"  HBM 访问从 8 次减少到 1 次，这是加速的主要来源")


def main():
    if not torch.cuda.is_available():
        print("需要 CUDA GPU")
        return

    demo_inspect()
    demo_counting()
    demo_compare_backends()

    print("\n" + "=" * 60)
    print("总结:")
    print("  1. TorchDynamo 将 Python 代码追踪为 FX Graph")
    print("  2. FX Graph 中的节点对应具体的 tensor 操作")
    print("  3. 自定义后端可以分析、修改、优化这个图")
    print("  4. Inductor 是默认后端，生成融合的 Triton kernel")
    print("  5. pointwise 操作链是融合的最佳场景")
    print("=" * 60)


if __name__ == "__main__":
    main()
