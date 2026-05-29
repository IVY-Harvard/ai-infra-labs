"""
Lab 09: 多后端算子适配层 Demo

展示如何设计一个可扩展的多后端适配层。
虽然这里只实现了 CUDA 和 CPU 后端，但架构支持轻松添加新后端。

Usage: python backend_abstraction.py
"""

import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, List
import time


# ============================================================
# 1. 后端注册中心 (Backend Registry)
# ============================================================

class BackendRegistry:
    """
    后端注册中心: 管理所有可用的计算后端。

    设计原则:
    - 注册式: 新后端只需实现接口并注册
    - 自动检测: 根据环境自动选择最优后端
    - 可配置: 支持运行时切换后端
    """
    _backends: Dict[str, 'ComputeBackend'] = {}
    _current: Optional[str] = None

    @classmethod
    def register(cls, name: str, backend: 'ComputeBackend'):
        """注册一个新后端"""
        cls._backends[name] = backend
        print(f"  [Registry] 注册后端: {name}")

    @classmethod
    def get(cls, name: Optional[str] = None) -> 'ComputeBackend':
        """获取后端实例"""
        if name is None:
            name = cls._current or cls.auto_detect()
        if name not in cls._backends:
            raise ValueError(f"未注册的后端: {name}. 可用: {list(cls._backends.keys())}")
        return cls._backends[name]

    @classmethod
    def set_default(cls, name: str):
        """设置默认后端"""
        if name not in cls._backends:
            raise ValueError(f"未注册的后端: {name}")
        cls._current = name
        print(f"  [Registry] 默认后端设为: {name}")

    @classmethod
    def auto_detect(cls) -> str:
        """自动检测最优后端"""
        # 优先级: cuda > npu > mlu > cpu
        priority = ['cuda', 'npu', 'mlu', 'cpu']
        for name in priority:
            if name in cls._backends and cls._backends[name].is_available():
                cls._current = name
                return name
        raise RuntimeError("没有可用的计算后端")

    @classmethod
    def list_backends(cls) -> List[str]:
        return list(cls._backends.keys())


# ============================================================
# 2. 计算后端抽象基类
# ============================================================

class ComputeBackend(ABC):
    """
    计算后端抽象基类。

    每个后端需要实现这些方法。
    不支持的操作可以抛出 NotImplementedError 或返回 None，
    上层会自动 fallback 到 CPU 实现。
    """

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def get_device(self, device_id: int = 0) -> torch.device:
        pass

    @abstractmethod
    def device_count(self) -> int:
        pass

    @abstractmethod
    def device_name(self, device_id: int = 0) -> str:
        pass

    @abstractmethod
    def memory_info(self, device_id: int = 0) -> Tuple[int, int]:
        """返回 (已用, 总量) 字节"""
        pass

    @abstractmethod
    def synchronize(self):
        pass

    # ---- 算子接口 ----

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """矩阵乘法"""
        return torch.matmul(a, b)

    def layer_norm(self, x: torch.Tensor, normalized_shape, weight=None, bias=None, eps=1e-5):
        """Layer Normalization"""
        return F.layer_norm(x, normalized_shape, weight, bias, eps)

    def softmax(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Softmax"""
        return F.softmax(x, dim=dim)

    def attention(self, q, k, v, mask=None, dropout_p=0.0):
        """Scaled Dot-Product Attention"""
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)

    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
        """RMS Normalization (fallback 实现)"""
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        return x / rms * weight


# ============================================================
# 3. CUDA 后端实现
# ============================================================

class CUDABackend(ComputeBackend):
    """NVIDIA CUDA 后端"""

    def name(self) -> str:
        return "cuda"

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def get_device(self, device_id: int = 0) -> torch.device:
        return torch.device(f"cuda:{device_id}")

    def device_count(self) -> int:
        return torch.cuda.device_count()

    def device_name(self, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    def memory_info(self, device_id: int = 0) -> Tuple[int, int]:
        free, total = torch.cuda.mem_get_info(device_id)
        return (total - free, total)

    def synchronize(self):
        torch.cuda.synchronize()

    def attention(self, q, k, v, mask=None, dropout_p=0.0):
        """CUDA 上使用 FlashAttention (通过 SDPA)"""
        # PyTorch SDPA 自动选择最优后端
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=dropout_p
        )


# ============================================================
# 4. CPU 后端实现
# ============================================================

class CPUBackend(ComputeBackend):
    """CPU 后端 (fallback)"""

    def name(self) -> str:
        return "cpu"

    def is_available(self) -> bool:
        return True

    def get_device(self, device_id: int = 0) -> torch.device:
        return torch.device("cpu")

    def device_count(self) -> int:
        return 1

    def device_name(self, device_id: int = 0) -> str:
        import platform
        return platform.processor()

    def memory_info(self, device_id: int = 0) -> Tuple[int, int]:
        import psutil
        mem = psutil.virtual_memory()
        return (mem.used, mem.total)

    def synchronize(self):
        pass  # CPU 不需要同步


# ============================================================
# 5. 模拟的 NPU 后端（展示如何添加新后端）
# ============================================================

class MockNPUBackend(ComputeBackend):
    """
    模拟的华为昇腾 NPU 后端。

    在实际工程中，这里会调用 torch_npu 的 API。
    这里只是展示后端接口的设计。
    """

    def name(self) -> str:
        return "npu"

    def is_available(self) -> bool:
        try:
            import torch_npu
            return torch.npu.is_available()
        except ImportError:
            return False

    def get_device(self, device_id: int = 0) -> torch.device:
        return torch.device(f"npu:{device_id}")

    def device_count(self) -> int:
        try:
            import torch_npu
            return torch.npu.device_count()
        except ImportError:
            return 0

    def device_name(self, device_id: int = 0) -> str:
        return "Ascend 910B (mock)"

    def memory_info(self, device_id: int = 0) -> Tuple[int, int]:
        return (0, 64 * 1024**3)  # 模拟 64GB

    def synchronize(self):
        try:
            import torch_npu
            torch.npu.synchronize()
        except ImportError:
            pass

    def attention(self, q, k, v, mask=None, dropout_p=0.0):
        """
        NPU 上可能没有 FlashAttention，需要 fallback。
        实际中可能调用华为的 FlashAttention 适配。
        """
        # Fallback to manual attention
        scale = q.shape[-1] ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        if dropout_p > 0:
            attn = F.dropout(attn, p=dropout_p)
        return torch.matmul(attn, v)


# ============================================================
# 6. 高层封装: 设备无关的 API
# ============================================================

class DeviceAgnosticOps:
    """
    设备无关的算子封装。

    使用方法:
        ops = DeviceAgnosticOps()  # 自动检测后端
        # 或
        ops = DeviceAgnosticOps(backend="cuda")  # 指定后端

        result = ops.matmul(a, b)
        result = ops.attention(q, k, v)
    """

    def __init__(self, backend: Optional[str] = None):
        self.backend = BackendRegistry.get(backend)
        self.device = self.backend.get_device()
        print(f"  [Ops] 使用后端: {self.backend.name()}, 设备: {self.device}")

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """将张量移到当前设备"""
        return tensor.to(self.device)

    def matmul(self, a, b):
        return self.backend.matmul(a, b)

    def layer_norm(self, x, normalized_shape, weight=None, bias=None):
        return self.backend.layer_norm(x, normalized_shape, weight, bias)

    def softmax(self, x, dim=-1):
        return self.backend.softmax(x, dim)

    def attention(self, q, k, v, mask=None):
        return self.backend.attention(q, k, v, mask)

    def sync(self):
        self.backend.synchronize()


# ============================================================
# 7. 一致性测试框架
# ============================================================

def consistency_test(backends: List[str]):
    """测试不同后端输出的一致性"""
    print("\n" + "=" * 60)
    print("后端一致性测试")
    print("=" * 60)

    torch.manual_seed(42)

    # 在 CPU 上生成参考数据
    x = torch.randn(4, 8, 256)
    q = torch.randn(4, 8, 64, 32)
    k = torch.randn(4, 8, 64, 32)
    v = torch.randn(4, 8, 64, 32)
    weight = torch.ones(256)
    bias = torch.zeros(256)

    results = {}

    for backend_name in backends:
        try:
            ops = DeviceAgnosticOps(backend=backend_name)

            # 将数据移到目标设备
            x_dev = ops.to_device(x)
            q_dev = ops.to_device(q)
            k_dev = ops.to_device(k)
            v_dev = ops.to_device(v)
            w_dev = ops.to_device(weight)
            b_dev = ops.to_device(bias)

            # 执行各操作
            results[backend_name] = {
                'softmax': ops.softmax(x_dev).cpu(),
                'layer_norm': ops.layer_norm(x_dev, [256], w_dev, b_dev).cpu(),
                'matmul': ops.matmul(x_dev, x_dev.transpose(-1, -2)).cpu(),
            }
            ops.sync()
            print(f"  {backend_name}: 所有操作执行成功")

        except Exception as e:
            print(f"  {backend_name}: 跳过 ({e})")

    # 比较结果
    if len(results) >= 2:
        ref_name = list(results.keys())[0]
        ref = results[ref_name]

        print(f"\n  以 {ref_name} 为参考:")
        for name, res in results.items():
            if name == ref_name:
                continue
            for op in ['softmax', 'layer_norm', 'matmul']:
                diff = (ref[op] - res[op]).abs().max().item()
                status = "PASS" if diff < 1e-4 else "FAIL"
                print(f"    {name} vs {ref_name} | {op}: max_diff={diff:.2e} [{status}]")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Lab 09: 多后端算子适配层 Demo")
    print("=" * 60)

    # 注册后端
    print("\n--- 注册后端 ---")
    BackendRegistry.register("cpu", CPUBackend())
    if torch.cuda.is_available():
        BackendRegistry.register("cuda", CUDABackend())
    BackendRegistry.register("npu", MockNPUBackend())

    # 自动检测
    print(f"\n--- 自动检测 ---")
    auto_backend = BackendRegistry.auto_detect()
    print(f"  自动选择: {auto_backend}")
    print(f"  可用后端: {BackendRegistry.list_backends()}")

    # 使用设备无关 API
    print(f"\n--- 设备无关 API ---")
    ops = DeviceAgnosticOps()  # 自动选择最优后端

    x = ops.to_device(torch.randn(4, 8, 256))
    print(f"  输入: {x.shape} on {x.device}")

    result = ops.softmax(x)
    print(f"  Softmax 输出: {result.shape}")

    result = ops.layer_norm(x, [256])
    print(f"  LayerNorm 输出: {result.shape}")

    # 一致性测试
    available = [b for b in BackendRegistry.list_backends()
                 if BackendRegistry.get(b).is_available()]
    consistency_test(available)

    # 性能对比
    if torch.cuda.is_available():
        print(f"\n--- 性能对比 ---")
        x_cpu = torch.randn(32, 512, 4096)
        x_cuda = x_cpu.cuda()

        # CPU
        start = time.perf_counter()
        for _ in range(10):
            _ = F.softmax(x_cpu, dim=-1)
        cpu_ms = (time.perf_counter() - start) / 10 * 1000

        # CUDA
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            _ = F.softmax(x_cuda, dim=-1)
        torch.cuda.synchronize()
        cuda_ms = (time.perf_counter() - start) / 10 * 1000

        print(f"  Softmax (32, 512, 4096):")
        print(f"    CPU:  {cpu_ms:.2f} ms")
        print(f"    CUDA: {cuda_ms:.2f} ms")
        print(f"    加速: {cpu_ms/cuda_ms:.1f}x")

    print(f"\n" + "=" * 60)
    print("总结:")
    print("  1. 注册式设计让新后端接入不需要修改上层代码")
    print("  2. 一致性测试确保不同后端输出相同结果")
    print("  3. 实际工程中需要处理: 算子缺失、精度差异、性能差异")
    print("  4. PyTorch 的 torch.device 机制已经是很好的抽象")
    print("=" * 60)


if __name__ == "__main__":
    main()
