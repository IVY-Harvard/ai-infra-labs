# 07 - 多后端适配：CUDA / ROCm / 国产芯片

## 为什么需要多后端适配？

作为 AI Infra 工程师，你面临的现实是：

1. **供应链风险**：出口管制使得 NVIDIA 高端 GPU 供应不稳定
2. **成本优化**：不同硬件有不同的性价比优势
3. **客户需求**：私有化部署可能要求支持国产芯片
4. **技术演进**：AMD、Intel 在 AI 芯片上持续追赶

你的代码不能只跑在 NVIDIA 上。

## 硬件生态全景

### NVIDIA CUDA 生态

```
硬件: A100, H100, H20, L40S, ...
软件栈:
├── CUDA Runtime / Driver
├── cuBLAS (矩阵运算)
├── cuDNN (深度学习原语)
├── NCCL (多卡通信)
├── TensorRT (推理优化)
├── Triton (GPU kernel)
└── cuFFT, cuSPARSE, ...

优势:
- 生态最成熟，文档最完善
- 几乎所有 AI 框架的第一优先级
- 性能工具链完善（NSight）

劣势:
- 闭源，vendor lock-in
- 受出口管制影响
- 价格昂贵
```

### AMD ROCm 生态

```
硬件: MI250X, MI300X, MI300A (APU), ...
软件栈:
├── HIP Runtime (CUDA 兼容层)
├── rocBLAS (矩阵运算)
├── MIOpen (深度学习原语)
├── RCCL (多卡通信, NCCL 兼容)
├── ROCm compiler (基于 LLVM)
└── Composable Kernel (CK)

HIP 与 CUDA 的关系:
- HIP 可以编译 CUDA 代码（大部分）
- hipify-perl / hipify-clang 自动转换工具
- API 1:1 映射（cudaMalloc → hipMalloc）

优势:
- CUDA 迁移成本低（HIP 兼容性好）
- MI300X 性能追上 H100
- 开源程度更高
- 不受出口管制

劣势:
- 生态成熟度不如 CUDA（约落后 2-3 年）
- 文档和社区支持较弱
- 部分算子性能差距明显
- Debug 工具不够完善
```

### 华为昇腾 (Ascend)

```
硬件: Atlas 300/800, 昇腾 910B, ...
软件栈:
├── CANN (Compute Architecture for Neural Networks)
│   ├── AscendCL (底层 API)
│   ├── GE (Graph Engine, 图编译器)
│   └── TBE (Tensor Boost Engine, 算子开发)
├── MindSpore (自研框架)
├── PyTorch Adapter (torch_npu)
└── HCCL (多卡通信)

架构特点:
- Da Vinci 核心: 向量单元 + Cube 单元(矩阵)
- 自研指令集，不兼容 CUDA
- AI Core 设计针对矩阵运算优化
- HBM2e 显存

优势:
- 国产自主可控
- 政策支持力度大
- 在特定场景性能不错

劣势:
- 生态封闭，文档有限
- 算子覆盖率不够（长尾算子缺失）
- 开发效率低（TBE 学习曲线陡峭）
- 性能调优工具不够成熟
- 框架支持有限（主要是 MindSpore + torch_npu）
```

### 寒武纪 (Cambricon)

```
硬件: MLU370, MLU590, ...
软件栈:
├── BANG (寒武纪编程语言, 类CUDA)
├── CNToolkit
├── CNNL (算子库)
├── CNCL (通信库)
├── MagicMind (推理引擎)
└── PyTorch Adapter (torch_mlu)

特点:
- BANG 语言和 CUDA 有相似之处
- 算子性能参差不齐
- 推理场景相对成熟
```

## 适配层设计思路

### 设计原则

```
1. 接口统一: 上层代码不感知底层硬件
2. 最小抽象: 不要过度抽象导致性能损失
3. 可扩展: 新硬件接入不需要改上层代码
4. 优雅降级: 某后端不支持某功能时有 fallback
```

### 分层架构

```
┌────────────────────────────────────────────┐
│          Application Layer                  │
│     (推理服务, 训练框架, benchmark)          │
├────────────────────────────────────────────┤
│         Backend Abstraction Layer           │
│  ┌──────────────────────────────────────┐  │
│  │  Device / Stream / Memory Management  │  │
│  │  Tensor Operations Interface          │  │
│  │  Communication Interface              │  │
│  └──────────────────────────────────────┘  │
├─────────┬──────────┬──────────┬───────────┤
│  CUDA   │   ROCm   │  Ascend  │ Cambricon │
│ Backend │ Backend  │ Backend  │  Backend  │
├─────────┼──────────┼──────────┼───────────┤
│ cuBLAS  │ rocBLAS  │  CANN    │   CNNL    │
│ cuDNN   │ MIOpen   │  torch   │   torch   │
│ NCCL    │ RCCL     │  _npu    │   _mlu    │
└─────────┴──────────┴──────────┴───────────┘
```

### 核心接口设计

```python
from abc import ABC, abstractmethod
from typing import Optional, Tuple

class DeviceBackend(ABC):
    """设备后端抽象基类"""
    
    @abstractmethod
    def get_device_count(self) -> int:
        """获取可用设备数量"""
        pass
    
    @abstractmethod
    def get_device_name(self, device_id: int) -> str:
        """获取设备名称"""
        pass
    
    @abstractmethod
    def get_device_memory(self, device_id: int) -> Tuple[int, int]:
        """获取设备内存 (已用, 总量) 单位 bytes"""
        pass
    
    @abstractmethod
    def set_device(self, device_id: int) -> None:
        """设置当前设备"""
        pass
    
    @abstractmethod
    def synchronize(self, device_id: Optional[int] = None) -> None:
        """设备同步"""
        pass
    
    @abstractmethod
    def allocate(self, size: int) -> int:
        """分配设备内存，返回指针"""
        pass
    
    @abstractmethod
    def free(self, ptr: int) -> None:
        """释放设备内存"""
        pass


class TensorOpsBackend(ABC):
    """张量运算后端抽象基类"""
    
    @abstractmethod
    def matmul(self, a, b, dtype=None) -> 'Tensor':
        """矩阵乘法"""
        pass
    
    @abstractmethod
    def softmax(self, x, dim=-1) -> 'Tensor':
        """Softmax"""
        pass
    
    @abstractmethod
    def layer_norm(self, x, normalized_shape, weight=None, bias=None, eps=1e-5) -> 'Tensor':
        """Layer Normalization"""
        pass
    
    @abstractmethod
    def attention(self, q, k, v, mask=None, dropout_p=0.0) -> 'Tensor':
        """Scaled Dot-Product Attention"""
        pass


class CommBackend(ABC):
    """通信后端抽象基类"""
    
    @abstractmethod
    def all_reduce(self, tensor, op='sum') -> None:
        """AllReduce (in-place)"""
        pass
    
    @abstractmethod
    def all_gather(self, tensor_list, tensor) -> None:
        """AllGather"""
        pass
    
    @abstractmethod
    def broadcast(self, tensor, src=0) -> None:
        """Broadcast"""
        pass
```

### PyTorch 统一适配方式

PyTorch 已经有了一定程度的后端抽象：

```python
import torch

# CUDA
device = torch.device("cuda:0")
x = torch.randn(1024, 1024, device=device)

# ROCm (通过 HIP, 同样使用 "cuda" )
device = torch.device("cuda:0")  # ROCm 编译的 PyTorch 也用 "cuda"

# 昇腾
import torch_npu
device = torch.device("npu:0")
x = torch.randn(1024, 1024, device=device)

# 寒武纪
import torch_mlu
device = torch.device("mlu:0")
x = torch.randn(1024, 1024, device=device)
```

### 工程实践：设备无关代码

```python
import os
import torch

def get_backend():
    """根据环境自动检测后端"""
    backend = os.environ.get("AI_BACKEND", "auto")
    
    if backend == "auto":
        if torch.cuda.is_available():
            return "cuda"
        try:
            import torch_npu
            if torch.npu.is_available():
                return "npu"
        except ImportError:
            pass
        try:
            import torch_mlu
            if torch.mlu.is_available():
                return "mlu"
        except ImportError:
            pass
        return "cpu"
    
    return backend


def get_device(backend=None, device_id=0):
    """获取设备对象"""
    if backend is None:
        backend = get_backend()
    return torch.device(f"{backend}:{device_id}")


def get_device_module(backend=None):
    """获取设备对应的模块 (如 torch.cuda)"""
    if backend is None:
        backend = get_backend()
    
    module_map = {
        "cuda": torch.cuda,
        "npu": None,   # 需要 import torch_npu 后使用 torch.npu
        "mlu": None,   # 需要 import torch_mlu 后使用 torch.mlu
        "cpu": None,
    }
    
    if backend == "npu":
        import torch_npu
        return torch.npu
    elif backend == "mlu":
        import torch_mlu
        return torch.mlu
    
    return module_map.get(backend)
```

## 适配中的常见坑

### 1. 算子缺失

```
问题: 国产芯片的算子库不如 CUDA 完善

常见缺失算子:
- Flash Attention (通常需要自己适配)
- Group Query Attention
- 特定量化算子 (W4A16, FP8)
- 自定义 CUDA extension

解决策略:
1. Fallback 到 Python 实现（性能差但能跑）
2. 用基础算子组合实现（性能中等）
3. 在目标平台上重新实现（性能最好但开发成本高）
```

### 2. 精度差异

```
问题: 不同硬件的浮点实现有微小差异

示例:
- NVIDIA: FP16 计算严格符合 IEEE 754
- 昇腾: 某些算子的 FP16 精度可能略有差异
- 影响: 同样的模型在不同硬件上输出不同

应对:
1. 设定合理的精度容忍度（rtol=1e-3, atol=1e-3）
2. 关键算子（如 Attention）做精度对齐验证
3. 如果精度差异大，考虑用 FP32 fallback
```

### 3. 性能差异

```
问题: 算子在不同硬件上的最优实现策略不同

示例: GEMM
- CUDA: 用 cuBLAS + Tensor Core, tile size 128×128
- ROCm: 用 rocBLAS, 最优 tile size 可能不同
- 昇腾: 用 Cube 单元, 完全不同的调度策略

应对:
1. 每个后端独立做性能调优
2. 不要假设一个平台的最优策略适用于另一个
3. 建立 per-backend 的性能基准
```

### 4. 通信库差异

```
NCCL / RCCL / HCCL 的差异:

功能支持:
├── AllReduce: 全部支持
├── AllGather: 全部支持
├── ReduceScatter: 大部分支持
├── Send/Recv (P2P): NCCL 支持好，其他可能有限制
└── Asymmetric ops: 可能有差异

性能特点:
├── NCCL: 深度优化，NVLink 利用率高
├── RCCL: 与 NCCL 接口兼容，Infinity Fabric 优化
└── HCCL: 昇腾专用，HCCS 互联优化
```

## 跨平台 Kernel 开发

### 方案 1：HIP 转换（CUDA → AMD）

```bash
# 使用 hipify 工具自动转换 CUDA 代码
hipify-perl cuda_kernel.cu > hip_kernel.cpp

# 常见映射:
# cudaMalloc → hipMalloc
# cudaMemcpy → hipMemcpy
# __global__ → __global__ (不变!)
# __shared__ → __shared__ (不变!)
# atomicAdd → atomicAdd (不变!)

# 不能自动转换的:
# - Tensor Core (WMMA → 需要用 AMD 的 MFMA)
# - Inline PTX → 需要用 AMD 的 GCN assembly
# - CUDA 特有的 API (如 cooperative groups 的部分功能)
```

### 方案 2：Triton 跨平台

```python
# Triton 正在支持多后端
# 同一份 Triton 代码可以编译到:
# - NVIDIA GPU (PTX/CUBIN)
# - AMD GPU (AMDGCN) ← 实验性支持
# - CPU (LLVM) ← 开发中

@triton.jit
def vector_add(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)

# 理论上这份代码在 NVIDIA 和 AMD 上都能跑
# 但实际上 Triton 对 AMD 的支持还在完善中
```

### 方案 3：TVM/MLIR 路线

```python
# TVM 可以为不同后端生成优化代码
import tvm
from tvm import relay

# 从 PyTorch 导入模型
model = torch.jit.trace(model, example_input)
mod, params = relay.frontend.from_pytorch(model, [("input", input_shape)])

# 编译到不同目标
# CUDA
target = tvm.target.cuda()
lib_cuda = relay.build(mod, target, params=params)

# LLVM (CPU)
target = tvm.target.Target("llvm")
lib_cpu = relay.build(mod, target, params=params)

# 昇腾 (需要 TVM 支持)
# target = tvm.target.Target("ascend")
```

## 工程建议

### 短期策略（1-3 个月）

```
1. 以 CUDA/PyTorch 为主线开发
2. 使用 PyTorch 原生 API（避免直接调用 cuda 特有函数）
3. 将硬件相关代码隔离到独立模块
4. 建立 CI 中的跨平台测试
```

### 中期策略（3-6 个月）

```
1. 设计并实现适配层
2. 在目标平台上做性能基准测试
3. 针对性优化性能差距大的算子
4. 建立跨平台的性能回归测试
```

### 长期策略（6-12 个月）

```
1. 关注 Triton 的多后端进展
2. 评估 TVM/MLIR 路线
3. 建立统一的算子注册和调度机制
4. 维护各平台的性能 parity dashboard
```

## 本章要点总结

1. **多后端适配**是现实需求：供应链安全、成本优化、客户要求
2. **ROCm/HIP** 是 CUDA 迁移成本最低的选择（API 兼容）
3. **国产芯片**生态不成熟，需要更多适配工作（算子缺失、精度差异）
4. 设计适配层时遵循**接口统一、最小抽象、可扩展、优雅降级**原则
5. **PyTorch** 已有一定的后端抽象能力（torch_npu, torch_mlu）
6. **Triton** 的多后端是未来方向，但目前非 NVIDIA 支持还不成熟
7. 不要假设一个平台的优化策略适用于另一个——**每个平台需要独立调优**

## 延伸阅读

- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [HIP Porting Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/user_guide/hip_porting_guide.html)
- [华为昇腾 CANN 文档](https://www.hiascend.com/document)
- [PyTorch PrivateUse1 Backend](https://pytorch.org/tutorials/advanced/privateuseone.html)
- [Triton Multi-Backend RFC](https://github.com/openai/triton/issues)
