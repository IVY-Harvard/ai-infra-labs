# 03 - 计算密集 vs 访存密集：如何判断瓶颈在哪？

## 为什么这个问题如此重要？

你写了一个 GPU kernel，发现它比预期慢 10 倍。怎么办？

如果不知道瓶颈在哪，你可能：
- 花三天优化了计算逻辑，结果发现瓶颈是内存带宽
- 费力做了内存优化，结果发现 Tensor Core 利用率才 5%

**判断瓶颈类型是所有优化的第一步。** 方向错了，一切努力白费。

## 基本概念

### 计算密集（Compute-bound）

硬件的**计算单元**是瓶颈。数据供应充足，但算不过来。

特征：
- GPU 算力利用率高（>70%）
- 内存带宽利用率低（<50%）
- 增加算力（更多 SM、更高频率）能提升性能
- 增加内存带宽没有帮助

典型场景：大矩阵乘法、卷积（大 batch）

### 访存密集（Memory-bound）

硬件的**内存带宽**是瓶颈。计算单元在等数据，吃不饱。

特征：
- GPU 算力利用率低（<30%）
- 内存带宽接近峰值
- 增加内存带宽能提升性能
- 增加算力没有帮助

典型场景：元素级操作（ReLU, LayerNorm）、小 batch 的矩阵乘、LLM Decode

## Arithmetic Intensity（算术强度）

### 定义

```
Arithmetic Intensity (AI) = FLOPs / Bytes
                          = 计算量 / 数据搬运量
单位：FLOP/Byte
```

这个指标衡量"每搬运一个字节的数据，做了多少次运算"。

### 关键阈值

```
机器平衡点 = 峰值算力 / 峰值带宽

H20 (FP16 Tensor Core):
  机器平衡点 = 148 TFLOPS / 4.0 TB/s = 37 FLOP/Byte

含义：
- 如果你的算子 AI > 37 → 计算密集（H20 上）
- 如果你的算子 AI < 37 → 访存密集（H20 上）
```

### 常见算子的 Arithmetic Intensity

| 算子 | AI (FLOP/Byte) | 类型 |
|------|----------------|------|
| Vector Add | 0.25 | 极度访存密集 |
| ReLU | 0.125 | 极度访存密集 |
| LayerNorm | ~1 | 访存密集 |
| Softmax | ~2 | 访存密集 |
| GEMM (M=N=K=4096, FP16) | ~2048 | 极度计算密集 |
| GEMM (M=1, N=K=4096, FP16) | ~2 | 访存密集！ |
| Attention (long seq) | ~100+ | 计算密集 |
| Attention (short seq, decode) | ~1-5 | 访存密集 |

### 计算示例：矩阵乘法

```
GEMM: C[M×N] = A[M×K] × B[K×N]

计算量 = 2 × M × N × K FLOPs（乘+加）

数据量 = (M×K + K×N + M×N) × bytes_per_element
       ≈ (M×K + K×N) × bytes  （忽略输出，因为通常更小）

对于 M=N=K=4096, FP16:
  FLOPs = 2 × 4096³ = 137 GFLOPs
  Bytes = (4096² + 4096² + 4096²) × 2 = 96 MB
  AI = 137G / 96M ≈ 1428 FLOP/Byte → 极度计算密集

对于 M=1, N=K=4096, FP16 (decode场景):
  FLOPs = 2 × 1 × 4096 × 4096 = 33.5 MFLOPs
  Bytes = (1×4096 + 4096×4096) × 2 ≈ 33.6 MB
  AI = 33.5M / 33.6M ≈ 1 FLOP/Byte → 访存密集！
```

**关键洞察**：同样是矩阵乘法，M（batch维度）的大小决定了它是计算密集还是访存密集。这直接解释了 LLM 的 Prefill vs Decode 性能差异。

## Roofline 模型

### 什么是 Roofline？

Roofline 是一个可视化工具，帮助你一眼看出 kernel 的瓶颈。

```
性能                      计算密集区域
(TFLOPS)                    /
    ^                      / ← 算力上限（屋顶）
    |                     /
    |                    /
    |                   /
    |                  /
    |    访存密集区域  /
    |               /
    |              / ← 斜率 = 内存带宽
    |             /
    |            /
    |           /
    |──────────/─────────────────── → Arithmetic Intensity
              ^                       (FLOP/Byte)
           机器平衡点
```

### 解读方法

```
对于 H20 (FP16 Tensor):
峰值算力 = 148 TFLOPS
HBM 带宽 = 4.0 TB/s
平衡点 = 148 / 4.0 = 37 FLOP/Byte

Roofline:
性能上限 = min(峰值算力, AI × 带宽)
         = min(148 TFLOPS, AI × 4.0 TB/s)
```

### 实际应用

1. **计算 kernel 的 AI**
2. **在 Roofline 图上标记这个点**
3. **看它离哪条线（屋顶/斜率）更近**
4. **选择优化方向**

```
如果在斜率部分（访存密集）：
├── 减少内存访问（算子融合、共享内存缓存）
├── 提高内存访问效率（coalescing、向量化读写）
└── 考虑用更快的内存（shared memory 替代 global）

如果在屋顶部分（计算密集）：
├── 使用 Tensor Core（WMMA/cuBLAS）
├── 降低精度（FP32 → FP16 → INT8）
└── 优化计算逻辑（减少冗余计算）
```

## AI 推理中的应用：Prefill vs Decode

### LLM 推理的两个阶段

**Prefill（预填充）**：
- 输入：整个 prompt（例如 512 个 token）
- 操作：一次性处理所有 token
- 矩阵乘：A[512×hidden] × B[hidden×hidden]
- **计算密集**：大 batch 的 GEMM

**Decode（生成）**：
- 输入：只有 1 个新 token
- 操作：逐 token 生成
- 矩阵乘：A[1×hidden] × B[hidden×hidden]
- **访存密集**：batch=1 的 GEMV

```
Prefill:
┌─────────┐
│ Prompt   │ → [一次大矩阵乘] → 所有 KV Cache
│ (512 tok)│    AI 很高，计算密集
└─────────┘

Decode:
┌───────┐
│1 token│ → [矩阵向量乘] → 1 个新 token → [矩阵向量乘] → ...
└───────┘    AI 很低，访存密集
```

### 对 H20 的工程含义

```
H20: 算力低但带宽高
├── Prefill 慢（算力不够）
│   └── 缓解：增大 batch（多 request 合并 prefill）
├── Decode 相对不差（带宽足够）
│   └── 优势：高带宽 + 大显存 = 适合长序列推理
└── 策略：
    ├── Continuous Batching（持续合批）
    ├── 分离 Prefill 和 Decode 到不同集群
    └── 用 INT8/FP8 量化降低带宽需求
```

## 实际判断流程

### 方法 1：理论分析

```python
def analyze_bottleneck(flops, bytes_transferred, peak_flops, peak_bandwidth):
    """判断一个 kernel 是计算密集还是访存密集"""
    ai = flops / bytes_transferred
    machine_balance = peak_flops / peak_bandwidth
    
    if ai > machine_balance:
        print(f"计算密集 (AI={ai:.1f} > 平衡点={machine_balance:.1f})")
        efficiency = (flops / peak_flops) * 100
        print(f"算力利用率: {efficiency:.1f}%")
    else:
        print(f"访存密集 (AI={ai:.1f} < 平衡点={machine_balance:.1f})")
        efficiency = (bytes_transferred / peak_bandwidth) * 100
        print(f"带宽利用率: {efficiency:.1f}%")
```

### 方法 2：Profiler 实测

用 NVIDIA NSight Compute：
1. `ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./your_kernel`
2. 看两个指标：
   - `sm__throughput`：计算吞吐利用率
   - `dram__throughput`：显存带宽利用率
3. 哪个更接近 100%，就是瓶颈

### 方法 3：简单经验法则

```
如果你的 kernel 主要在做：
├── 大 GEMM (M,N,K > 512)        → 计算密集
├── 小 GEMM (M < 32)              → 访存密集
├── 元素级操作 (add, mul, relu)    → 访存密集
├── Reduction (sum, max, softmax)  → 访存密集
├── Attention (长序列, 大batch)    → 计算密集
├── Attention (短序列/decode)      → 访存密集
└── 卷积 (大 batch, 大 channel)    → 计算密集
```

## 优化策略总结

### 访存密集型优化

| 策略 | 原理 | 典型收益 |
|------|------|----------|
| 算子融合 | 减少中间结果读写 HBM | 2-5x |
| 共享内存缓存 | 复用数据，减少 HBM 访问 | 2-10x |
| 向量化读写 | float4 一次读16字节 vs float 读4字节 | 1.5-2x |
| 数据类型降精度 | FP32→FP16 带宽需求减半 | ~2x |
| Coalescing | 减少内存事务数 | 2-10x |

### 计算密集型优化

| 策略 | 原理 | 典型收益 |
|------|------|----------|
| Tensor Core | 专用矩阵乘硬件 | 8-16x |
| 混合精度 | FP16 算力是 FP32 的 2-8x | 2-8x |
| 算法优化 | 减少 FLOPs（如 Strassen、FFT） | 1.5-3x |
| 流水线化 | 计算和访存重叠 | 1.5-2x |

## 本章要点总结

1. **Arithmetic Intensity** = FLOPs/Bytes 是判断瓶颈的关键指标
2. **Roofline 模型**可视化地展示瓶颈位置
3. **H20 平衡点** ≈ 37 FLOP/Byte（FP16 Tensor），低于此为访存密集
4. **LLM Prefill 是计算密集，Decode 是访存密集**——这是分离式架构的理论基础
5. **先判断瓶颈，再选择优化方向**——这是性能优化的第一原则
6. H20 的**高带宽+大显存**使其更适合 Decode 而非 Prefill

## 延伸阅读

- [Roofline: An Insightful Visual Performance Model](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html)
- [Understanding the Roofline Model (NERSC)](https://docs.nersc.gov/tools/performance/roofline/)
- [LLM Inference Performance Engineering (Databricks)](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
