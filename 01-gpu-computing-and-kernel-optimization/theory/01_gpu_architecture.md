# 01 - GPU 硬件架构全解析

## 为什么要深入理解 GPU 架构？

作为 AI Infra 工程师，你的日常工作是让模型在 GPU 上跑得更快。但"更快"不是玄学——每一个优化决策都对应着硬件层面的具体原因。理解架构，才能做出正确的优化判断。

举个例子：为什么 batch size 要是 32 的倍数？为什么共享内存能加速？为什么 H20 的 FP32 算力不如 A100 但推理场景下性能不差？这些问题的答案都在架构里。

## GPU vs CPU：本质区别

### CPU 的设计哲学：低延迟

```
CPU Core 的面积分配：
┌─────────────────────────────────────┐
│  Control Logic (分支预测/乱序执行)    │  ~50%
│  Cache (L1/L2/L3)                   │  ~30%
│  ALU (实际计算单元)                   │  ~20%
└─────────────────────────────────────┘
```

CPU 把大量晶体管用于：
- **分支预测**：猜测下一条指令，减少 pipeline stall
- **乱序执行**：指令不按顺序执行，尽量填满计算单元
- **大容量缓存**：减少访问主存的延迟

目标：让**单个线程**跑得尽可能快。

### GPU 的设计哲学：高吞吐

```
GPU 的面积分配：
┌─────────────────────────────────────┐
│  ALU (大量计算单元)                   │  ~70%
│  Control Logic (简单的调度器)          │  ~10%
│  Cache/Shared Memory                │  ~20%
└─────────────────────────────────────┘
```

GPU 的策略完全不同：
- **大量简单核心**：不需要乱序执行，不需要复杂分支预测
- **用并行度掩盖延迟**：一个 warp 等待数据时，切换到另一个 warp
- **小缓存 + 高带宽内存**：靠吞吐量而非低延迟

目标：让**数千个线程**同时执行，总吞吐量最大化。

### 类比理解

| 维度 | CPU | GPU |
|------|-----|-----|
| 类比 | 几个博士生 | 几千个流水线工人 |
| 单任务能力 | 极强 | 一般 |
| 并行任务数 | 少（8-64 核） | 极多（数千核） |
| 适合场景 | 复杂逻辑、分支多 | 相同操作重复执行 |

## NVIDIA GPU 架构层次

### Streaming Multiprocessor (SM)

SM 是 GPU 的基本计算单元，类似 CPU 的一个核心（但更简单）。

```
一个 SM 的内部结构（以 Hopper/H20 为例）：
┌────────────────────────────────────────────┐
│  Warp Scheduler × 4                        │
│  ├── Dispatch Unit × 4                     │
│                                            │
│  FP32 CUDA Cores × 128                    │
│  FP64 CUDA Cores × 64                     │
│  Tensor Cores × 4 (4th Gen)               │
│  Load/Store Units × 32                     │
│  Special Function Units (SFU) × 16        │
│                                            │
│  Register File: 256 KB                     │
│  Shared Memory / L1 Cache: 228 KB         │
│  (可配置比例)                               │
└────────────────────────────────────────────┘
```

**关键理解**：SM 是资源分配的基本粒度。你的 kernel 占用多少个 SM，直接决定了并行度。

### Warp：执行的最小单位

一个 Warp = 32 个线程，它们**同时执行同一条指令**（SIMT: Single Instruction, Multiple Threads）。

```
Warp 执行模型：
时刻T:  Thread 0  Thread 1  Thread 2  ... Thread 31
         ADD R1   ADD R1    ADD R1    ... ADD R1      ← 同一条指令
时刻T+1: MUL R2   MUL R2    MUL R2    ... MUL R2
```

**Warp Divergence**（分支发散）：
```cuda
if (threadIdx.x < 16) {
    // 前 16 个线程执行这里
    do_something_A();
} else {
    // 后 16 个线程执行这里
    do_something_B();
}
// 实际执行：先执行 A（后16线程闲置），再执行 B（前16线程闲置）
// 总时间 = A + B，而非 max(A, B)
```

这就是为什么 GPU 代码要尽量避免分支——分支会让 warp 内的线程"串行化"。

### CUDA Core vs Tensor Core

**CUDA Core**：通用计算单元
- 每周期执行一次 FP32 加法或乘法
- 类似 CPU 的 ALU，但更简单

**Tensor Core**：专用矩阵计算单元（从 Volta 架构开始）
- 每周期执行一次 4×4 矩阵乘加（D = A × B + C）
- 第四代 Tensor Core（Hopper）支持 FP8/FP16/BF16/TF32/INT8
- 一次操作 = 64 次 FMA（Fused Multiply-Add），相比 CUDA Core 效率提升 ~16x

```
Tensor Core 单次操作：
D[4×4] = A[4×4] × B[4×4] + C[4×4]

CUDA Core 做同样的事需要：
4 × 4 × 4 = 64 次乘法 + 64 次加法 = 128 次操作
```

## 内存层次（Memory Hierarchy）

这是性能优化最关键的部分。GPU 计算快，但数据搬运慢。

```
                    容量        带宽          延迟
┌──────────┐
│ Registers │    256KB/SM    ~20 TB/s      1 cycle
├──────────┤
│ Shared Mem│    228KB/SM    ~19 TB/s      ~20 cycles
├──────────┤
│ L1 Cache  │   (与Shared共享)              ~30 cycles
├──────────┤
│ L2 Cache  │    60 MB       ~6 TB/s       ~200 cycles
├──────────┤
│ HBM (全局)│    96 GB       4.0 TB/s      ~400 cycles
├──────────┤
│ Host RAM  │    ~TB级       ~64 GB/s      ~10000 cycles (经PCIe)
└──────────┘
```

### 关键洞察

1. **寄存器最快但最少**：每个线程能用的寄存器有限（通常 255 个），用多了会 "register spill" 到 local memory（实际是全局内存），性能骤降。

2. **共享内存是程序员的武器**：它是唯一由程序员显式管理的高速缓存。经典优化 pattern：
   - 从 HBM 加载数据到 shared memory
   - 在 shared memory 中反复计算
   - 最后写回 HBM

3. **HBM 带宽是瓶颈**：4 TB/s 听起来大，但 Tensor Core 的算力需要更大的带宽喂数据。这就是为什么很多 AI 推理场景是"memory-bound"。

## H20 具体架构参数

NVIDIA H20 是针对中国市场的 Hopper 架构芯片（受出口管制影响的降配版本）：

| 参数 | H20 | H100 (对比) |
|------|-----|-------------|
| 架构 | Hopper (GH200) | Hopper (GH100) |
| SM 数量 | 78 | 132 |
| CUDA Cores | 9984 | 16896 |
| Tensor Cores | 312 (4th Gen) | 528 (4th Gen) |
| FP32 算力 | ~44 TFLOPS | ~67 TFLOPS |
| FP16 Tensor | ~148 TFLOPS | ~990 TFLOPS |
| 显存类型 | HBM3 | HBM3 |
| 显存容量 | 96 GB | 80 GB |
| 显存带宽 | 4.0 TB/s | 3.35 TB/s |
| L2 Cache | 60 MB | 50 MB |
| TDP | 400W | 700W |
| NVLink | 900 GB/s (总) | 900 GB/s (总) |

### H20 的独特定位

H20 是一个**高带宽、大显存**的芯片：
- 算力被削减（FP16 Tensor 约 H100 的 1/7）
- 但显存带宽反而更高（4.0 vs 3.35 TB/s）
- 显存容量更大（96 vs 80 GB）

**工程含义**：
- H20 适合**访存密集型**负载（如 LLM Decode 阶段）
- 不太适合计算密集型负载（如大规模训练的 Forward）
- 大显存适合部署大模型（96GB 能放下 70B 模型的半精度参数）

## NVLink vs PCIe

### PCIe Gen5
- 带宽：64 GB/s（双向）
- 连接：CPU ↔ GPU，GPU ↔ GPU（无 NVLink 时的 fallback）
- 特点：通用接口，延迟较高

### NVLink (4th Gen, Hopper)
- 带宽：900 GB/s（双向总带宽，18 links × 50 GB/s）
- 连接：GPU ↔ GPU
- 特点：专用高速互连，延迟低

```
8-GPU 服务器典型拓扑（NVSwitch 全互联）：
┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐
│GPU 0│────│GPU 1│────│GPU 2│────│GPU 3│
└──┬──┘     └──┬──┘     └──┬──┘     └──┬──┘
   │           │           │           │
   │      NVSwitch (全互联)              │
   │           │           │           │
┌──┴──┐     ┌──┴──┐     ┌──┴──┐     ┌──┴──┐
│GPU 4│────│GPU 5│────│GPU 6│────│GPU 7│
└─────┘     └─────┘     └─────┘     └─────┘
```

**工程含义**：
- Tensor Parallelism 要求高带宽通信 → 必须用 NVLink
- Pipeline Parallelism 通信量小 → PCIe 也够用
- AllReduce 在 NVLink 上几乎不是瓶颈（900 GB/s >> 模型梯度大小）

## GPU 执行流程概览

```
1. Host (CPU) 准备数据
2. 数据通过 PCIe 传输到 GPU HBM
3. GPU 启动 Kernel
4. SM 调度器将线程块分配到各 SM
5. 每个 SM 内：
   a. Warp 调度器选择就绪的 warp
   b. 从 HBM/L2/L1/Shared Memory 加载数据到寄存器
   c. CUDA/Tensor Core 执行计算
   d. 结果写回内存
6. 所有线程块完成 → Kernel 结束
7. 结果通过 PCIe 传回 Host（如需要）
```

## 本章要点总结

1. GPU 用**大量简单核心 + 高带宽内存**换取吞吐量，牺牲单线程延迟
2. **SM** 是资源分配单元，**Warp (32 threads)** 是执行单元
3. **Tensor Core** 是 AI 计算的主力（比 CUDA Core 快 ~16x 做矩阵乘）
4. **内存层次**决定了优化的上限——数据越靠近计算单元越好
5. **H20 的特点**：高带宽大显存，算力受限，适合推理
6. **NVLink** 是多卡并行的关键基础设施

## 延伸阅读

- [NVIDIA Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core)
- [CUDA C++ Programming Guide - Hardware Implementation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation)
- H20 Datasheet（联系 NVIDIA 或服务器厂商获取）
