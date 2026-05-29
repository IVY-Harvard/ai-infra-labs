# 02 - CUDA 编程模型

## 为什么要理解 CUDA 编程模型？

即使你日常用 PyTorch/Triton，不直接写 CUDA，理解 CUDA 编程模型仍然至关重要：
1. **性能分析**：NSight 等工具的输出全是 CUDA 术语（occupancy, warps, blocks）
2. **优化决策**：选择 block size、grid size 需要理解硬件映射
3. **Debug**：CUDA 相关错误（illegal memory access, launch failure）需要理解执行模型
4. **技术选型**：判断某个算子是否值得手写 kernel

## 核心抽象：Grid → Block → Thread

CUDA 用三层抽象组织并行计算：

```
Grid (一个 Kernel 启动)
├── Block (0,0)
│   ├── Thread (0,0)
│   ├── Thread (1,0)
│   └── ...
├── Block (1,0)
│   ├── Thread (0,0)
│   └── ...
├── Block (0,1)
└── ...
```

### 维度

三层抽象都支持 1D/2D/3D 索引：

```cuda
// 1D Grid, 1D Block
kernel<<<numBlocks, threadsPerBlock>>>(...)

// 2D Grid, 2D Block
dim3 grid(gridX, gridY);
dim3 block(blockX, blockY);
kernel<<<grid, block>>>(...)

// 3D 同理
dim3 grid(gx, gy, gz);
dim3 block(bx, by, bz);
```

### 索引计算

```cuda
// 1D 情况下计算全局线程 ID
int tid = blockIdx.x * blockDim.x + threadIdx.x;

// 2D 情况
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int global_id = row * width + col;
```

### 为什么是这样的层次结构？

这不是随意设计，而是对应硬件：

| 抽象层 | 硬件映射 | 资源共享 |
|--------|----------|----------|
| Grid | 整个 GPU | 全局内存 |
| Block | 一个 SM | 共享内存、L1 Cache |
| Warp (32 threads) | SM 内的执行单元 | 寄存器、指令流 |
| Thread | CUDA Core | 寄存器 |

**关键约束**：
- 同一个 Block 内的线程可以同步（`__syncthreads()`）
- 不同 Block 的线程**不能**直接同步（除了结束 kernel）
- 一个 Block 的所有线程必须在同一个 SM 上执行

## 执行模型详解

### Kernel 启动流程

```
1. CPU 调用 kernel<<<grid, block>>>(args)
2. CUDA Runtime 将 kernel 放入 GPU 的工作队列
3. GPU 的 GigaThread Engine 接收任务
4. 将 Block 分配到空闲的 SM
5. SM 将 Block 中的线程划分为 Warp（每32个一组）
6. Warp Scheduler 调度 Warp 执行
```

### Warp 调度

每个 SM 有 4 个 Warp Scheduler，每个周期可以发射一条指令：

```
时间 →
Warp 0: [执行][执行][等内存...........][执行][执行]
Warp 1: [等内存.][执行][执行][等内存...........][执行]
Warp 2: [执行][等内存...........][执行][执行][执行]
Warp 3: [执行][执行][执行][等内存...........][执行]
         ↑ 任何时刻都有 warp 在执行，掩盖了内存延迟
```

**延迟隐藏（Latency Hiding）**：这是 GPU 的核心设计思想。不像 CPU 用缓存减少延迟，GPU 用大量线程**绕过**延迟——当一个 warp 等数据时，切换到另一个 warp 继续算。

### Occupancy（占用率）

Occupancy = 实际活跃 Warp 数 / SM 支持的最大 Warp 数

影响 Occupancy 的因素：
1. **每个线程用的寄存器数**：寄存器总量有限，线程用得多，能同时驻留的线程就少
2. **每个 Block 用的共享内存**：共享内存有限，Block 用得多，能同时放的 Block 就少
3. **Block 大小**：Block 太大或太小都可能浪费

```
示例：H20 的一个 SM
- 最大 Warp 数：48（= 1536 threads）
- 寄存器文件：65536 个 32-bit 寄存器
- 共享内存：最多 228 KB

如果你的 kernel 每线程用 128 个寄存器：
65536 / 128 = 512 个线程 = 16 个 Warp
Occupancy = 16 / 48 = 33%

如果每线程用 32 个寄存器：
65536 / 32 = 2048，但受限于 1536
实际 = 1536 线程 = 48 Warp
Occupancy = 48 / 48 = 100%
```

**注意**：Occupancy 100% 不等于最优性能！有时降低 occupancy 换取更多寄存器（减少 spill）反而更快。

## 内存模型

### 各种内存类型

```cuda
// 全局内存（Global Memory）- 所有线程可访问，容量大但慢
__global__ void kernel() {
    float* data;  // 指向全局内存
}

// 共享内存（Shared Memory）- Block 内线程共享，快
__shared__ float smem[256];

// 寄存器（Registers）- 线程私有，最快
float local_var = 0.0f;  // 编译器分配到寄存器

// 常量内存（Constant Memory）- 只读，有缓存
__constant__ float params[64];

// 纹理内存（Texture Memory）- 只读，2D局部性优化
// 在 AI 中较少使用
```

### 全局内存访问模式：Coalescing

GPU 以 128 字节（32 个 float）为单位访问全局内存。Warp 的 32 个线程同时访问内存时：

```
Good: Coalesced Access（合并访问）
Thread 0 → addr[0]
Thread 1 → addr[1]
Thread 2 → addr[2]
...
Thread 31 → addr[31]
→ 1 次内存事务，128 bytes

Bad: Strided Access（跨步访问）
Thread 0 → addr[0]
Thread 1 → addr[32]
Thread 2 → addr[64]
...
→ 32 次内存事务！带宽浪费 32x
```

**工程含义**：矩阵按行存储时，让同一 warp 的线程访问同一行的连续元素。

## 同步机制

### Block 内同步

```cuda
__global__ void kernel() {
    __shared__ float smem[256];
    
    // 阶段 1：所有线程写入共享内存
    smem[threadIdx.x] = input[globalIdx];
    
    // 屏障：等待所有线程完成写入
    __syncthreads();
    
    // 阶段 2：所有线程读取其他线程写入的数据
    float val = smem[(threadIdx.x + 1) % 256];
}
```

`__syncthreads()` 确保 Block 内所有线程都到达这个点后才继续。

**危险**：如果 `__syncthreads()` 在条件分支里，且不是所有线程都能到达，会导致**死锁**。

### Warp 内同步

Warp 内的线程天然同步（SIMT），但从 Volta 架构开始引入了 Independent Thread Scheduling，需要显式同步：

```cuda
// Warp-level primitives
__syncwarp();                          // Warp 内屏障
int val = __shfl_sync(mask, var, src); // Warp 内数据交换
int vote = __ballot_sync(mask, pred);  // Warp 投票
```

### 原子操作

```cuda
// 当多个线程要更新同一地址时
atomicAdd(&global_counter, 1);      // 原子加
atomicMax(&global_max, local_val);  // 原子取最大
atomicCAS(&addr, expected, desired); // Compare-And-Swap
```

原子操作保证正确性但有性能代价，尽量减少使用。

## Kernel 设计最佳实践

### 1. Block Size 选择

```
推荐：128 或 256 个线程/Block
原因：
- 必须是 32 的倍数（Warp 大小）
- 太小（32/64）→ Occupancy 低
- 太大（1024）→ 寄存器/共享内存压力大
- 128/256 通常是最佳平衡点
```

### 2. Grid Size 选择

```
一般原则：Grid 要"覆盖"整个数据
grid_size = (N + block_size - 1) / block_size  // 向上取整

高级：考虑 SM 数量
// 确保 Grid 至少有足够的 Block 填满所有 SM
// H20 有 78 个 SM，每个 SM 至少需要 2-4 个 Block
// 所以 Grid 至少 156-312 个 Block
```

### 3. 避免 Warp Divergence

```cuda
// Bad: 分支导致 warp 内线程发散
if (threadIdx.x % 2 == 0) { ... } else { ... }

// Better: 让整个 warp 走同一分支
if (threadIdx.x / 32 < some_threshold) { ... }
// 这样每个 warp 要么全进 if，要么全进 else
```

### 4. 内存访问优化

```cuda
// Bad: AoS (Array of Structures)
struct Particle { float x, y, z, w; };
Particle particles[N];
// 访问所有 x 时：跨步 16 字节

// Good: SoA (Structure of Arrays)
float x[N], y[N], z[N], w[N];
// 访问所有 x 时：连续内存，完美 coalescing
```

## CUDA Stream 和异步执行

### 什么是 Stream？

Stream 是 GPU 上的命令队列。同一个 stream 内的操作按顺序执行，不同 stream 可以并行。

```cuda
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// 这两个操作可以重叠执行
kernel_A<<<grid, block, 0, stream1>>>(data_A);
kernel_B<<<grid, block, 0, stream2>>>(data_B);

// 经典 pattern：计算和传输重叠
for (int i = 0; i < N; i++) {
    cudaMemcpyAsync(d_in, h_in+i*chunk, size, H2D, stream[i%2]);
    kernel<<<grid, block, 0, stream[i%2]>>>(d_in, d_out);
    cudaMemcpyAsync(h_out+i*chunk, d_out, size, D2H, stream[i%2]);
}
```

### 为什么重要？

在推理服务中，Stream 用于：
- 重叠 Prefill 和 Decode（continuous batching）
- 重叠通信和计算（流水线并行）
- 多 request 并行处理

## 错误处理

```cuda
// 每个 CUDA API 调用都应该检查错误
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// Kernel 启动后检查
kernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());      // 检查启动错误
CUDA_CHECK(cudaDeviceSynchronize()); // 等待完成并检查执行错误
```

## 本章要点总结

1. **Grid/Block/Thread** 三层结构对应 GPU/SM/Core 的硬件映射
2. **Warp（32 线程）** 是实际执行的最小调度单位，避免 divergence
3. **Occupancy** 决定延迟隐藏能力，但不是越高越好
4. **Coalesced Access** 是内存优化的第一原则
5. **`__syncthreads()`** 只能 Block 内同步，Block 间无法同步
6. **Stream** 实现异步和重叠执行

## 延伸阅读

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- Professional CUDA C Programming (book by John Cheng)
