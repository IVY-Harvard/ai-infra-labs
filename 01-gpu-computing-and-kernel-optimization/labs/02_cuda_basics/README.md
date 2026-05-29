# Lab 02: CUDA 基础 — 向量加法

## 实验目的

通过实现最简单的 CUDA kernel（向量加法），掌握 CUDA 编程的基本流程：
1. 内存分配（Host + Device）
2. 数据传输（H2D + D2H）
3. Kernel 启动配置（Grid/Block）
4. 结果验证

## 前置要求

- NVIDIA CUDA Toolkit（`nvcc` 编译器）
- 已读 theory/02（CUDA 编程模型）

## 编译和运行

```bash
make          # 编译
./vector_add  # 运行
make clean    # 清理
```

## 实验步骤

1. 阅读 `vector_add.cu` 中的注释
2. 编译运行，观察输出
3. 修改 `BLOCK_SIZE`（32, 64, 128, 256, 512, 1024），观察性能变化
4. 修改 `N`（数据量），观察 kernel 时间如何变化
5. 思考：向量加法是计算密集还是访存密集？

## 关键概念

- `__global__`：声明 GPU kernel 函数
- `<<<grid, block>>>`：kernel 启动配置
- `cudaMalloc / cudaMemcpy / cudaFree`：设备内存管理
- `blockIdx.x * blockDim.x + threadIdx.x`：全局线程 ID 计算

## 预期结果

- 正确性验证通过
- 不同 BLOCK_SIZE 的性能差异不大（因为这个 kernel 是 memory-bound）
- 随着 N 增大，kernel 时间线性增长
