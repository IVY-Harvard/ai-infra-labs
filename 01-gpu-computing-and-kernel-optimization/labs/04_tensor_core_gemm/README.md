# Lab 04: Tensor Core GEMM 性能对比

## 实验目的

对比三种矩阵乘法实现的性能：
1. **朴素 GEMM**：纯 CUDA Core，每线程计算一个输出元素
2. **WMMA GEMM**：使用 Tensor Core 的 WMMA API
3. **cuBLAS GEMM**：NVIDIA 官方优化库

通过对比理解：
- Tensor Core 相比 CUDA Core 快多少
- 手写 kernel 与厂商库的差距
- 不同矩阵大小对性能的影响

## 前置要求

- CUDA Toolkit（含 cuBLAS）
- 已读 theory/04（Tensor Core 部分）

## 编译和运行

```bash
nvcc -O2 -arch=sm_90 -lcublas -o naive_gemm naive_gemm.cu
nvcc -O2 -arch=sm_90 -lcublas -o wmma_gemm wmma_gemm.cu
nvcc -O2 -arch=sm_90 -lcublas -o cublas_gemm cublas_gemm.cu
```

## 预期结果

在 H20 上大致性能排序：
- 朴素 GEMM: ~5-10 TFLOPS (FP32)
- WMMA GEMM: ~20-50 TFLOPS (FP16, 未充分优化)
- cuBLAS: ~140+ TFLOPS (FP16, 接近理论峰值)

差距来源：
- cuBLAS 有多级 tiling、pipeline、双缓冲等高级优化
- 我们的 WMMA kernel 只用了最基本的 Tensor Core 调用
