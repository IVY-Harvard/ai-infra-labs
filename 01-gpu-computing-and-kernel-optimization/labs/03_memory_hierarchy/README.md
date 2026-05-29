# Lab 03: 内存层次实测

## 实验目的

用实际测量理解 GPU 内存层次的性能差异：
1. 测量 HBM（全局内存）带宽
2. 测量共享内存带宽
3. 展示共享内存优化矩阵转置的效果
4. 理解 bank conflict

## 前置要求

- CUDA Toolkit
- 已读 theory/01（内存层次部分）

## 编译和运行

```bash
nvcc -O2 -arch=sm_90 -o bandwidth_test bandwidth_test.cu
nvcc -O2 -arch=sm_90 -o shared_memory shared_memory.cu
./bandwidth_test
./shared_memory
```

## 实验内容

### bandwidth_test.cu
- 测量全局内存的 read/write/copy 带宽
- 测量不同访问模式（coalesced vs strided）的带宽差异
- 量化 coalescing 的重要性

### shared_memory.cu
- 矩阵转置的两种实现：朴素版 vs 共享内存版
- 展示共享内存如何将 non-coalesced 写入转为 coalesced 写入
- Bank conflict 的影响和解决

## 关键思考

- 为什么 coalesced 和 non-coalesced 有如此大的差距？
- 共享内存为什么能加速矩阵转置？
- bank conflict 是怎么产生的？padding 为什么能解决？
