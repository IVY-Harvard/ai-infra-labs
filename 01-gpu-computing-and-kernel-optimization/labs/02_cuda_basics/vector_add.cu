/**
 * Lab 02: CUDA 向量加法
 *
 * 这是最简单的 CUDA kernel，但包含了 GPU 编程的所有核心步骤。
 * 理解了这个流程，就理解了所有 CUDA 程序的骨架。
 *
 * 编译: nvcc -O2 -o vector_add vector_add.cu
 * 运行: ./vector_add
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// ============================================================
// 错误检查宏 —— 每个 CUDA API 调用都应该检查错误
// 在生产代码中这是必须的，能帮你快速定位问题
// ============================================================
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ============================================================
// GPU Kernel: 向量加法
//
// __global__ 关键字表示这个函数在 GPU 上执行，由 CPU 调用
// 每个线程处理一个元素 —— 这是最基本的并行模式
// ============================================================
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    // 计算全局线程 ID
    // blockIdx.x: 当前 block 在 grid 中的索引
    // blockDim.x: 每个 block 有多少线程
    // threadIdx.x: 当前线程在 block 中的索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查：因为 grid 大小向上取整，可能有多余的线程
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// ============================================================
// CPU 版本（用于验证正确性）
// ============================================================
void vector_add_cpu(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// ============================================================
// 验证 GPU 结果是否正确
// ============================================================
int verify_result(const float* gpu_result, const float* cpu_result, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(gpu_result[i] - cpu_result[i]) > 1e-5) {
            printf("Mismatch at index %d: GPU=%.6f, CPU=%.6f\n",
                   i, gpu_result[i], cpu_result[i]);
            return 0;
        }
    }
    return 1;
}

int main() {
    // 参数设置
    const int N = 1 << 24;  // 16M 个元素 (~64MB for float)
    const int BLOCK_SIZE = 256;  // 每个 block 256 个线程（推荐值）

    printf("Vector Addition: N = %d (%.1f MB)\n", N, N * sizeof(float) / 1e6);
    printf("Block size: %d\n", BLOCK_SIZE);

    // ================================================================
    // Step 1: 在 Host (CPU) 上分配内存并初始化数据
    // ================================================================
    size_t bytes = N * sizeof(float);
    float* h_a = (float*)malloc(bytes);  // host 输入 A
    float* h_b = (float*)malloc(bytes);  // host 输入 B
    float* h_c = (float*)malloc(bytes);  // host 输出（GPU 结果）
    float* h_ref = (float*)malloc(bytes);  // host 参考结果（CPU 计算）

    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)(rand()) / RAND_MAX;
        h_b[i] = (float)(rand()) / RAND_MAX;
    }

    // ================================================================
    // Step 2: 在 Device (GPU) 上分配内存
    //
    // cudaMalloc 类似 malloc，但分配的是 GPU 显存
    // 返回的指针只能在 GPU 上使用
    // ================================================================
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // ================================================================
    // Step 3: 将数据从 Host 复制到 Device (H2D)
    //
    // 这一步通过 PCIe 总线传输，带宽 ~32 GB/s (Gen4 x16)
    // 对于大数据量，传输时间可能比计算时间还长！
    // ================================================================
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // ================================================================
    // Step 4: 配置 Kernel 启动参数并执行
    //
    // Grid 大小 = 需要多少个 block 才能覆盖所有数据
    // Block 大小 = 每个 block 多少线程（通常 128/256）
    // ================================================================
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 向上取整
    printf("Grid size: %d blocks\n", grid_size);
    printf("Total threads: %d\n", grid_size * BLOCK_SIZE);

    // 使用 CUDA Event 计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 预热（第一次 kernel 启动有额外开销）
    vector_add_kernel<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 正式计时
    CUDA_CHECK(cudaEventRecord(start));

    // 启动 Kernel！
    // <<<grid_size, BLOCK_SIZE>>> 是 CUDA 特有的语法
    vector_add_kernel<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_c, N);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

    // ================================================================
    // Step 5: 将结果从 Device 复制回 Host (D2H)
    // ================================================================
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // ================================================================
    // Step 6: 验证正确性
    // ================================================================
    vector_add_cpu(h_a, h_b, h_ref, N);
    if (verify_result(h_c, h_ref, N)) {
        printf("\n✓ 结果验证通过！\n");
    } else {
        printf("\n✗ 结果验证失败！\n");
    }

    // ================================================================
    // 性能分析
    // ================================================================
    printf("\n--- 性能分析 ---\n");
    printf("Kernel 执行时间: %.3f ms\n", kernel_ms);

    // 计算有效带宽
    // 向量加法：读 2 个数组 + 写 1 个数组 = 3 × N × sizeof(float)
    float total_bytes = 3.0f * N * sizeof(float);
    float bandwidth_gb_s = total_bytes / (kernel_ms / 1000.0f) / 1e9;
    printf("有效带宽: %.1f GB/s\n", bandwidth_gb_s);
    printf("  (理论峰值 HBM 带宽: ~4000 GB/s for H20)\n");
    printf("  (带宽利用率: ~%.1f%%)\n", bandwidth_gb_s / 4000.0f * 100);

    // 计算 Arithmetic Intensity
    float flops = N;  // N 次加法
    float ai = flops / total_bytes;
    printf("\nArithmetic Intensity: %.4f FLOP/Byte\n", ai);
    printf("  → 极度访存密集（AI << 1）\n");
    printf("  → 性能受限于 HBM 带宽，增加算力无帮助\n");

    // ================================================================
    // Step 7: 释放内存
    // ================================================================
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_ref);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
