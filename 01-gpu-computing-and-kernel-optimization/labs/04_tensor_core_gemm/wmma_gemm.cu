/**
 * Lab 04: WMMA GEMM — 使用 Tensor Core 的矩阵乘法
 *
 * 通过 WMMA (Warp Matrix Multiply-Accumulate) API 调用 Tensor Core。
 * 这是一个教学级实现，展示 Tensor Core 的基本用法。
 *
 * D[M×N] = A[M×K] × B[K×N] + C[M×N]
 * 输入: FP16, 累加: FP32
 *
 * 编译: nvcc -O2 -arch=sm_90 -o wmma_gemm wmma_gemm.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// WMMA 操作的矩阵尺寸（Tensor Core 的固定大小）
// 每个 Warp 一次操作处理 16×16×16 的矩阵块
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// ============================================================
// WMMA GEMM Kernel
//
// 每个 Warp 负责计算输出矩阵中一个 16×16 的 tile。
// 沿 K 维度循环，每次加载 A[16×16] 和 B[16×16]，
// 在 Tensor Core 上做 MMA 并累加到 C。
// ============================================================
__global__ void wmma_gemm_kernel(const half* A, const half* B, float* C,
                                  int M, int N, int K) {
    // 计算当前 Warp 负责的输出 tile 的位置
    // 注意：WMMA 操作以 Warp 为单位，所以需要按 Warp 索引
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = blockIdx.y;

    // 输出 tile 的左上角位置
    int row = warpM * WMMA_M;
    int col = warpN * WMMA_N;

    if (row >= M || col >= N) return;

    // 声明矩阵片段（Fragment）
    // 这些是 Warp 级别的数据结构，32 个线程共同持有一个矩阵
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 初始化累加器为 0
    wmma::fill_fragment(c_frag, 0.0f);

    // 沿 K 维度循环
    for (int k = 0; k < K; k += WMMA_K) {
        // 从全局内存加载 A 和 B 的 tile 到 fragment
        // load_matrix_sync: 整个 Warp 协作加载
        if (row < M && k < K) {
            wmma::load_matrix_sync(a_frag, A + row * K + k, K);
        }
        if (k < K && col < N) {
            wmma::load_matrix_sync(b_frag, B + k + col * K, K);
        }

        // Tensor Core 执行矩阵乘加！
        // 这一条指令在硬件上执行 16×16×16 的 MMA
        // = 16×16×16×2 = 8192 FLOPs
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 将结果从 fragment 存储到全局内存
    if (row < M && col < N) {
        wmma::store_matrix_sync(C + row * N + col, c_frag, N, wmma::mem_row_major);
    }
}

void init_matrix_half(half* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    }
}

int main() {
    int sizes[] = {512, 1024, 2048, 4096};
    int num_sizes = 4;

    printf("================================================================\n");
    printf("WMMA GEMM 性能测试 (FP16 input, FP32 accumulate, Tensor Core)\n");
    printf("================================================================\n\n");

    for (int s = 0; s < num_sizes; s++) {
        int M = sizes[s], N = sizes[s], K = sizes[s];

        // 确保维度是 WMMA_M/N/K 的倍数
        M = ((M + WMMA_M - 1) / WMMA_M) * WMMA_M;
        N = ((N + WMMA_N - 1) / WMMA_N) * WMMA_N;
        K = ((K + WMMA_K - 1) / WMMA_K) * WMMA_K;

        size_t bytes_A = M * K * sizeof(half);
        size_t bytes_B = K * N * sizeof(half);
        size_t bytes_C = M * N * sizeof(float);

        // Host 分配
        half* h_A = (half*)malloc(bytes_A);
        half* h_B = (half*)malloc(bytes_B);
        init_matrix_half(h_A, M, K);
        init_matrix_half(h_B, K, N);

        // Device 分配
        half *d_A, *d_B;
        float *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
        CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
        CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

        // 启动配置
        // 每个 block 有多个 warp，每个 warp 处理一个 16×16 的 tile
        int warps_per_block = 4;
        dim3 block(32 * warps_per_block, 1);  // 32 threads per warp × 4 warps
        dim3 grid((M / WMMA_M + warps_per_block - 1) / warps_per_block,
                  N / WMMA_N);

        // Warmup
        wmma_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Benchmark
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        int runs = (sizes[s] <= 1024) ? 20 : 5;
        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < runs; r++) {
            wmma_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= runs;

        double flops = 2.0 * M * N * K;
        double tflops = flops / (ms / 1000.0) / 1e12;
        printf("M=N=K=%d:\n", sizes[s]);
        printf("  WMMA GEMM: %.2f ms, %.2f TFLOPS (FP16)\n", ms, tflops);
        printf("  (理论峰值 FP16 Tensor: ~148 TFLOPS on H20)\n");
        printf("  (利用率: %.1f%%)\n\n", tflops / 148.0 * 100);

        // 清理
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        free(h_A);
        free(h_B);
    }

    printf("注意: 这个 WMMA kernel 是教学级实现，没有做:\n");
    printf("  - 多级 tiling (block-level + warp-level)\n");
    printf("  - 共享内存预加载 + 双缓冲\n");
    printf("  - 全局内存访问优化\n");
    printf("  要达到接近峰值性能，请使用 cuBLAS。\n");

    return 0;
}
