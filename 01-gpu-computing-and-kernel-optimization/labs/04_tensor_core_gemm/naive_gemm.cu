/**
 * Lab 04: 朴素 GEMM — 每线程计算输出矩阵的一个元素
 *
 * 这是最直观的矩阵乘法实现，但性能很差。
 * 通过对比后面的 Tensor Core 版本，理解硬件加速的巨大差异。
 *
 * C[M×N] = A[M×K] × B[K×N]
 *
 * 编译: nvcc -O2 -arch=sm_90 -o naive_gemm naive_gemm.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================
// 版本 1: 最朴素 — 每线程计算 C 的一个元素
// 每个线程读 A 的一整行和 B 的一整列 → 全局内存访问量巨大
// ============================================================
__global__ void gemm_naive(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ============================================================
// 版本 2: 共享内存 tiling — 减少全局内存访问
// 将 A 和 B 的 tile 加载到共享内存，复用数据
// ============================================================
#define TILE_SIZE 32

__global__ void gemm_tiled(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // 沿 K 维度分 tile
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 加载 A 的 tile 到共享内存
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 加载 B 的 tile 到共享内存
        if (t * TILE_SIZE + threadIdx.y < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // 在共享内存中计算部分和
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

int main() {
    // 测试多个矩阵大小
    int sizes[] = {512, 1024, 2048, 4096};
    int num_sizes = 4;

    printf("================================================================\n");
    printf("朴素 GEMM 性能测试 (FP32, CUDA Core)\n");
    printf("================================================================\n\n");

    for (int s = 0; s < num_sizes; s++) {
        int M = sizes[s], N = sizes[s], K = sizes[s];
        size_t bytes_A = M * K * sizeof(float);
        size_t bytes_B = K * N * sizeof(float);
        size_t bytes_C = M * N * sizeof(float);

        // 分配
        float *h_A = (float*)malloc(bytes_A);
        float *h_B = (float*)malloc(bytes_B);
        init_matrix(h_A, M, K);
        init_matrix(h_B, K, N);

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
        CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
        CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        printf("M=N=K=%d:\n", sizes[s]);

        // ---- Naive ----
        // Warmup
        gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        int runs = (sizes[s] <= 1024) ? 10 : 3;
        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < runs; r++) {
            gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= runs;

        double flops = 2.0 * M * N * K;
        double tflops = flops / (ms / 1000.0) / 1e12;
        printf("  Naive:  %.2f ms, %.2f TFLOPS\n", ms, tflops);

        // ---- Tiled ----
        gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < runs; r++) {
            gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= runs;
        tflops = flops / (ms / 1000.0) / 1e12;
        printf("  Tiled:  %.2f ms, %.2f TFLOPS\n", ms, tflops);
        printf("  (理论峰值 FP32: ~44 TFLOPS on H20)\n\n");

        // 清理
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        free(h_A);
        free(h_B);
    }

    return 0;
}
