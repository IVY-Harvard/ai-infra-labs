/**
 * Lab 04: cuBLAS GEMM — 厂商优化的矩阵乘法
 *
 * cuBLAS 是 NVIDIA 花费数千工程师年优化的库。
 * 这里展示如何调用它，以及它能达到的性能水平。
 *
 * 编译: nvcc -O2 -arch=sm_90 -lcublas -o cublas_gemm cublas_gemm.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error at %s:%d: %d\n", \
                __FILE__, __LINE__, status); \
        exit(1); \
    } \
} while(0)

int main() {
    int sizes[] = {512, 1024, 2048, 4096, 8192};
    int num_sizes = 5;

    printf("================================================================\n");
    printf("cuBLAS GEMM 性能测试\n");
    printf("对比 FP32 (CUDA Core) vs FP16 (Tensor Core)\n");
    printf("================================================================\n\n");

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // 启用 Tensor Core（TF32 模式 for FP32）
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    for (int s = 0; s < num_sizes; s++) {
        int M = sizes[s], N = sizes[s], K = sizes[s];

        printf("M=N=K=%d:\n", M);

        // ============================================================
        // FP32 GEMM (使用 TF32 Tensor Core 加速)
        // ============================================================
        {
            size_t bytes_A = M * K * sizeof(float);
            size_t bytes_B = K * N * sizeof(float);
            size_t bytes_C = M * N * sizeof(float);

            float *d_A, *d_B, *d_C;
            CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
            CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
            CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

            // 随机初始化
            float* h_temp = (float*)malloc(bytes_A > bytes_B ? bytes_A : bytes_B);
            for (int i = 0; i < M * K; i++) h_temp[i] = ((float)rand()/RAND_MAX)*2-1;
            CUDA_CHECK(cudaMemcpy(d_A, h_temp, bytes_A, cudaMemcpyHostToDevice));
            for (int i = 0; i < K * N; i++) h_temp[i] = ((float)rand()/RAND_MAX)*2-1;
            CUDA_CHECK(cudaMemcpy(d_B, h_temp, bytes_B, cudaMemcpyHostToDevice));
            free(h_temp);

            float alpha = 1.0f, beta = 0.0f;

            // Warmup
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
            CUDA_CHECK(cudaDeviceSynchronize());

            // Benchmark
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            int runs = (M <= 1024) ? 50 : (M <= 4096 ? 20 : 5);
            CUDA_CHECK(cudaEventRecord(start));
            for (int r = 0; r < runs; r++) {
                CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
            }
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            ms /= runs;

            double flops = 2.0 * M * N * K;
            double tflops = flops / (ms / 1000.0) / 1e12;
            printf("  FP32 (TF32 Tensor):  %7.2f ms, %6.2f TFLOPS\n", ms, tflops);

            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_B));
            CUDA_CHECK(cudaFree(d_C));
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
        }

        // ============================================================
        // FP16 GEMM (Tensor Core)
        // ============================================================
        {
            size_t bytes_A = M * K * sizeof(half);
            size_t bytes_B = K * N * sizeof(half);
            size_t bytes_C = M * N * sizeof(half);

            half *d_A, *d_B, *d_C;
            CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
            CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
            CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

            // 初始化
            half* h_temp = (half*)malloc(M * K > K * N ? bytes_A : bytes_B);
            for (int i = 0; i < M * K; i++)
                h_temp[i] = __float2half(((float)rand()/RAND_MAX)*2-1);
            CUDA_CHECK(cudaMemcpy(d_A, h_temp, bytes_A, cudaMemcpyHostToDevice));
            for (int i = 0; i < K * N; i++)
                h_temp[i] = __float2half(((float)rand()/RAND_MAX)*2-1);
            CUDA_CHECK(cudaMemcpy(d_B, h_temp, bytes_B, cudaMemcpyHostToDevice));
            free(h_temp);

            half alpha_h = __float2half(1.0f);
            half beta_h = __float2half(0.0f);

            // Warmup
            CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, M, K, &alpha_h, d_B, N, d_A, K, &beta_h, d_C, N));
            CUDA_CHECK(cudaDeviceSynchronize());

            // Benchmark
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            int runs = (M <= 1024) ? 50 : (M <= 4096 ? 20 : 5);
            CUDA_CHECK(cudaEventRecord(start));
            for (int r = 0; r < runs; r++) {
                CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         N, M, K, &alpha_h, d_B, N, d_A, K, &beta_h, d_C, N));
            }
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            ms /= runs;

            double flops = 2.0 * M * N * K;
            double tflops = flops / (ms / 1000.0) / 1e12;
            printf("  FP16 (Tensor Core):  %7.2f ms, %6.2f TFLOPS\n", ms, tflops);
            printf("  Peak FP16 Tensor: ~148 TFLOPS, 利用率: %.1f%%\n\n", tflops/148*100);

            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_B));
            CUDA_CHECK(cudaFree(d_C));
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
        }
    }

    printf("================================================================\n");
    printf("关键观察:\n");
    printf("  1. cuBLAS FP16 能接近理论峰值（>80%%利用率）\n");
    printf("  2. 小矩阵利用率低（不够填满 GPU）\n");
    printf("  3. FP16 比 FP32 快 ~2-3x (Tensor Core vs TF32)\n");
    printf("  4. 这就是为什么 AI 推理/训练都用 FP16/BF16\n");
    printf("================================================================\n");

    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
