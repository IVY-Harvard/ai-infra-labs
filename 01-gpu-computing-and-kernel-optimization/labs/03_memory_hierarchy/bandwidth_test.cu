/**
 * Lab 03: GPU 内存带宽实测
 *
 * 测量不同访问模式下的 HBM 带宽，理解 coalescing 的重要性。
 *
 * 编译: nvcc -O2 -arch=sm_90 -o bandwidth_test bandwidth_test.cu
 */

#include <stdio.h>
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
// Kernel 1: 顺序读写（Coalesced Access）
// 每个线程访问连续的地址 → 内存事务最少 → 带宽最高
// ============================================================
__global__ void coalesced_copy(float* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// ============================================================
// Kernel 2: 跨步读写（Strided Access）
// 线程访问间隔 stride 的地址 → 多次内存事务 → 带宽暴跌
// ============================================================
__global__ void strided_copy(float* dst, const float* src, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 每个线程访问 idx * stride 位置的数据
    int offset = (idx * stride) % n;
    if (idx < n / stride) {
        dst[offset] = src[offset];
    }
}

// ============================================================
// Kernel 3: 向量化读写（float4 = 一次读 16 bytes）
// 利用宽内存事务进一步提升带宽
// ============================================================
__global__ void vectorized_copy(float4* dst, const float4* src, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        dst[idx] = src[idx];
    }
}

// ============================================================
// Kernel 4: 只读（测量纯读带宽）
// ============================================================
__global__ void read_only(const float* src, float* dummy, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;
    if (idx < n) {
        val = src[idx];
    }
    // 防止编译器优化掉读操作
    if (val == -999.0f) {
        dummy[0] = val;
    }
}

float benchmark_kernel(void (*kernel_wrapper)(float*, const float*, int, cudaStream_t),
                       float* d_dst, const float* d_src, int n,
                       int warmup_runs, int benchmark_runs) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        kernel_wrapper(d_dst, d_src, n, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < benchmark_runs; i++) {
        kernel_wrapper(d_dst, d_src, n, stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return ms / benchmark_runs;
}

void launch_coalesced(float* dst, const float* src, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    coalesced_copy<<<grid, block, 0, stream>>>(dst, src, n);
}

void launch_strided(float* dst, const float* src, int n, cudaStream_t stream) {
    int stride = 32;  // 步长32 → 每个 warp 的线程访问相隔 32 个 float
    int block = 256;
    int grid = (n / stride + block - 1) / block;
    strided_copy<<<grid, block, 0, stream>>>(dst, src, n, stride);
}

void launch_vectorized(float* dst, const float* src, int n, cudaStream_t stream) {
    int n4 = n / 4;
    int block = 256;
    int grid = (n4 + block - 1) / block;
    vectorized_copy<<<grid, block, 0, stream>>>((float4*)dst, (const float4*)src, n4);
}

int main() {
    const int N = 1 << 26;  // 64M floats = 256 MB
    const size_t bytes = N * sizeof(float);
    const int WARMUP = 5;
    const int RUNS = 20;

    printf("================================================================\n");
    printf("GPU 内存带宽测试\n");
    printf("数据量: %d 元素 (%.0f MB)\n", N, bytes / 1e6);
    printf("================================================================\n\n");

    // 分配内存
    float *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, bytes));

    // 初始化
    float* h_data = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(d_src, h_data, bytes, cudaMemcpyHostToDevice));

    // ---- Test 1: Coalesced Copy ----
    {
        float ms = benchmark_kernel(launch_coalesced, d_dst, d_src, N, WARMUP, RUNS);
        float bw = 2.0f * bytes / (ms / 1000.0f) / 1e9;  // 读+写
        printf("1. Coalesced Copy (顺序访问)\n");
        printf("   时间: %.3f ms, 有效带宽: %.1f GB/s\n", ms, bw);
        printf("   → 这是最优的访问模式，线程连续访问连续地址\n\n");
    }

    // ---- Test 2: Strided Copy ----
    {
        float ms = benchmark_kernel(launch_strided, d_dst, d_src, N, WARMUP, RUNS);
        float bw = 2.0f * (bytes / 32) / (ms / 1000.0f) / 1e9;
        printf("2. Strided Copy (步长=32 的跨步访问)\n");
        printf("   时间: %.3f ms, 有效带宽: %.1f GB/s\n", ms, bw);
        printf("   → Warp 内线程访问不连续，导致多次内存事务\n");
        printf("   → 带宽利用率大幅下降\n\n");
    }

    // ---- Test 3: Vectorized Copy ----
    {
        float ms = benchmark_kernel(launch_vectorized, d_dst, d_src, N, WARMUP, RUNS);
        float bw = 2.0f * bytes / (ms / 1000.0f) / 1e9;
        printf("3. Vectorized Copy (float4, 一次读16字节)\n");
        printf("   时间: %.3f ms, 有效带宽: %.1f GB/s\n", ms, bw);
        printf("   → 每个线程一次处理 4 个 float，减少指令数\n");
        printf("   → 通常比标量版本快 10-20%%\n\n");
    }

    printf("================================================================\n");
    printf("关键结论:\n");
    printf("  1. Coalesced access 是内存优化的第一要务\n");
    printf("  2. 跨步访问的带宽可能只有顺序访问的 1/N\n");
    printf("  3. 向量化读写 (float4) 是简单有效的优化手段\n");
    printf("  4. 有效带宽 vs 理论峰值的差距反映了优化空间\n");
    printf("================================================================\n");

    // 清理
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    free(h_data);

    return 0;
}
