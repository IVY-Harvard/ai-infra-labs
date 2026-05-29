/**
 * Lab 03: 共享内存优化 — 矩阵转置
 *
 * 矩阵转置是展示共享内存威力的经典案例：
 * - 朴素实现：读 coalesced，写 non-coalesced（或反过来）
 * - 共享内存版：读写都是 coalesced
 *
 * 编译: nvcc -O2 -arch=sm_90 -o shared_memory shared_memory.cu
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

#define TILE_DIM 32   // tile 大小（要等于 warp 大小）
#define BLOCK_ROWS 8  // 每个 block 处理的行数

// ============================================================
// Kernel 1: 朴素矩阵转置
//
// 问题：读是 coalesced（行优先读），但写是 non-coalesced
// 因为转置后的写入变成了按列写，步长 = width
// ============================================================
__global__ void transpose_naive(float* dst, const float* src, int width, int height) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            // 读: src[y+j][x] → coalesced（x 连续）
            // 写: dst[x][y+j] → NON-coalesced（y+j 连续但 x 是行索引）
            dst[x * height + (y + j)] = src[(y + j) * width + x];
        }
    }
}

// ============================================================
// Kernel 2: 共享内存矩阵转置
//
// 思路：
// 1. 从全局内存 coalesced 读 → 写入共享内存
// 2. __syncthreads()
// 3. 从共享内存读（转置后的位置）→ coalesced 写入全局内存
//
// 关键：共享内存充当"跳板"，让两次全局内存访问都是 coalesced
// ============================================================
__global__ void transpose_shared(float* dst, const float* src, int width, int height) {
    // 共享内存 tile
    // 注意：TILE_DIM+1 是为了避免 bank conflict（后面解释）
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Step 1: Coalesced 读全局内存 → 写共享内存
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = src[(y + j) * width + x];
        }
    }

    // 等待所有线程完成写入
    __syncthreads();

    // Step 2: 转置坐标 —— 从另一个 tile 的位置读
    x = blockIdx.y * TILE_DIM + threadIdx.x;  // 注意: blockIdx.y 和 blockIdx.x 交换了
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Coalesced 写全局内存（从共享内存读转置后的数据）
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            // 从共享内存读: tile[threadIdx.x][threadIdx.y+j]
            // 注意 threadIdx.x 和 threadIdx.y 的交换！这就是转置
            dst[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// ============================================================
// Kernel 3: 共享内存 + 无 Bank Conflict 避免（对比用）
// 使用 TILE_DIM 而非 TILE_DIM+1 来展示 bank conflict 的影响
// ============================================================
__global__ void transpose_shared_bank_conflict(float* dst, const float* src,
                                                int width, int height) {
    // 没有 +1 padding → 会有 bank conflict
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = src[(y + j) * width + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            dst[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

void verify_transpose(const float* src, const float* dst, int width, int height) {
    float* h_src = (float*)malloc(width * height * sizeof(float));
    float* h_dst = (float*)malloc(width * height * sizeof(float));

    CUDA_CHECK(cudaMemcpy(h_src, src, width * height * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dst, dst, width * height * sizeof(float), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int y = 0; y < height && errors < 5; y++) {
        for (int x = 0; x < width && errors < 5; x++) {
            if (h_src[y * width + x] != h_dst[x * height + y]) {
                printf("  Mismatch at (%d,%d): expected %.2f, got %.2f\n",
                       x, y, h_src[y * width + x], h_dst[x * height + y]);
                errors++;
            }
        }
    }

    if (errors == 0) printf("  ✓ 验证通过\n");
    else printf("  ✗ 发现 %d 个错误\n", errors);

    free(h_src);
    free(h_dst);
}

float benchmark_transpose(void (*kernel)(float*, const float*, int, int),
                          float* d_dst, const float* d_src,
                          int width, int height) {
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

    // Warmup
    for (int i = 0; i < 5; i++) {
        kernel<<<grid, block>>>(d_dst, d_src, width, height);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int runs = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < runs; i++) {
        kernel<<<grid, block>>>(d_dst, d_src, width, height);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / runs;
}

int main() {
    const int WIDTH = 4096;
    const int HEIGHT = 4096;
    const size_t bytes = WIDTH * HEIGHT * sizeof(float);

    printf("================================================================\n");
    printf("共享内存优化: 矩阵转置\n");
    printf("矩阵大小: %d × %d (%.0f MB)\n", WIDTH, HEIGHT, bytes / 1e6);
    printf("================================================================\n\n");

    // 分配内存
    float *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, bytes));

    // 初始化
    float* h_data = (float*)malloc(bytes);
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        h_data[i] = (float)i;
    }
    CUDA_CHECK(cudaMemcpy(d_src, h_data, bytes, cudaMemcpyHostToDevice));

    // ---- Test 1: Naive Transpose ----
    {
        CUDA_CHECK(cudaMemset(d_dst, 0, bytes));
        float ms = benchmark_transpose(transpose_naive, d_dst, d_src, WIDTH, HEIGHT);
        float bw = 2.0f * bytes / (ms / 1000.0f) / 1e9;

        printf("1. 朴素转置 (写 non-coalesced)\n");
        printf("   时间: %.3f ms, 有效带宽: %.1f GB/s\n", ms, bw);
        verify_transpose(d_src, d_dst, WIDTH, HEIGHT);
        printf("\n");
    }

    // ---- Test 2: Shared Memory Transpose (with bank conflict) ----
    {
        CUDA_CHECK(cudaMemset(d_dst, 0, bytes));
        float ms = benchmark_transpose(transpose_shared_bank_conflict, d_dst, d_src, WIDTH, HEIGHT);
        float bw = 2.0f * bytes / (ms / 1000.0f) / 1e9;

        printf("2. 共享内存转置 (有 bank conflict)\n");
        printf("   时间: %.3f ms, 有效带宽: %.1f GB/s\n", ms, bw);
        verify_transpose(d_src, d_dst, WIDTH, HEIGHT);
        printf("\n");
    }

    // ---- Test 3: Shared Memory Transpose (no bank conflict) ----
    {
        CUDA_CHECK(cudaMemset(d_dst, 0, bytes));
        float ms = benchmark_transpose(transpose_shared, d_dst, d_src, WIDTH, HEIGHT);
        float bw = 2.0f * bytes / (ms / 1000.0f) / 1e9;

        printf("3. 共享内存转置 (padding 消除 bank conflict)\n");
        printf("   时间: %.3f ms, 有效带宽: %.1f GB/s\n", ms, bw);
        verify_transpose(d_src, d_dst, WIDTH, HEIGHT);
        printf("\n");
    }

    // ---- 解释 ----
    printf("================================================================\n");
    printf("Bank Conflict 解释:\n");
    printf("  共享内存分为 32 个 bank，每 4 字节一个 bank。\n");
    printf("  当 Warp 中多个线程访问同一个 bank 时发生 conflict。\n");
    printf("  tile[TILE_DIM][TILE_DIM] 中按列访问时，步长恰好是 32，\n");
    printf("  所有线程访问同一个 bank → 32-way conflict → 串行化！\n");
    printf("  tile[TILE_DIM][TILE_DIM+1] 的 +1 padding 打破了这个对齐。\n");
    printf("================================================================\n");

    // 清理
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    free(h_data);

    return 0;
}
