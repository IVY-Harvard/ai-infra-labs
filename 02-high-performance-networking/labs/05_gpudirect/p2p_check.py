#!/usr/bin/env python3
"""GPU P2P Connectivity & Bandwidth Check - 检测 GPU 间 P2P 可达性与带宽"""

import subprocess
import argparse
import json
import time
from typing import Dict, List, Optional

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_PYCUDA = True
except ImportError:
    HAS_PYCUDA = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def get_gpu_count() -> int:
    """获取 GPU 数量"""
    if HAS_PYCUDA:
        return cuda.Device.count()
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        return len([l for l in result.stdout.split("\n") if l.startswith("GPU")])
    except Exception:
        return 0


def check_p2p_access(src_gpu: int, dst_gpu: int) -> bool:
    """检查两个 GPU 之间是否支持 P2P 访问"""
    if not HAS_PYCUDA:
        return False
    src_dev = cuda.Device(src_gpu)
    dst_dev = cuda.Device(dst_gpu)
    return src_dev.can_access_peer(dst_dev)


def measure_p2p_bandwidth(src_gpu: int, dst_gpu: int,
                          size_bytes: int = 64 * 1024 * 1024,
                          iterations: int = 100) -> Optional[float]:
    """测量两个 GPU 之间的 P2P 带宽 (GB/s)"""
    if not HAS_PYCUDA:
        return None

    src_ctx = cuda.Device(src_gpu).make_context()
    src_buf = cuda.mem_alloc(size_bytes)

    dst_ctx = cuda.Device(dst_gpu).make_context()
    dst_buf = cuda.mem_alloc(size_bytes)

    # 启用 P2P
    try:
        cuda.Context.enable_peer_access(src_ctx)
    except cuda.LogicError:
        pass

    # Warmup
    for _ in range(10):
        cuda.memcpy_peer(dst_buf, src_buf, size_bytes, dst_ctx, src_ctx)
    dst_ctx.synchronize()

    # 计时
    start = time.perf_counter()
    for _ in range(iterations):
        cuda.memcpy_peer(dst_buf, src_buf, size_bytes, dst_ctx, src_ctx)
    dst_ctx.synchronize()
    elapsed = time.perf_counter() - start

    bandwidth = (size_bytes * iterations) / elapsed / (1024**3)

    dst_ctx.pop()
    src_ctx.pop()
    return bandwidth


def build_p2p_matrix(gpu_count: int) -> Dict[str, List[List]]:
    """构建 P2P 访问矩阵和带宽矩阵"""
    access_matrix = [[False] * gpu_count for _ in range(gpu_count)]
    bandwidth_matrix = [[0.0] * gpu_count for _ in range(gpu_count)]

    for i in range(gpu_count):
        for j in range(gpu_count):
            if i == j:
                access_matrix[i][j] = True
                continue
            access_matrix[i][j] = check_p2p_access(i, j)

    return {"access": access_matrix, "bandwidth": bandwidth_matrix}


def print_access_matrix(gpu_count: int, matrix: List[List[bool]]):
    """打印 P2P 访问矩阵"""
    print("\nP2P 访问矩阵 (O=支持, X=不支持):")
    header = "       " + "".join(f"GPU{j:2d}  " for j in range(gpu_count))
    print(header)
    for i in range(gpu_count):
        row = f"GPU{i:2d}  "
        for j in range(gpu_count):
            if i == j:
                row += "  -    "
            else:
                row += "  O    " if matrix[i][j] else "  X    "
        print(row)


def print_bandwidth_matrix(gpu_count: int, matrix: List[List[float]]):
    """打印带宽矩阵"""
    print("\nP2P 带宽矩阵 (GB/s):")
    header = "       " + "".join(f"GPU{j:2d}  " for j in range(gpu_count))
    print(header)
    for i in range(gpu_count):
        row = f"GPU{i:2d}  "
        for j in range(gpu_count):
            if i == j:
                row += "  -    "
            elif matrix[i][j] > 0:
                row += f"{matrix[i][j]:5.1f}  "
            else:
                row += "  N/A  "
        print(row)


def run_cuda_sample_p2p() -> str:
    """尝试运行 CUDA sample p2pBandwidthLatencyTest"""
    sample_paths = [
        "/usr/local/cuda/samples/bin/p2pBandwidthLatencyTest",
        "/usr/local/cuda/extras/demo_suite/p2pBandwidthLatencyTest",
    ]
    for path in sample_paths:
        try:
            result = subprocess.run([path], capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                return result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return ""


def main():
    parser = argparse.ArgumentParser(description="GPU P2P 连接检查工具")
    parser.add_argument("--bandwidth", action="store_true", help="测量 P2P 带宽")
    parser.add_argument("--size", type=int, default=64, help="传输数据大小 (MB, 默认: 64)")
    parser.add_argument("--output", "-o", help="输出 JSON 文件路径")
    args = parser.parse_args()

    gpu_count = get_gpu_count()
    if gpu_count == 0:
        print("错误: 未检测到 GPU")
        return

    print(f"检测到 {gpu_count} 个 GPU")
    print("=" * 50)

    matrices = build_p2p_matrix(gpu_count)
    print_access_matrix(gpu_count, matrices["access"])

    if args.bandwidth and HAS_PYCUDA:
        print("\n[*] 正在测量 P2P 带宽...")
        size_bytes = args.size * 1024 * 1024
        for i in range(gpu_count):
            for j in range(gpu_count):
                if i != j and matrices["access"][i][j]:
                    bw = measure_p2p_bandwidth(i, j, size_bytes)
                    if bw:
                        matrices["bandwidth"][i][j] = bw
        print_bandwidth_matrix(gpu_count, matrices["bandwidth"])

    if args.output:
        with open(args.output, "w") as f:
            json.dump(matrices, f, indent=2)
        print(f"\n[+] 结果已保存: {args.output}")


if __name__ == "__main__":
    main()
