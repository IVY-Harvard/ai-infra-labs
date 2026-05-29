"""
GPU 间 P2P (Peer-to-Peer) 带宽测试

测试任意两张 GPU 之间的直接数据传输带宽。
P2P 可以通过 NVLink 或 PCIe 进行，带宽差异巨大。
"""

import torch
import time
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class P2PResult:
    """P2P 带宽测试结果"""
    src_gpu: int
    dst_gpu: int
    can_access: bool
    bandwidth_gb_s: float
    data_size_mb: float


class P2PBenchmark:
    """GPU 间 P2P 带宽测试"""

    def __init__(self):
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus < 2:
            print("警告: 只检测到 1 张 GPU，P2P 测试需要至少 2 张")

    def check_p2p_access(self) -> List[List[bool]]:
        """检查 GPU 间 P2P 访问能力"""
        matrix = []
        for i in range(self.num_gpus):
            row = []
            for j in range(self.num_gpus):
                if i == j:
                    row.append(True)
                else:
                    row.append(torch.cuda.can_device_access_peer(i, j))
            matrix.append(row)
        return matrix

    def benchmark_pair(self, src: int, dst: int, size_mb: int = 256,
                       warmup: int = 10, runs: int = 50) -> P2PResult:
        """测试一对 GPU 间的 P2P 带宽"""
        can_access = torch.cuda.can_device_access_peer(src, dst)

        if src == dst:
            return P2PResult(src, dst, True, 0.0, size_mb)

        n_elements = size_mb * 1024 * 1024 // 4
        src_tensor = torch.randn(n_elements, device=f'cuda:{src}')
        dst_tensor = torch.empty(n_elements, device=f'cuda:{dst}')

        # 启用 P2P（如果支持）
        if can_access:
            try:
                torch.cuda.set_device(src)
                # P2P 访问在 PyTorch 中通过 tensor copy 自动使用
            except Exception:
                pass

        # Warmup
        for _ in range(warmup):
            dst_tensor.copy_(src_tensor)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(runs):
            dst_tensor.copy_(src_tensor)
        torch.cuda.synchronize()
        time_ms = (time.perf_counter() - start) / runs * 1000

        bw = size_mb * 1024 * 1024 / (time_ms / 1000) / 1e9

        del src_tensor, dst_tensor
        torch.cuda.empty_cache()

        return P2PResult(src, dst, can_access, bw, size_mb)

    def benchmark_all_pairs(self, size_mb: int = 256) -> List[List[P2PResult]]:
        """测试所有 GPU 对的 P2P 带宽"""
        results = []
        for i in range(self.num_gpus):
            row = []
            for j in range(self.num_gpus):
                result = self.benchmark_pair(i, j, size_mb)
                row.append(result)
            results.append(row)
        return results

    def print_results(self, results: List[List[P2PResult]]):
        """打印 P2P 带宽矩阵"""
        n = len(results)

        # P2P 访问矩阵
        print(f"\nP2P Access Matrix:")
        print(f"{'':>6}", end='')
        for j in range(n):
            print(f"  GPU{j:>2}", end='')
        print()
        for i in range(n):
            print(f"GPU{i:>2}:", end='')
            for j in range(n):
                if i == j:
                    print(f"     -", end='')
                else:
                    access = "Yes" if results[i][j].can_access else " No"
                    print(f"   {access}", end='')
            print()

        # 带宽矩阵
        print(f"\nP2P Bandwidth (GB/s):")
        print(f"{'':>6}", end='')
        for j in range(n):
            print(f"  GPU{j:>2}", end='')
        print()
        for i in range(n):
            print(f"GPU{i:>2}:", end='')
            for j in range(n):
                if i == j:
                    print(f"     -", end='')
                else:
                    bw = results[i][j].bandwidth_gb_s
                    print(f"  {bw:>4.0f}", end='')
            print()


if __name__ == "__main__":
    bench = P2PBenchmark()
    if bench.num_gpus >= 2:
        results = bench.benchmark_all_pairs()
        bench.print_results(results)
    else:
        print("需要至少 2 张 GPU")
