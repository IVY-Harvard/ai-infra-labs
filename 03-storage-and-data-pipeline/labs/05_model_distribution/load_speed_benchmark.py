"""
模型加载速度基准测试

对比不同格式和加载方式的速度：
1. PyTorch .bin（pickle 反序列化）
2. safetensors（内存映射）
3. mmap 直接映射

用法：
    python load_speed_benchmark.py --model-size 256 --output-dir /tmp/load_bench
"""

import os
import time
import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class LoadBenchResult:
    """加载测试结果"""
    format_name: str
    file_size_mb: float
    load_time_s: float
    throughput_mbps: float
    method: str


class ModelLoadBenchmark:
    """模型加载速度基准测试"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[LoadBenchResult] = []

    def create_test_model(self, hidden_size: int = 4096,
                          num_layers: int = 4) -> nn.Module:
        """创建测试模型"""
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.LayerNorm(hidden_size),
            ])
        model = nn.Sequential(*layers)
        return model

    def save_formats(self, model: nn.Module):
        """保存为不同格式"""
        state_dict = model.state_dict()

        # PyTorch .bin
        bin_path = self.output_dir / "model.bin"
        torch.save(state_dict, bin_path)
        print(f"  .bin 大小: {bin_path.stat().st_size/1024/1024:.1f}MB")

        # safetensors
        try:
            from safetensors.torch import save_file
            st_path = self.output_dir / "model.safetensors"
            save_file(state_dict, st_path)
            print(f"  .safetensors 大小: "
                  f"{st_path.stat().st_size/1024/1024:.1f}MB")
        except ImportError:
            print("  [跳过] safetensors 未安装")

        return state_dict

    def benchmark_pytorch_load(self, num_runs: int = 3) -> LoadBenchResult:
        """测试 PyTorch .bin 加载速度"""
        bin_path = self.output_dir / "model.bin"
        if not bin_path.exists():
            return None

        file_size = bin_path.stat().st_size
        times = []

        for _ in range(num_runs):
            # 清除 page cache
            self._drop_caches()

            t0 = time.perf_counter()
            state_dict = torch.load(bin_path, map_location="cpu",
                                   weights_only=True)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            del state_dict

        avg_time = sum(times) / len(times)
        size_mb = file_size / 1024 / 1024

        result = LoadBenchResult(
            format_name="PyTorch .bin",
            file_size_mb=size_mb,
            load_time_s=avg_time,
            throughput_mbps=size_mb / avg_time,
            method="torch.load (pickle)",
        )
        self.results.append(result)
        return result

    def benchmark_safetensors_load(self, num_runs: int = 3) -> LoadBenchResult:
        """测试 safetensors 加载速度"""
        try:
            from safetensors.torch import load_file
        except ImportError:
            print("  [跳过] safetensors 未安装")
            return None

        st_path = self.output_dir / "model.safetensors"
        if not st_path.exists():
            return None

        file_size = st_path.stat().st_size
        times = []

        for _ in range(num_runs):
            self._drop_caches()

            t0 = time.perf_counter()
            state_dict = load_file(st_path)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            del state_dict

        avg_time = sum(times) / len(times)
        size_mb = file_size / 1024 / 1024

        result = LoadBenchResult(
            format_name="safetensors",
            file_size_mb=size_mb,
            load_time_s=avg_time,
            throughput_mbps=size_mb / avg_time,
            method="mmap + zero-copy",
        )
        self.results.append(result)
        return result

    def benchmark_mmap_load(self, num_runs: int = 3) -> LoadBenchResult:
        """测试 mmap 直接映射"""
        bin_path = self.output_dir / "model.bin"
        if not bin_path.exists():
            return None

        file_size = bin_path.stat().st_size
        times = []

        for _ in range(num_runs):
            self._drop_caches()

            t0 = time.perf_counter()
            state_dict = torch.load(
                bin_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            # 触发实际读取
            for v in state_dict.values():
                _ = v.sum()
            t1 = time.perf_counter()
            times.append(t1 - t0)
            del state_dict

        avg_time = sum(times) / len(times)
        size_mb = file_size / 1024 / 1024

        result = LoadBenchResult(
            format_name="PyTorch mmap",
            file_size_mb=size_mb,
            load_time_s=avg_time,
            throughput_mbps=size_mb / avg_time,
            method="torch.load(mmap=True)",
        )
        self.results.append(result)
        return result

    def _drop_caches(self):
        """清除 page cache"""
        try:
            os.system("sync")
            with open("/proc/sys/vm/drop_caches", "w") as f:
                f.write("3")
        except (PermissionError, FileNotFoundError):
            pass

    def run_all(self, model_size_mb: int = 256):
        """运行全部测试"""
        print(f"\n{'#'*50}")
        print(f"# 模型加载速度基准测试")
        print(f"# 目标模型大小: ~{model_size_mb}MB")
        print(f"{'#'*50}")

        # 创建模型
        num_layers = max(1, model_size_mb // 128)
        model = self.create_test_model(hidden_size=4096,
                                       num_layers=num_layers)

        actual_size = sum(p.numel() * p.element_size()
                         for p in model.parameters()) / 1024 / 1024
        print(f"\n模型参数大小: {actual_size:.1f}MB ({num_layers} layers)")

        # 保存各格式
        print("\n保存模型为各格式...")
        self.save_formats(model)
        del model

        # 测试加载速度
        print("\n测试加载速度...")

        print("\n--- PyTorch .bin ---")
        r = self.benchmark_pytorch_load()
        if r:
            print(f"  {r.load_time_s:.3f}s ({r.throughput_mbps:.1f} MB/s)")

        print("\n--- safetensors ---")
        r = self.benchmark_safetensors_load()
        if r:
            print(f"  {r.load_time_s:.3f}s ({r.throughput_mbps:.1f} MB/s)")

        print("\n--- PyTorch mmap ---")
        r = self.benchmark_mmap_load()
        if r:
            print(f"  {r.load_time_s:.3f}s ({r.throughput_mbps:.1f} MB/s)")

        self.print_summary()

    def print_summary(self):
        """打印结果摘要"""
        print(f"\n{'='*60}")
        print("加载速度对比")
        print(f"{'='*60}")
        print(f"{'格式':<20} {'大小(MB)':<10} {'时间(s)':<10} "
              f"{'速度(MB/s)':<12} {'方法'}")
        print("-" * 60)

        baseline = self.results[0].load_time_s if self.results else 1

        for r in self.results:
            speedup = baseline / r.load_time_s if r.load_time_s > 0 else 0
            print(f"{r.format_name:<20} {r.file_size_mb:<10.1f} "
                  f"{r.load_time_s:<10.3f} {r.throughput_mbps:<12.1f} "
                  f"{r.method} ({speedup:.1f}x)")


def main():
    parser = argparse.ArgumentParser(description="模型加载速度测试")
    parser.add_argument("--model-size", type=int, default=256,
                       help="模型大小(MB)")
    parser.add_argument("--output-dir", type=str, default="/tmp/load_bench")
    args = parser.parse_args()

    bench = ModelLoadBenchmark(args.output_dir)
    bench.run_all(args.model_size)


if __name__ == "__main__":
    main()
