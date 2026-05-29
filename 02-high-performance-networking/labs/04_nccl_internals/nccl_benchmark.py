#!/usr/bin/env python3
"""NCCL Benchmark Wrapper - 封装 nccl-tests 进行集合通信性能测试"""

import subprocess
import argparse
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional


NCCL_TESTS_PATH = os.environ.get("NCCL_TESTS_PATH", "/usr/local/nccl-tests/build")

AVAILABLE_TESTS = [
    "all_reduce_perf",
    "all_gather_perf",
    "broadcast_perf",
    "reduce_perf",
    "reduce_scatter_perf",
    "sendrecv_perf",
    "alltoall_perf",
]


def parse_size(size_str: str) -> int:
    """解析大小字符串 (如 '1M', '512K', '2G') 为字节数"""
    units = {"K": 1024, "M": 1024**2, "G": 1024**3}
    match = re.match(r"^(\d+)([KMG]?)$", size_str.upper())
    if not match:
        raise ValueError(f"无效的大小格式: {size_str}")
    value, unit = int(match.group(1)), match.group(2)
    return value * units.get(unit, 1)


def run_nccl_test(test_name: str, min_bytes: int, max_bytes: int,
                  ngpus: int, nthreads: int = 1,
                  env_overrides: Optional[Dict] = None) -> Dict:
    """运行单个 nccl-test 并解析结果"""
    binary = os.path.join(NCCL_TESTS_PATH, test_name)
    if not os.path.exists(binary):
        raise FileNotFoundError(f"找不到测试二进制: {binary}")

    cmd = [
        binary,
        "-b", str(min_bytes),
        "-e", str(max_bytes),
        "-f", "2",  # factor: 每步数据大小翻倍
        "-g", str(ngpus),
        "-t", str(nthreads),
        "-w", "5",   # warmup 迭代次数
        "-n", "20",  # 测试迭代次数
    ]

    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return parse_nccl_output(result.stdout, test_name)


def parse_nccl_output(output: str, test_name: str) -> Dict:
    """解析 nccl-tests 输出，提取带宽和延迟数据"""
    results = {"test": test_name, "data_points": [], "timestamp": datetime.now().isoformat()}

    for line in output.split("\n"):
        # 匹配数据行: size count type redop root time algbw busbw
        match = re.match(
            r"\s*(\d+)\s+\d+\s+\w+\s+\w+\s+\S+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
            line
        )
        if match:
            results["data_points"].append({
                "size_bytes": int(match.group(1)),
                "time_us": float(match.group(2)),
                "algo_bw_gbps": float(match.group(3)),
                "bus_bw_gbps": float(match.group(4)),
            })
    return results


def run_benchmark_suite(tests: List[str], min_bytes: int, max_bytes: int,
                        ngpus: int, output_file: Optional[str] = None) -> List[Dict]:
    """运行一组基准测试"""
    all_results = []
    for test in tests:
        print(f"[*] 运行测试: {test}")
        try:
            result = run_nccl_test(test, min_bytes, max_bytes, ngpus)
            all_results.append(result)
            if result["data_points"]:
                max_bw = max(dp["bus_bw_gbps"] for dp in result["data_points"])
                print(f"    峰值总线带宽: {max_bw:.2f} GB/s")
            else:
                print("    警告: 未获取到数据点")
        except FileNotFoundError as e:
            print(f"    错误: {e}")
        except Exception as e:
            print(f"    异常: {e}")

    if output_file:
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n[+] 结果已保存到: {output_file}")

    return all_results


def print_summary(results: List[Dict]):
    """打印测试摘要"""
    print("\n" + "=" * 60)
    print("NCCL 基准测试摘要")
    print("=" * 60)
    for r in results:
        if r["data_points"]:
            max_bw = max(dp["bus_bw_gbps"] for dp in r["data_points"])
            min_lat = min(dp["time_us"] for dp in r["data_points"])
            print(f"  {r['test']:25s} | 峰值带宽: {max_bw:8.2f} GB/s | 最低延迟: {min_lat:8.1f} us")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="NCCL 基准测试工具")
    parser.add_argument("--test", choices=AVAILABLE_TESTS + ["all"], default="all",
                        help="要运行的测试 (默认: all)")
    parser.add_argument("--min-size", default="1M", help="最小数据大小 (默认: 1M)")
    parser.add_argument("--max-size", default="1G", help="最大数据大小 (默认: 1G)")
    parser.add_argument("--ngpus", type=int, default=0, help="GPU 数量 (0=自动检测)")
    parser.add_argument("--output", "-o", help="JSON 输出文件路径")
    args = parser.parse_args()

    ngpus = args.ngpus
    if ngpus == 0:
        try:
            result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
            ngpus = len([l for l in result.stdout.split("\n") if l.startswith("GPU")])
        except Exception:
            ngpus = 1
    print(f"[*] 使用 {ngpus} 个 GPU")

    tests = AVAILABLE_TESTS if args.test == "all" else [args.test]
    min_bytes = parse_size(args.min_size)
    max_bytes = parse_size(args.max_size)

    results = run_benchmark_suite(tests, min_bytes, max_bytes, ngpus, args.output)
    print_summary(results)


if __name__ == "__main__":
    main()
