#!/usr/bin/env python3
"""GPUDirect RDMA Bandwidth Benchmark - 测试 GPU 与 RDMA 网卡之间的直接数据传输性能"""

import subprocess
import argparse
import json
import re
import os
from typing import Dict, List, Optional, Tuple


def check_gdr_support() -> Dict[str, bool]:
    """检查 GPUDirect RDMA 支持状态"""
    checks = {
        "nvidia_peermem": False,
        "nv_peer_mem": False,
        "ib_devices": False,
        "gpu_available": False,
    }

    # 检查 nvidia-peermem 模块
    try:
        result = subprocess.run(["lsmod"], capture_output=True, text=True)
        checks["nvidia_peermem"] = "nvidia_peermem" in result.stdout
        checks["nv_peer_mem"] = "nv_peer_mem" in result.stdout
    except Exception:
        pass

    # 检查 IB 设备
    try:
        result = subprocess.run(["ibstat"], capture_output=True, text=True)
        checks["ib_devices"] = "Active" in result.stdout
    except Exception:
        pass

    # 检查 GPU
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        checks["gpu_available"] = "GPU" in result.stdout
    except Exception:
        pass

    return checks


def get_ib_devices() -> List[str]:
    """获取可用的 InfiniBand 设备列表"""
    try:
        result = subprocess.run(["ibstat", "-l"], capture_output=True, text=True)
        return [dev.strip() for dev in result.stdout.strip().split("\n") if dev.strip()]
    except Exception:
        return []


def run_perftest(test_type: str, ib_dev: str, gpu_id: int,
                 size: int, iterations: int = 1000,
                 server: Optional[str] = None) -> Dict:
    """运行 perftest 工具进行 RDMA 带宽测试"""
    cmd_map = {
        "write": "ib_write_bw",
        "read": "ib_read_bw",
        "send": "ib_send_bw",
    }
    binary = cmd_map.get(test_type, "ib_write_bw")

    cmd = [
        binary,
        "-d", ib_dev,
        "-s", str(size),
        "-n", str(iterations),
        "--use_cuda", str(gpu_id),
        "-F",  # 不使用 CPU 绑定
        "--report_gbits",
    ]

    if server:
        cmd.append(server)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return parse_perftest_output(result.stdout, test_type)
    except subprocess.TimeoutExpired:
        return {"error": "测试超时"}
    except FileNotFoundError:
        return {"error": f"未找到 {binary}，请安装 perftest 包"}


def parse_perftest_output(output: str, test_type: str) -> Dict:
    """解析 perftest 输出"""
    result = {"test_type": test_type, "data_points": []}

    for line in output.split("\n"):
        # 匹配数据行: bytes iterations bw_peak bw_avg
        match = re.match(r"\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)", line)
        if match:
            result["data_points"].append({
                "size_bytes": int(match.group(1)),
                "iterations": int(match.group(2)),
                "bw_peak_gbps": float(match.group(3)),
                "bw_avg_gbps": float(match.group(4)),
            })
    return result


def run_gdr_comparison(ib_dev: str, gpu_id: int, sizes: List[int]) -> Dict:
    """对比 GPUDirect RDMA 启用/禁用时的带宽"""
    results = {"gdr_enabled": [], "gdr_disabled": []}

    # GDR 启用测试
    print("[*] 测试 GPUDirect RDMA 启用...")
    for size in sizes:
        res = run_perftest("write", ib_dev, gpu_id, size)
        if res.get("data_points"):
            results["gdr_enabled"].append(res["data_points"][-1])

    # GDR 禁用测试 (通过环境变量)
    print("[*] 测试 GPUDirect RDMA 禁用 (走 CPU bounce buffer)...")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for size in sizes:
        res = run_perftest("write", ib_dev, -1, size)
        if res.get("data_points"):
            results["gdr_disabled"].append(res["data_points"][-1])
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    return results


def print_gdr_status(checks: Dict[str, bool]):
    """打印 GDR 支持状态"""
    print("GPUDirect RDMA 环境检查:")
    print("=" * 50)
    for key, value in checks.items():
        status = "OK" if value else "未就绪"
        print(f"  {key:20s}: [{status}]")

    gdr_ready = (checks["nvidia_peermem"] or checks["nv_peer_mem"]) and \
                checks["ib_devices"] and checks["gpu_available"]
    print(f"\n  GPUDirect RDMA 就绪: {'是' if gdr_ready else '否'}")
    if not gdr_ready:
        if not (checks["nvidia_peermem"] or checks["nv_peer_mem"]):
            print("  提示: 执行 'modprobe nvidia-peermem' 加载内核模块")
    print("=" * 50)
    return gdr_ready


def print_bandwidth_results(results: Dict):
    """打印带宽测试结果"""
    print("\n带宽测试结果 (Gbps):")
    print(f"{'Size':>12s} {'GDR On':>12s} {'GDR Off':>12s} {'加速比':>10s}")
    print("-" * 50)
    for i, (on, off) in enumerate(
        zip(results.get("gdr_enabled", []), results.get("gdr_disabled", []))
    ):
        size = on.get("size_bytes", 0)
        bw_on = on.get("bw_avg_gbps", 0)
        bw_off = off.get("bw_avg_gbps", 0)
        speedup = bw_on / bw_off if bw_off > 0 else 0
        print(f"{size:>12d} {bw_on:>12.2f} {bw_off:>12.2f} {speedup:>9.2f}x")


def main():
    parser = argparse.ArgumentParser(description="GPUDirect RDMA 带宽测试")
    parser.add_argument("--ib-dev", default="mlx5_0", help="IB 设备名 (默认: mlx5_0)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID (默认: 0)")
    parser.add_argument("--server", help="远程服务器地址 (客户端模式)")
    parser.add_argument("--check-only", action="store_true", help="仅检查环境")
    parser.add_argument("--output", "-o", help="输出 JSON 文件")
    args = parser.parse_args()

    checks = check_gdr_support()
    gdr_ready = print_gdr_status(checks)

    if args.check_only:
        return

    if not gdr_ready:
        print("\n警告: GPUDirect RDMA 环境未就绪，部分测试可能失败")

    # 测试不同大小
    sizes = [1024, 4096, 65536, 262144, 1048576, 4194304, 16777216, 67108864]
    print(f"\n[*] IB 设备: {args.ib_dev}, GPU: {args.gpu}")
    print(f"[*] 测试数据大小: {len(sizes)} 种 ({sizes[0]}B - {sizes[-1]//1024//1024}MB)")

    results = run_gdr_comparison(args.ib_dev, args.gpu, sizes)
    print_bandwidth_results(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"checks": checks, "results": results}, f, indent=2)
        print(f"\n[+] 结果已保存: {args.output}")


if __name__ == "__main__":
    main()
