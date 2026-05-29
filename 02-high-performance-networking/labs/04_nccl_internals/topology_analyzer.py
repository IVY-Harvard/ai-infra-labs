#!/usr/bin/env python3
"""GPU Topology Analyzer - 解析 nvidia-smi topo -m 输出并生成拓扑报告"""

import subprocess
import argparse
import json
import re
from typing import Dict, List, Tuple


# 连接类型优先级 (数值越小越快)
LINK_PRIORITY = {
    "NV12": 1, "NV10": 1, "NV8": 1, "NV6": 1, "NV4": 2, "NV3": 2,
    "NV2": 3, "NV1": 3, "PIX": 4, "PXB": 5, "PHB": 6, "SYS": 7, "X": 99
}

LINK_DESCRIPTIONS = {
    "NV": "NVLink (高速直连)",
    "PIX": "PCIe Switch 内部 (同一 PCIe switch 下)",
    "PXB": "PCIe Bridge 跨 switch (经过 PCIe bridge)",
    "PHB": "PCIe Host Bridge (经过 CPU)",
    "SYS": "系统互连 (跨 NUMA/QPI/UPI)",
    "X": "自身",
}


def get_topo_output() -> str:
    """获取 nvidia-smi topo -m 输出"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True, text=True, check=True
        )
        return result.stdout
    except FileNotFoundError:
        raise RuntimeError("nvidia-smi 未找到，请确认 NVIDIA 驱动已安装")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"nvidia-smi 执行失败: {e.stderr}")


def parse_topology(topo_output: str) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    """解析拓扑矩阵输出"""
    lines = [l.strip() for l in topo_output.strip().split("\n") if l.strip()]

    # 查找表头行 (包含 GPU0, GPU1, ... )
    header_idx = -1
    for i, line in enumerate(lines):
        if re.search(r"GPU\d", line):
            header_idx = i
            break

    if header_idx == -1:
        raise ValueError("无法解析拓扑输出: 未找到表头")

    # 解析表头获取设备列表
    headers = lines[header_idx].split()
    devices = [h for h in headers if h.startswith("GPU") or h.startswith("mlx") or h.startswith("NIC")]

    # 解析数据行
    topology = {}
    for line in lines[header_idx + 1:]:
        if not line or line.startswith("Legend"):
            break
        parts = line.split()
        if not parts or not (parts[0].startswith("GPU") or parts[0].startswith("mlx")):
            continue

        src = parts[0]
        topology[src] = {}
        for j, dst in enumerate(devices):
            if j + 1 < len(parts):
                topology[src][dst] = parts[j + 1]

    gpu_devices = [d for d in devices if d.startswith("GPU")]
    return gpu_devices, topology


def analyze_nvlink_pairs(gpus: List[str], topo: Dict) -> List[Dict]:
    """分析 NVLink 连接对"""
    nvlink_pairs = []
    for i, gpu_a in enumerate(gpus):
        for gpu_b in gpus[i+1:]:
            if gpu_a in topo and gpu_b in topo[gpu_a]:
                link = topo[gpu_a][gpu_b]
                if link.startswith("NV"):
                    nvlink_pairs.append({
                        "gpu_a": gpu_a, "gpu_b": gpu_b,
                        "link_type": link,
                        "nvlink_count": int(re.search(r"\d+", link).group()) if re.search(r"\d+", link) else 1
                    })
    return nvlink_pairs


def generate_recommendations(gpus: List[str], topo: Dict) -> List[str]:
    """根据拓扑生成优化建议"""
    recommendations = []

    # 检查是否有 NVLink
    has_nvlink = any(
        topo.get(g, {}).get(g2, "").startswith("NV")
        for g in gpus for g2 in gpus if g != g2
    )

    if has_nvlink:
        recommendations.append("检测到 NVLink 连接，建议启用 NCCL_P2P_LEVEL=NVL")
    else:
        recommendations.append("未检测到 NVLink，建议使用 PCIe P2P 或 GPUDirect RDMA")

    # 检查跨 NUMA 通信
    sys_links = sum(
        1 for g in gpus for g2 in gpus
        if g != g2 and topo.get(g, {}).get(g2, "") == "SYS"
    )
    if sys_links > 0:
        recommendations.append("存在跨 NUMA 节点 GPU 通信，建议设置 CPU 亲和性")
        recommendations.append("对于跨 NUMA 对，考虑使用 NCCL_NET_GDR_LEVEL=SYS")

    # GPU 数量建议
    if len(gpus) > 4:
        recommendations.append(f"检测到 {len(gpus)} 个 GPU，建议使用 Tree 算法: NCCL_ALGO=Tree")

    return recommendations


def print_report(gpus: List[str], topo: Dict):
    """打印拓扑分析报告"""
    print("=" * 60)
    print("GPU 拓扑分析报告")
    print("=" * 60)
    print(f"\n检测到 {len(gpus)} 个 GPU: {', '.join(gpus)}\n")

    # 打印连接矩阵
    print("连接矩阵:")
    header = f"{'':8s}" + "".join(f"{g:8s}" for g in gpus)
    print(header)
    for gpu in gpus:
        row = f"{gpu:8s}"
        for dst in gpus:
            link = topo.get(gpu, {}).get(dst, "?")
            row += f"{link:8s}"
        print(row)

    # NVLink 分析
    nvlinks = analyze_nvlink_pairs(gpus, topo)
    if nvlinks:
        print(f"\nNVLink 连接 ({len(nvlinks)} 对):")
        for pair in nvlinks:
            print(f"  {pair['gpu_a']} <-> {pair['gpu_b']}: {pair['link_type']}")

    # 建议
    recommendations = generate_recommendations(gpus, topo)
    print("\n优化建议:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="GPU 拓扑分析器")
    parser.add_argument("--input", "-i", help="从文件读取 topo 输出 (用于离线分析)")
    parser.add_argument("--output", "-o", help="输出 JSON 报告文件路径")
    args = parser.parse_args()

    if args.input:
        with open(args.input, "r") as f:
            topo_output = f.read()
    else:
        topo_output = get_topo_output()

    gpus, topo = parse_topology(topo_output)
    print_report(gpus, topo)

    if args.output:
        report = {
            "gpus": gpus,
            "topology": topo,
            "nvlink_pairs": analyze_nvlink_pairs(gpus, topo),
            "recommendations": generate_recommendations(gpus, topo),
        }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n[+] JSON 报告已保存: {args.output}")


if __name__ == "__main__":
    main()
