"""
Lab 05 - GPU 拓扑感知映射
==========================
检测 GPU 间的互联类型（NVLink / PCIe），自动推荐 3D 并行配置。

核心原则:
  - TP 放在 NVLink 互联的 GPU 组内（带宽要求最高）
  - PP 放在 NVLink 组间 / 跨机（P2P 通信量小）
  - DP 放在最外层（一步一次，可 overlap）

运行:
    torchrun --nproc_per_node=8 topology_mapper.py
"""

import os
import subprocess
import torch
import torch.distributed as dist


def get_gpu_topology():
    """
    通过 nvidia-smi 获取 GPU 拓扑信息。
    返回: NVLink 连接矩阵
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def detect_nvlink_groups(num_gpus: int):
    """
    检测 NVLink 互联的 GPU 组。

    H20 典型拓扑: 8 GPU 全互联 NVLink
    某些机器: 4+4 NVLink (两个 NVSwitch 组)

    简化检测: 通过 CUDA API 测量 P2P 带宽
    """
    groups = []
    visited = set()

    # 简化: 假设能 P2P 访问的 GPU 在同一 NVLink 组
    for i in range(num_gpus):
        if i in visited:
            continue
        group = [i]
        visited.add(i)
        for j in range(i + 1, num_gpus):
            if j in visited:
                continue
            # 检查 P2P 可达性
            can_access = torch.cuda.can_device_access_peer(i, j)
            if can_access:
                group.append(j)
                visited.add(j)
        groups.append(group)

    return groups


def recommend_config(num_gpus: int, model_size_b: float, nvlink_groups: list):
    """
    根据 GPU 数量、模型大小和拓扑推荐 3D 并行配置。
    """
    nvlink_group_size = len(nvlink_groups[0]) if nvlink_groups else num_gpus

    recommendations = []

    if model_size_b <= 3:
        # 小模型: 纯 DDP
        recommendations.append({
            "config": f"TP=1, PP=1, DP={num_gpus}",
            "reason": "模型小于 3B，DDP 足够，最大化吞吐",
        })

    if model_size_b <= 7:
        # 中等模型
        tp = min(4, nvlink_group_size)
        dp = num_gpus // tp
        recommendations.append({
            "config": f"TP={tp}, PP=1, DP={dp}",
            "reason": f"TP={tp} 切分参数，DP={dp} 扩展吞吐",
        })

    if model_size_b <= 13:
        tp = min(4, nvlink_group_size)
        pp = 2
        dp = num_gpus // (tp * pp)
        if dp >= 1:
            recommendations.append({
                "config": f"TP={tp}, PP={pp}, DP={dp}",
                "reason": f"TP 放 NVLink 内，PP 分两个 stage",
            })

    if model_size_b > 13:
        tp = min(8, nvlink_group_size)
        pp = max(2, num_gpus // tp)
        dp = num_gpus // (tp * pp)
        dp = max(1, dp)
        recommendations.append({
            "config": f"TP={tp}, PP={pp}, DP={dp}",
            "reason": "超大模型需要 TP + PP 组合",
        })

    # ZeRO 替代方案
    recommendations.append({
        "config": f"ZeRO-3 on {num_gpus} GPUs",
        "reason": "不需要改模型代码，FSDP FULL_SHARD 等效",
    })

    return recommendations


def measure_p2p_bandwidth(rank, world_size, device):
    """测量 GPU 间 P2P 带宽"""
    if rank >= 2:
        return  # 只测 GPU 0 和 GPU 1

    size_mb = 64
    data = torch.randn(size_mb * 1024 * 1024 // 4, device=device)
    warmup = 3
    n_iters = 10

    if rank == 0:
        # 预热
        for _ in range(warmup):
            dist.send(data, dst=1)
        torch.cuda.synchronize()

        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(n_iters):
            dist.send(data, dst=1)
        t1.record()
        torch.cuda.synchronize()

        time_ms = t0.elapsed_time(t1) / n_iters
        bw = size_mb / (time_ms / 1000) / 1024  # GB/s
        print(f"\n  P2P 带宽 (GPU 0 → GPU 1): {bw:.1f} GB/s ({time_ms:.2f} ms for {size_mb}MB)")

    elif rank == 1:
        for _ in range(warmup):
            dist.recv(data, src=0)
        torch.cuda.synchronize()
        for _ in range(n_iters):
            dist.recv(data, src=0)
        torch.cuda.synchronize()


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print("=" * 60)
        print("GPU 拓扑感知映射")
        print("=" * 60)

        # GPU 信息
        print(f"\nGPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_mem / 1e9:.0f} GB)")

        # 拓扑
        topo = get_gpu_topology()
        if topo:
            print(f"\n--- nvidia-smi topo -m ---")
            # 只打印前几行
            lines = topo.strip().split('\n')
            for line in lines[:12]:
                print(f"  {line}")

        # NVLink 组检测
        nvlink_groups = detect_nvlink_groups(torch.cuda.device_count())
        print(f"\nNVLink 组: {nvlink_groups}")

        # 推荐配置
        for model_size in [3, 7, 13, 70]:
            print(f"\n--- {model_size}B 模型推荐 ---")
            recs = recommend_config(world_size, model_size, nvlink_groups)
            for i, rec in enumerate(recs):
                marker = "★" if i == 0 else " "
                print(f"  {marker} {rec['config']}")
                print(f"    {rec['reason']}")

    dist.barrier()

    # P2P 带宽测试
    if rank == 0:
        print(f"\n--- P2P 带宽测试 ---")
    measure_p2p_bandwidth(rank, world_size, device)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
