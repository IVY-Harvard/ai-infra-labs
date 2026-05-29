"""
Lab 01: 查询 GPU 硬件参数并解读工程含义

本脚本使用 pynvml (NVIDIA Management Library) 和 torch.cuda 查询 GPU 信息。
重点不是打印数字，而是理解每个数字对 AI Infra 工作的意义。

Usage:
    pip install pynvml torch
    python query_device.py
"""

import sys
from dataclasses import dataclass
from typing import Optional


def check_dependencies():
    """检查依赖是否安装"""
    missing = []
    try:
        import pynvml
    except ImportError:
        missing.append("pynvml")
    try:
        import torch
    except ImportError:
        missing.append("torch")
    if missing:
        print(f"缺少依赖: {', '.join(missing)}")
        print(f"请运行: pip install {' '.join(missing)}")
        sys.exit(1)


@dataclass
class GPUInfo:
    """存储 GPU 信息的数据类"""
    index: int
    name: str
    uuid: str
    # 计算能力
    compute_capability: tuple
    sm_count: int
    cuda_cores: int  # 估算值
    # 内存
    total_memory_gb: float
    free_memory_gb: float
    memory_bus_width: int  # bits
    memory_clock_mhz: int
    memory_bandwidth_gb_s: float  # 估算值
    # 频率
    gpu_clock_mhz: int
    gpu_clock_max_mhz: int
    # 功耗
    power_draw_w: float
    power_limit_w: float
    # 温度
    temperature_c: int
    # PCIe
    pcie_gen: int
    pcie_width: int


def estimate_cuda_cores(sm_count: int, compute_capability: tuple) -> int:
    """
    根据 SM 数量和架构估算 CUDA Core 数量。

    不同架构每 SM 的 CUDA Core 数不同：
    - Volta (7.0): 64 FP32 cores/SM
    - Turing (7.5): 64 FP32 cores/SM
    - Ampere (8.0/8.6): 64 FP32 cores/SM (但 dual-issue 可以达到 128)
    - Hopper (9.0): 128 FP32 cores/SM
    """
    major, minor = compute_capability
    cores_per_sm = {
        (7, 0): 64,   # V100
        (7, 5): 64,   # T4
        (8, 0): 64,   # A100 (实际 FP32 throughput 等效 128)
        (8, 6): 128,  # RTX 3090
        (8, 9): 128,  # RTX 4090
        (9, 0): 128,  # H100/H20
    }
    return sm_count * cores_per_sm.get((major, minor), 64)


def estimate_memory_bandwidth(bus_width_bits: int, memory_clock_mhz: int) -> float:
    """
    估算显存带宽 (GB/s)。

    公式: 带宽 = 频率 × 位宽 × 2 (DDR) / 8 (bits to bytes)
    对于 HBM: 频率 × 位宽 × 2 / 8 / 1000 (转 GB/s)
    注意: 这是理论峰值，实际可达 ~80-90%
    """
    # HBM 的等效计算
    bandwidth_gb_s = bus_width_bits * memory_clock_mhz * 2 / 8 / 1000
    return bandwidth_gb_s


def query_gpu_with_nvml(device_id: int) -> Optional[GPUInfo]:
    """使用 pynvml 查询 GPU 信息"""
    import pynvml

    pynvml.nvmlInit()

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode('utf-8')

        uuid = pynvml.nvmlDeviceGetUUID(handle)
        if isinstance(uuid, bytes):
            uuid = uuid.decode('utf-8')

        # 内存信息
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        # 频率
        gpu_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        gpu_clock_max = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
        mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)

        # 功耗
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
        power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0

        # 温度
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

        # PCIe
        try:
            pcie_gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
            pcie_width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
        except Exception:
            pcie_gen, pcie_width = 0, 0

        # 需要 torch 获取的信息
        import torch
        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            props = torch.cuda.get_device_properties(device_id)
            compute_capability = (props.major, props.minor)
            sm_count = props.multi_processor_count
            mem_bus_width = 5120  # H20 是 5120-bit (HBM3)，这里用默认值
            # 对于某些 GPU，bus width 需要查规格表
        else:
            compute_capability = (0, 0)
            sm_count = 0
            mem_bus_width = 0

        cuda_cores = estimate_cuda_cores(sm_count, compute_capability)
        mem_bw = estimate_memory_bandwidth(mem_bus_width, mem_clock)

        return GPUInfo(
            index=device_id,
            name=name,
            uuid=uuid,
            compute_capability=compute_capability,
            sm_count=sm_count,
            cuda_cores=cuda_cores,
            total_memory_gb=mem_info.total / (1024**3),
            free_memory_gb=mem_info.free / (1024**3),
            memory_bus_width=mem_bus_width,
            memory_clock_mhz=mem_clock,
            memory_bandwidth_gb_s=mem_bw,
            gpu_clock_mhz=gpu_clock,
            gpu_clock_max_mhz=gpu_clock_max,
            power_draw_w=power,
            power_limit_w=power_limit,
            temperature_c=temp,
            pcie_gen=pcie_gen,
            pcie_width=pcie_width,
        )
    finally:
        pynvml.nvmlShutdown()


def print_gpu_info(info: GPUInfo):
    """打印 GPU 信息及工程解读"""
    print("=" * 70)
    print(f"GPU {info.index}: {info.name}")
    print("=" * 70)

    # ---- 基本信息 ----
    print(f"\n--- 基本信息 ---")
    print(f"  UUID: {info.uuid}")
    print(f"  Compute Capability: {info.compute_capability[0]}.{info.compute_capability[1]}")

    arch_names = {
        7: "Volta/Turing", 8: "Ampere/Ada", 9: "Hopper"
    }
    arch = arch_names.get(info.compute_capability[0], "Unknown")
    print(f"  架构: {arch}")

    # ---- 计算资源 ----
    print(f"\n--- 计算资源 ---")
    print(f"  SM 数量: {info.sm_count}")
    print(f"  CUDA Cores (估算): {info.cuda_cores}")

    max_threads = info.sm_count * 1536  # Hopper 每 SM 最多 1536 线程
    print(f"  最大并发线程数: {max_threads:,}")
    print(f"    ↳ 工程含义: 你的 kernel 的 grid_size × block_size 应该")
    print(f"      至少达到 {max_threads:,} 才能充分利用 GPU")

    # 估算 FP32 算力
    # FP32 TFLOPS = cores × clock × 2 (FMA) / 1e6
    fp32_tflops = info.cuda_cores * info.gpu_clock_max_mhz * 2 / 1e6
    print(f"  FP32 算力 (估算): {fp32_tflops:.1f} TFLOPS")

    # ---- 显存 ----
    print(f"\n--- 显存 ---")
    print(f"  总容量: {info.total_memory_gb:.1f} GB")
    print(f"  空闲: {info.free_memory_gb:.1f} GB")
    print(f"  已用: {info.total_memory_gb - info.free_memory_gb:.1f} GB")
    print(f"  带宽 (估算): {info.memory_bandwidth_gb_s:.0f} GB/s")

    # 估算能放多大的模型
    model_size_fp16 = info.total_memory_gb / 2 * 1e9 / 1e9  # 粗略
    print(f"    ↳ 工程含义:")
    print(f"      - 能放下约 {info.total_memory_gb / 2 * 1:.0f}B 参数的 FP16 模型")
    print(f"        (每参数 2 bytes, 加上 KV Cache 和 activation 还需更多)")
    print(f"      - 显存带宽决定了 decode 阶段的 token/s 上限")

    # ---- 频率 ----
    print(f"\n--- 频率 ---")
    print(f"  当前 GPU 频率: {info.gpu_clock_mhz} MHz")
    print(f"  最大 GPU 频率: {info.gpu_clock_max_mhz} MHz")
    print(f"  显存频率: {info.memory_clock_mhz} MHz")

    if info.gpu_clock_mhz < info.gpu_clock_max_mhz * 0.9:
        print(f"    ⚠ GPU 当前频率低于最大值的 90%，可能受功耗或温度限制")

    # ---- 功耗与温度 ----
    print(f"\n--- 功耗与温度 ---")
    print(f"  当前功耗: {info.power_draw_w:.0f} W")
    print(f"  功耗上限: {info.power_limit_w:.0f} W")
    print(f"  温度: {info.temperature_c} °C")

    if info.temperature_c > 80:
        print(f"    ⚠ 温度较高，可能触发降频")
    if info.power_draw_w > info.power_limit_w * 0.95:
        print(f"    ⚠ 接近功耗上限，可能触发 power throttling")

    # ---- PCIe ----
    print(f"\n--- PCIe ---")
    print(f"  PCIe Gen: {info.pcie_gen}")
    print(f"  PCIe Width: x{info.pcie_width}")

    pcie_bw = {3: 1.0, 4: 2.0, 5: 4.0}  # GB/s per lane (单向)
    if info.pcie_gen in pcie_bw:
        total_bw = pcie_bw[info.pcie_gen] * info.pcie_width
        print(f"  PCIe 带宽: ~{total_bw:.0f} GB/s (单向)")
        print(f"    ↳ 工程含义: Host-Device 传输瓶颈在 {total_bw:.0f} GB/s")
        print(f"      这远低于显存带宽 ({info.memory_bandwidth_gb_s:.0f} GB/s)")
        print(f"      所以要尽量避免频繁的 CPU-GPU 数据传输")

    # ---- Roofline 平衡点 ----
    print(f"\n--- Roofline 分析 ---")
    if info.memory_bandwidth_gb_s > 0:
        balance_fp32 = fp32_tflops * 1000 / info.memory_bandwidth_gb_s
        print(f"  FP32 机器平衡点: {balance_fp32:.1f} FLOP/Byte")
        print(f"    ↳ AI > {balance_fp32:.0f} 的算子是计算密集型")
        print(f"    ↳ AI < {balance_fp32:.0f} 的算子是访存密集型")
        print(f"    ↳ 大部分 element-wise 操作的 AI < 1，极度访存密集")


def query_nvlink_topology():
    """查询 NVLink 拓扑"""
    import pynvml
    pynvml.nvmlInit()

    device_count = pynvml.nvmlDeviceGetCount()
    if device_count < 2:
        print("\n只有 1 个 GPU，无法查询 NVLink 拓扑")
        return

    print("\n" + "=" * 70)
    print("NVLink / P2P 拓扑")
    print("=" * 70)

    # P2P 矩阵
    print("\nP2P 访问矩阵 (通过 CUDA):")
    import torch
    if not torch.cuda.is_available():
        print("  CUDA 不可用")
        return

    n = min(device_count, torch.cuda.device_count())
    print(f"  {'':>6}", end="")
    for j in range(n):
        print(f"  GPU{j}", end="")
    print()

    for i in range(n):
        print(f"  GPU{i}:", end="")
        for j in range(n):
            if i == j:
                print(f"    X ", end="")
            else:
                can_access = torch.cuda.can_device_access_peer(i, j)
                print(f"  {'Yes' if can_access else 'No ':>3} ", end="")
        print()

    print(f"\n  ↳ 工程含义:")
    print(f"    - P2P=Yes: 两张 GPU 可以直接互相读写显存（通过 NVLink 或 PCIe）")
    print(f"    - 这对 Tensor Parallelism 的 AllReduce 性能至关重要")
    print(f"    - NVLink 带宽 (~900 GB/s) >> PCIe (~64 GB/s)")

    pynvml.nvmlShutdown()


def query_all_gpus():
    """查询所有 GPU"""
    import pynvml
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    pynvml.nvmlShutdown()

    print(f"\n检测到 {device_count} 个 GPU\n")

    for i in range(device_count):
        info = query_gpu_with_nvml(i)
        if info:
            print_gpu_info(info)
            print()

    # 多卡拓扑
    if device_count > 1:
        query_nvlink_topology()


def main():
    check_dependencies()

    print("=" * 70)
    print("  GPU 硬件参数查询与工程解读")
    print("  Lab 01: Understanding Your Hardware")
    print("=" * 70)

    query_all_gpus()

    print("\n" + "=" * 70)
    print("学习检查清单:")
    print("=" * 70)
    print("  □ 能解释 SM 数量对 kernel 并行度的影响")
    print("  □ 理解显存带宽与算力的关系（Roofline 平衡点）")
    print("  □ 知道 H20 适合什么类型的工作负载")
    print("  □ 理解 NVLink 拓扑对多卡并行策略的影响")
    print("  □ 知道 PCIe 带宽为什么是 CPU-GPU 通信的瓶颈")


if __name__ == "__main__":
    main()
