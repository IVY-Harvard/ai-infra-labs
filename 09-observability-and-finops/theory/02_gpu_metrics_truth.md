# 02 - GPU 指标的真相：超越 nvidia-smi

## nvidia-smi utilization 到底在测什么？

### 误导性的定义

当你运行 `nvidia-smi` 看到 `GPU-Util: 85%` 时，你可能觉得 GPU 在高效工作。

**真相**：这个数字只代表在采样周期内（默认 1 秒），有至少一个 SM（Streaming Multiprocessor）在执行至少一个 warp 的时间占比。

```
采样周期 = 1 秒
GPU-Util = (SM 有任何活动的时间) / (采样周期) × 100%

问题：
  - 96 个 SM 中只有 1 个在跑 → 也算 "active"
  - SM 跑了 1 个 warp（32 个线程）占 2048 个线程 → 也算 "active"
  - SM 在等内存而不是在计算 → 也算 "active"
```

### 实际例子

```python
# 场景 1：nvidia-smi 显示 95%，但实际效率只有 15%
# 原因：大量小 kernel 发射，SM 频繁切换但每个 kernel 只用了少量 SM
# 表现：GPU-Util 高，但 Tensor Core 几乎没被用到

# 场景 2：nvidia-smi 显示 50%，但实际计算效率很高
# 原因：大的 GEMM 操作，每次用满所有 SM 和 Tensor Core
# 但两次计算之间有数据搬运间隙
# 表现：GPU-Util 不高，但吞吐量很好

# 场景 3：nvidia-smi 显示 99%，但推理延迟很差
# 原因：KV Cache 占满显存，频繁的内存分配/释放
# SM 一直在忙，但大部分时间在做内存管理不是计算
```

### Memory Utilization 也会误导

```
nvidia-smi Memory-Usage: 75GB / 96GB

这只告诉你分配了多少显存，不告诉你：
  - 多少是 KV Cache（有效利用）
  - 多少是内存碎片（浪费）
  - 多少是 PyTorch 的 memory pool 预留（未使用但已分配）
  - 显存带宽的实际利用率
```

---

## DCGM：真正的 GPU 指标体系

### DCGM（Data Center GPU Manager）全貌

NVIDIA DCGM 是专为数据中心设计的 GPU 管理和监控工具。比 nvidia-smi 提供更精细、更准确的指标。

### 指标分类完全清单

#### 第一类：Profiling Metrics（性能剖析指标）— 最重要

```
DCGM_FI_PROF_GR_ENGINE_ACTIVE
  含义：Graphics Engine 活跃时间占比
  范围：0.0 - 1.0
  解读：类似 nvidia-smi utilization 但更精确
  注意：仍然不能反映计算效率

DCGM_FI_PROF_SM_ACTIVE
  含义：至少有一个 warp 在执行的 SM 比例的时间加权平均
  范围：0.0 - 1.0
  解读：比 GR_ENGINE_ACTIVE 更有意义
  例：0.8 表示平均 80% 的 SM 在工作

DCGM_FI_PROF_SM_OCCUPANCY ★★★
  含义：常驻 warp 数占 SM 最大支持 warp 数的比例
  范围：0.0 - 1.0
  解读：SM Occupancy 高意味着 GPU 能有效隐藏内存延迟
  关键：这是衡量 GPU 并行度利用效率的核心指标
  目标：推理场景通常 0.4-0.7 是合理范围

DCGM_FI_PROF_PIPE_TENSOR_ACTIVE ★★★★★
  含义：Tensor Core 管线活跃时间占比
  范围：0.0 - 1.0
  解读：这才是衡量 AI 计算效率的 #1 指标
  目标：Prefill 阶段应该 > 0.5，Decode 阶段通常 < 0.3
  注意：H20 的 Tensor Core 是 FP8/INT8 为主

DCGM_FI_PROF_PIPE_FP64_ACTIVE
  含义：FP64 管线活跃时间占比
  解读：AI 推理中应该接近 0（如果高了说明有问题）

DCGM_FI_PROF_PIPE_FP32_ACTIVE
  含义：FP32 管线活跃时间占比
  解读：AI 推理中应该较低（大部分应该在 Tensor Core）

DCGM_FI_PROF_PIPE_FP16_ACTIVE
  含义：FP16 管线活跃时间占比
  解读：如果模型用 FP16 推理，这个应该高于 FP32

DCGM_FI_PROF_DRAM_ACTIVE ★★★★
  含义：HBM 接口活跃时间占比（内存带宽利用率）
  范围：0.0 - 1.0
  解读：Memory-bound 操作（如 Decode）这个值会很高
  关键：H20 的 HBM 带宽是关键瓶颈指标
  目标：Decode 阶段 0.6-0.9 说明带宽充分利用

DCGM_FI_PROF_PCIE_TX_BYTES / DCGM_FI_PROF_PCIE_RX_BYTES
  含义：PCIe 发送/接收字节数
  解读：监控 CPU-GPU 数据传输，过高可能是瓶颈

DCGM_FI_PROF_NVLINK_TX_BYTES / DCGM_FI_PROF_NVLINK_RX_BYTES
  含义：NVLink 发送/接收字节数
  解读：Tensor Parallelism 场景关键，监控 GPU 间通信量
```

#### 第二类：Health Metrics（健康指标）

```
DCGM_FI_DEV_GPU_TEMP
  含义：GPU 核心温度（°C）
  告警阈值：> 80°C 警告，> 90°C 严重
  H20 特别说明：H20 TDP 400W，正常运行 65-80°C

DCGM_FI_DEV_MEMORY_TEMP
  含义：HBM 温度（°C）
  告警阈值：> 95°C 警告，> 100°C 严重（HBM 比核心耐高温）

DCGM_FI_DEV_POWER_USAGE
  含义：实时功耗（W）
  解读：功耗和计算负载成正比，可以作为利用率的间接指标
  H20 参考：空闲 ~75W，推理 ~200-350W，满载 ~400W

DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION
  含义：累计能耗（mJ）
  用途：成本计算的基础

DCGM_FI_DEV_ECC_SBE_VOL_TOTAL
  含义：单比特 ECC 错误计数（易失性，重启清零）
  解读：偶尔有是正常的（宇宙射线），持续增长需要关注

DCGM_FI_DEV_ECC_DBE_VOL_TOTAL
  含义：双比特 ECC 错误计数（不可纠正）
  告警：任何 DBE 都需要立即处理
  处理：通常需要退役该 GPU

DCGM_FI_DEV_RETIRED_SBE / DCGM_FI_DEV_RETIRED_DBE
  含义：因 ECC 错误退役的内存页数
  告警：退役页数持续增长说明 HBM 在老化

DCGM_FI_DEV_XID_ERRORS
  含义：最近的 XID 错误代码
  解读：参考 XID 错误速查表

DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL
  含义：NVLink CRC 错误计数
  告警：持续增长可能预示 NVLink 链路硬件问题

DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL
  含义：NVLink 数据 CRC 错误
  告警：比 flit 错误更严重

DCGM_FI_DEV_PCIE_REPLAY_COUNTER
  含义：PCIe 重传计数
  解读：持续增长说明 PCIe 链路质量差

DCGM_FI_DEV_GPU_UTIL / DCGM_FI_DEV_MEM_COPY_UTIL
  含义：与 nvidia-smi 相同的利用率指标
  注意：有同样的误导性问题
```

#### 第三类：Clock & Throttle Metrics

```
DCGM_FI_DEV_SM_CLOCK
  含义：SM 当前时钟频率（MHz）
  解读：如果低于最大频率，可能在被限频

DCGM_FI_DEV_MEM_CLOCK
  含义：内存时钟频率（MHz）
  解读：H20 的 HBM 频率决定了显存带宽上限

DCGM_FI_DEV_CLOCK_THROTTLE_REASONS
  含义：限频原因 bitmap
  解读：
    bit 1: GPU Idle
    bit 2: Applications Clocks Setting
    bit 4: SW Power Cap
    bit 8: HW Slowdown (温度/功耗)
    bit 16: Sync Boost
    bit 32: SW Thermal Slowdown
    bit 64: HW Thermal Slowdown
    bit 128: HW Power Brake Slowdown
```

#### 第四类：Memory Metrics

```
DCGM_FI_DEV_FB_FREE / DCGM_FI_DEV_FB_USED
  含义：Framebuffer（显存）空闲/已用量（MB）
  H20 参考：96GB HBM3

DCGM_FI_DEV_FB_TOTAL
  含义：总显存（MB）

DCGM_FI_DEV_VGPU_LICENSE_STATUS
  含义：vGPU 许可证状态（如果使用 MIG/vGPU）
```

---

## 真实 GPU 利用率：如何计算

### 计算密度效率

```python
# 真正的 GPU 利用率 = Tensor Core 活跃率 × SM 利用率
real_gpu_utilization = DCGM_FI_PROF_PIPE_TENSOR_ACTIVE * DCGM_FI_PROF_SM_ACTIVE

# 推理效率（Decode 阶段更看内存带宽）
inference_efficiency = max(
    DCGM_FI_PROF_PIPE_TENSOR_ACTIVE,  # Compute-bound 指标
    DCGM_FI_PROF_DRAM_ACTIVE           # Memory-bound 指标
) * DCGM_FI_PROF_SM_ACTIVE
```

### Prefill vs Decode 的指标差异

```
Prefill Phase (计算密集):
  ┌─────────────────────────────────────────┐
  │ Tensor Core Active:  0.6 - 0.9  ★高    │
  │ SM Occupancy:        0.5 - 0.8  ★高    │
  │ DRAM Active:         0.3 - 0.5  中等   │
  │ SM Active:           0.8 - 0.95 ★高    │
  │ 瓶颈：计算能力（TFLOPS）               │
  └─────────────────────────────────────────┘

Decode Phase (内存密集):
  ┌─────────────────────────────────────────┐
  │ Tensor Core Active:  0.1 - 0.3  低     │
  │ SM Occupancy:        0.2 - 0.5  中等   │
  │ DRAM Active:         0.6 - 0.9  ★高    │
  │ SM Active:           0.4 - 0.7  中等   │
  │ 瓶颈：HBM 带宽（GB/s）                │
  └─────────────────────────────────────────┘
```

### H20 的特殊考虑

```
H20 GPU Specs:
  - FP8 Tensor Core: 148 TFLOPS
  - HBM3: 96GB, 4TB/s bandwidth
  - TDP: 400W
  - NVLink: 900 GB/s (全互联)

关键：H20 是高带宽型 GPU（相比同代 H100 算力低但 HBM 大）
  → Decode 阶段的效率上限更高
  → 适合大 batch、长序列的推理
  → 监控重点应该放在 DRAM_ACTIVE 和 NVLink 利用率上
```

---

## 从 nvidia-smi 迁移到 DCGM 的路径

### Step 1: 安装 DCGM

```bash
# Ubuntu
sudo apt-get install -y datacenter-gpu-manager

# 或使用 Docker
docker run -d --gpus all \
  --name dcgm-exporter \
  -p 9400:9400 \
  nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.1-ubuntu22.04
```

### Step 2: 启用 Profiling Metrics

```bash
# 默认 DCGM 不启用 profiling metrics（怕影响性能）
# 实测影响 < 1%，生产环境可以放心开启

# 检查 DCGM 状态
dcgmi discovery -l

# 启用 profiling
dcgmi profile --list  # 查看可用 profiling metrics
```

### Step 3: 配置 DCGM Exporter 采集 Profiling Metrics

```csv
# /etc/dcgm-exporter/custom-counters.csv
# 格式: DCGM_FIELD_ID, Prometheus_Metric_Type, Help_String

# Profiling Metrics（关键！）
DCGM_FI_PROF_GR_ENGINE_ACTIVE, gauge, GPU engine active ratio
DCGM_FI_PROF_SM_ACTIVE, gauge, SM active ratio
DCGM_FI_PROF_SM_OCCUPANCY, gauge, SM occupancy ratio
DCGM_FI_PROF_PIPE_TENSOR_ACTIVE, gauge, Tensor Core active ratio
DCGM_FI_PROF_PIPE_FP32_ACTIVE, gauge, FP32 pipe active ratio
DCGM_FI_PROF_PIPE_FP16_ACTIVE, gauge, FP16 pipe active ratio
DCGM_FI_PROF_DRAM_ACTIVE, gauge, DRAM active ratio
DCGM_FI_PROF_PCIE_TX_BYTES, gauge, PCIe TX bytes per second
DCGM_FI_PROF_PCIE_RX_BYTES, gauge, PCIe RX bytes per second
DCGM_FI_PROF_NVLINK_TX_BYTES, gauge, NVLink TX bytes per second
DCGM_FI_PROF_NVLINK_RX_BYTES, gauge, NVLink RX bytes per second

# Health Metrics
DCGM_FI_DEV_GPU_TEMP, gauge, GPU temperature
DCGM_FI_DEV_MEMORY_TEMP, gauge, HBM temperature
DCGM_FI_DEV_POWER_USAGE, gauge, Power usage in watts
DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION, counter, Total energy in millijoules
DCGM_FI_DEV_ECC_SBE_VOL_TOTAL, counter, Single-bit ECC errors
DCGM_FI_DEV_ECC_DBE_VOL_TOTAL, counter, Double-bit ECC errors
DCGM_FI_DEV_RETIRED_SBE, counter, Retired pages due to SBE
DCGM_FI_DEV_RETIRED_DBE, counter, Retired pages due to DBE
DCGM_FI_DEV_PCIE_REPLAY_COUNTER, counter, PCIe replay count
DCGM_FI_DEV_XID_ERRORS, gauge, Last XID error

# Clock Metrics
DCGM_FI_DEV_SM_CLOCK, gauge, SM clock MHz
DCGM_FI_DEV_MEM_CLOCK, gauge, Memory clock MHz
DCGM_FI_DEV_CLOCK_THROTTLE_REASONS, gauge, Throttle reason bitmap

# Memory Metrics
DCGM_FI_DEV_FB_FREE, gauge, Free framebuffer MB
DCGM_FI_DEV_FB_USED, gauge, Used framebuffer MB
```

### Step 4: 用正确的指标替换 Dashboard

```
替换前（nvidia-smi 思维）：
  Panel: "GPU Utilization" → nvidia_smi_gpu_utilization_percentage

替换后（DCGM 思维）：
  Panel: "Compute Efficiency" → DCGM_FI_PROF_PIPE_TENSOR_ACTIVE
  Panel: "SM Utilization" → DCGM_FI_PROF_SM_ACTIVE
  Panel: "SM Occupancy" → DCGM_FI_PROF_SM_OCCUPANCY
  Panel: "Memory Bandwidth" → DCGM_FI_PROF_DRAM_ACTIVE
  Panel: "NVLink Utilization" → DCGM_FI_PROF_NVLINK_TX_BYTES + RX_BYTES
```

---

## 黄金信号：GPU 健康度评分

### 综合健康度模型

```python
def gpu_health_score(metrics: dict) -> float:
    """
    计算 GPU 综合健康度评分 (0-100)
    """
    score = 100.0

    # 温度惩罚
    if metrics['gpu_temp'] > 85:
        score -= (metrics['gpu_temp'] - 85) * 3  # 每度扣3分
    if metrics['hbm_temp'] > 95:
        score -= (metrics['hbm_temp'] - 95) * 5  # HBM 温度更严重

    # ECC 错误惩罚
    score -= metrics['ecc_sbe_rate'] * 10  # 每个 SBE/分钟 扣10分
    score -= metrics['ecc_dbe_count'] * 50  # 每个 DBE 扣50分

    # 限频惩罚
    if metrics['throttle_reasons'] & 0x68:  # Thermal/Power throttle
        score -= 20

    # PCIe 错误惩罚
    if metrics['pcie_replay_rate'] > 10:  # 每秒超过10次重传
        score -= 15

    # NVLink 错误惩罚
    if metrics['nvlink_crc_error_rate'] > 0:
        score -= 25

    # 退役页惩罚
    if metrics['retired_pages_sbe'] > 10:
        score -= 10
    if metrics['retired_pages_dbe'] > 0:
        score -= 30  # 有 DBE 退役页很危险

    return max(0, score)
```

---

## 常见误区和纠正

| 误区 | 真相 |
|------|------|
| GPU-Util 90% = GPU 很忙 | 可能只有少量 SM 在空转 |
| 显存用满 = 高效 | 可能大部分是碎片或预分配 |
| 温度低 = 没问题 | 温度过低可能说明 GPU 没在工作 |
| ECC 错误 = GPU 坏了 | SBE 是正常的（宇宙射线），只有 DBE 需要担心 |
| Power 低 = 省电好 | 功耗过低说明 GPU 没被充分利用 |
| NVLink 没流量 = 正常 | 如果在做 TP 推理，没 NVLink 流量说明配置错误 |

---

## 下一步

→ 进入 [03_inference_sli_slo.md](03_inference_sli_slo.md) 了解如何基于这些指标定义推理服务 SLI/SLO
