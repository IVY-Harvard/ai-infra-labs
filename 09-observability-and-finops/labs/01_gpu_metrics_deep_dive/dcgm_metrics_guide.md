# DCGM 指标速查手册

## Profiling Metrics（性能剖析 — 最关键）

### DCGM_FI_PROF_PIPE_TENSOR_ACTIVE

**AI 计算效率的 #1 指标。**

- 含义：Tensor Core 管线活跃时间占比
- 范围：0.0 - 1.0
- Prefill 阶段目标：> 0.5（大矩阵乘法，Tensor Core 应该很忙）
- Decode 阶段目标：< 0.3（Memory-bound，Tensor Core 活跃率自然低）
- H20 特点：FP8/INT8 Tensor Core 为主，148 TFLOPS

```promql
# Grafana 查询：Tensor Core 活跃率趋势
DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{gpu=~"$gpu"}
```

### DCGM_FI_PROF_SM_ACTIVE

- 含义：至少有 1 个 warp 在执行的 SM 占比（时间加权平均）
- 范围：0.0 - 1.0
- 目标：> 0.7（推理负载下）
- 如果 SM_ACTIVE 高但 TENSOR_ACTIVE 低：说明 SM 在做非 Tensor Core 工作（内存搬运、控制流等）

```promql
# 真正的计算效率 = Tensor Core × SM Active
DCGM_FI_PROF_PIPE_TENSOR_ACTIVE * DCGM_FI_PROF_SM_ACTIVE
```

### DCGM_FI_PROF_SM_OCCUPANCY

- 含义：常驻 warp 数 / SM 最大 warp 数
- 范围：0.0 - 1.0
- 目标：0.4 - 0.7（推理场景）
- 过低（< 0.3）：GPU 无法有效隐藏内存延迟，吞吐下降
- 过高（> 0.9）：可能寄存器或 shared memory 压力大

### DCGM_FI_PROF_DRAM_ACTIVE

- 含义：HBM 接口活跃时间占比（显存带宽利用率）
- 范围：0.0 - 1.0
- H20 关键指标：4TB/s HBM3 带宽是 H20 的核心优势
- Decode 阶段目标：0.6 - 0.9
- 如果 DRAM_ACTIVE 很低但延迟高：说明瓶颈不在显存带宽

### DCGM_FI_PROF_NVLINK_TX_BYTES / RX_BYTES

- 含义：NVLink 发送/接收字节数
- H20 NVLink 带宽：900 GB/s（全互联）
- 重要性：TP=8 推理时，GPU 间通信量决定了 all-reduce 延迟
- 如果 TP 推理但 NVLink 几乎无流量：配置可能有误

## Health Metrics（健康指标）

| 指标 | 正常范围 | 告警阈值 | 严重阈值 |
|------|---------|---------|---------|
| GPU_TEMP | 50-75°C | > 80°C | > 90°C |
| MEMORY_TEMP | 60-85°C | > 95°C | > 100°C |
| POWER_USAGE | 150-350W | > 380W | > 395W |
| ECC_SBE_RATE | 0/min | > 5/min | > 20/min |
| ECC_DBE | 0 | > 0 | - |
| PCIE_REPLAY | < 1/s | > 10/s | > 100/s |

## Clock & Throttle Metrics

```
限频原因 bitmap 解读：
  bit 1 (0x01): GPU Idle           → 正常
  bit 2 (0x02): App Clocks Setting → 检查配置
  bit 4 (0x04): SW Power Cap       → 调整功率上限
  bit 8 (0x08): HW Slowdown        → 散热问题
  bit 32 (0x20): SW Thermal        → 软件触发降温
  bit 64 (0x40): HW Thermal        → 硬件过热
  bit 128 (0x80): HW Power Brake   → 电源不足
```

## H20 特有参考值

```
正常推理负载（72B 模型 FP8, TP=8）：
  SM Active:         0.5 - 0.8
  Tensor Active:     0.2 - 0.6（取决于 Prefill/Decode 比例）
  DRAM Active:       0.5 - 0.8
  SM Occupancy:      0.4 - 0.6
  Power:             200 - 350W per GPU
  GPU Temp:          65 - 78°C
  NVLink TX+RX:      50 - 200 GB/s per GPU（TP 通信）
```
