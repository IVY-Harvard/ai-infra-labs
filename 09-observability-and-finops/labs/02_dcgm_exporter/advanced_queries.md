# DCGM 高级 PromQL 查询手册

## 概述

本文档包含基于 DCGM Profiling Metrics 的高级 PromQL 查询，
覆盖真实利用率分析、负载均衡、异常检测、容量规划四大场景。

所有查询假设你已按 `dcgm_exporter_setup.yaml` 部署了 Recording Rules。

---

## 1. 真实利用率分析

### 1.1 真实计算效率 vs nvidia-smi 风格利用率

```promql
# 真实计算效率 = Tensor Core Active × SM Active
# 这才是 GPU 真正在做 AI 计算的比例
gpu:compute_efficiency:ratio

# nvidia-smi 风格利用率 ≈ SM Active
# 只告诉你 "有没有东西在跑"，不告诉你 "跑得好不好"
DCGM_FI_PROF_SM_ACTIVE

# 两者差距（"假忙" 程度）
DCGM_FI_PROF_SM_ACTIVE - gpu:compute_efficiency:ratio
```

**解读**：
- 差距越大 → GPU "looks busy but isn't really computing"
- 典型场景：CPU Offloading 不足、Tokenizer 占用 GPU 时间、Attention 实现低效

### 1.2 Prefill vs Decode 阶段识别

```promql
# Prefill 特征: Tensor Active 高, DRAM Active 中等 (Compute-bound)
DCGM_FI_PROF_PIPE_TENSOR_ACTIVE > 0.4
  AND
DCGM_FI_PROF_DRAM_ACTIVE < 0.5

# Decode 特征: Tensor Active 低, DRAM Active 高 (Memory-bound)
DCGM_FI_PROF_PIPE_TENSOR_ACTIVE < 0.2
  AND
DCGM_FI_PROF_DRAM_ACTIVE > 0.5

# Prefill/Decode 比例推断
# Tensor Active / (Tensor Active + DRAM Active) 越高 → Prefill 占比越大
DCGM_FI_PROF_PIPE_TENSOR_ACTIVE / (DCGM_FI_PROF_PIPE_TENSOR_ACTIVE + DCGM_FI_PROF_DRAM_ACTIVE)
```

### 1.3 每瓦计算效率（能效比）

```promql
# 每瓦 Tensor Core 利用率
DCGM_FI_PROF_PIPE_TENSOR_ACTIVE / DCGM_FI_DEV_POWER_USAGE

# 集群平均能效比
avg(DCGM_FI_PROF_PIPE_TENSOR_ACTIVE / DCGM_FI_DEV_POWER_USAGE)

# 找出能效最差的 GPU（同样的功耗但计算效率最低）
bottomk(3,
  DCGM_FI_PROF_PIPE_TENSOR_ACTIVE / DCGM_FI_DEV_POWER_USAGE
)
```

---

## 2. 多 GPU 负载均衡分析

### 2.1 GPU 间负载方差

```promql
# SM Active 的标准差（跨所有 GPU）
# 值越大 → 负载越不均衡
stddev(DCGM_FI_PROF_SM_ACTIVE)

# 变异系数 (CV = stddev/mean)
# > 0.2 需要关注, > 0.3 需要调查
cluster:gpu_load_imbalance:ratio
```

### 2.2 找出最闲和最忙的 GPU

```promql
# 最忙的 3 张 GPU
topk(3, DCGM_FI_PROF_SM_ACTIVE)

# 最闲的 3 张 GPU
bottomk(3, DCGM_FI_PROF_SM_ACTIVE)

# 最忙与最闲的差距
max(DCGM_FI_PROF_SM_ACTIVE) - min(DCGM_FI_PROF_SM_ACTIVE)
```

### 2.3 Tensor Parallel 通信分析

```promql
# 每张 GPU 的 NVLink 总带宽
gpu:nvlink_bandwidth:bytes_per_sec

# 所有 GPU NVLink 带宽是否均匀（TP 场景应该大致相同）
stddev(gpu:nvlink_bandwidth:bytes_per_sec) /
  avg(gpu:nvlink_bandwidth:bytes_per_sec)

# NVLink 带宽利用率（相对于 H20 的 900 GB/s）
gpu:nvlink_bandwidth:bytes_per_sec / 900e9

# 找出 NVLink 带宽异常低的 GPU（可能 TP 配置错误）
gpu:nvlink_bandwidth:bytes_per_sec < 1e9
  AND
DCGM_FI_PROF_SM_ACTIVE > 0.3
```

---

## 3. 异常检测查询

### 3.1 SM Active 突变检测

```promql
# SM Active 5 分钟内变化率
deriv(DCGM_FI_PROF_SM_ACTIVE[5m])

# 检测突然下跌（可能是服务崩溃或流量骤降）
deriv(DCGM_FI_PROF_SM_ACTIVE[5m]) < -0.1

# 对比 5 分钟均值和当前值，偏差超过 2 个标准差
(DCGM_FI_PROF_SM_ACTIVE - avg_over_time(DCGM_FI_PROF_SM_ACTIVE[5m]))
  /
stddev_over_time(DCGM_FI_PROF_SM_ACTIVE[30m])
```

### 3.2 温度异常趋势

```promql
# 温度上升趋势（每分钟上升速率）
deriv(DCGM_FI_DEV_GPU_TEMP[10m]) * 60

# 温度上升速率 > 1°C/min 且已高于 80°C
deriv(DCGM_FI_DEV_GPU_TEMP[10m]) * 60 > 1
  AND
DCGM_FI_DEV_GPU_TEMP > 80

# GPU 温度与 HBM 温度差异（正常应在 10-20°C 以内）
DCGM_FI_DEV_MEMORY_TEMP - DCGM_FI_DEV_GPU_TEMP > 25
```

### 3.3 ECC 错误趋势

```promql
# 单比特 ECC 错误增长率
rate(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL[5m]) * 60

# ECC 错误率 Z-Score（相对于历史均值）
(
  rate(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL[5m])
  - avg_over_time(rate(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL[5m])[1h:])
) / stddev_over_time(rate(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL[5m])[1h:])
```

### 3.4 限频事件关联分析

```promql
# 正在限频的 GPU
DCGM_FI_DEV_CLOCK_THROTTLE_REASONS > 1

# 限频期间的性能下降
DCGM_FI_PROF_PIPE_TENSOR_ACTIVE
  AND
DCGM_FI_DEV_CLOCK_THROTTLE_REASONS > 1

# 限频前后 SM Clock 对比
DCGM_FI_DEV_SM_CLOCK
  AND
DCGM_FI_DEV_CLOCK_THROTTLE_REASONS > 1
```

---

## 4. 容量规划查询

### 4.1 GPU 闲置时间占比

```promql
# GPU 闲置比例（SM Active < 5%）
count(DCGM_FI_PROF_SM_ACTIVE < 0.05) / count(DCGM_FI_PROF_SM_ACTIVE)

# 过去 24 小时平均闲置时间（小时）
avg_over_time(
  (DCGM_FI_PROF_SM_ACTIVE < bool 0.05)[24h:]
) * 24

# 过去 1 周 GPU 利用率分布（P50 / P90 / P99）
histogram_quantile(0.50, rate(DCGM_FI_PROF_SM_ACTIVE[1w]))
histogram_quantile(0.90, rate(DCGM_FI_PROF_SM_ACTIVE[1w]))
histogram_quantile(0.99, rate(DCGM_FI_PROF_SM_ACTIVE[1w]))
```

### 4.2 显存容量规划

```promql
# 显存使用率分布
gpu:fb_utilization:ratio

# 显存使用率 > 90% 的 GPU 数量
count(gpu:fb_utilization:ratio > 0.9)

# 显存增长趋势（预测何时耗尽）
predict_linear(DCGM_FI_DEV_FB_USED[6h], 3600 * 24)

# 显存使用量与 KV Cache 大小的关系
# （需要 vLLM metrics 配合）
DCGM_FI_DEV_FB_USED
  AND ON(instance)
vllm:gpu_cache_usage_perc
```

### 4.3 功耗与成本估算

```promql
# 当前集群总功耗 (kW)
sum(DCGM_FI_DEV_POWER_USAGE) / 1000

# 过去 24 小时总能耗 (kWh)
sum(avg_over_time(DCGM_FI_DEV_POWER_USAGE[24h])) / 1000 * 24

# 按 GPU 排列的功耗排行
topk(8, DCGM_FI_DEV_POWER_USAGE)

# 功耗与计算效率散点（找出高功耗低效率的 GPU）
DCGM_FI_DEV_POWER_USAGE > 300
  AND
gpu:compute_efficiency:ratio < 0.1
```

---

## 5. Grafana Dashboard 变量

在 Grafana Dashboard 中配置以下变量以支持交互式查询：

```
# GPU 选择器
label_values(DCGM_FI_PROF_SM_ACTIVE, gpu)

# 节点选择器
label_values(DCGM_FI_PROF_SM_ACTIVE, instance)

# 集群选择器
label_values(DCGM_FI_PROF_SM_ACTIVE, cluster)
```

## 6. 常用 Grafana Panel 配置

### 6.1 真实利用率 vs 表面利用率

```
Panel Type: Time Series
Query A: DCGM_FI_PROF_SM_ACTIVE{gpu=~"$gpu"} (Legend: nvidia-smi style)
Query B: gpu:compute_efficiency:ratio{gpu=~"$gpu"} (Legend: Real compute efficiency)
Y-Axis: 0 - 1 (Percentage)
Color: A=Yellow, B=Green (差距用红色填充)
```

### 6.2 四维雷达图

```
Panel Type: Stat / Gauge
Query 1: gpu:sm_active:avg5m{gpu="$gpu"}           (Compute)
Query 2: gpu:dram_active:avg5m{gpu="$gpu"}          (Memory BW)
Query 3: gpu:sm_occupancy:avg5m{gpu="$gpu"}         (Parallelism)
Query 4: gpu:nvlink_bandwidth:bytes_per_sec{gpu="$gpu"} / 900e9  (Communication)
Thresholds: 0.3=Red, 0.5=Yellow, 0.7=Green
```

### 6.3 GPU 健康热力图

```
Panel Type: Heatmap
Query: DCGM_FI_DEV_GPU_TEMP{gpu=~".*"}
X-Axis: Time
Y-Axis: GPU ID
Color: 50=Blue, 70=Green, 80=Yellow, 90=Red
```
