# vLLM Metrics 完整指标手册

## 概述

vLLM 通过 Prometheus `/metrics` 端点暴露约 30+ 核心指标，
覆盖吞吐、延迟、调度、KV Cache、资源五大维度。

本手册按功能域详细解析每个指标的：
- 含义与计算方式
- 典型值范围（基于 8×H20 + Qwen2.5-72B 场景）
- 告警阈值建议
- 相关 PromQL 查询

---

## 1. 吞吐量指标 (Throughput)

### 1.1 vllm:generation_tokens_total

| 属性 | 值 |
|------|-----|
| 类型 | Counter |
| 含义 | 累计生成的 output token 数 |
| 标签 | `model`, `instance` |
| 典型 PromQL | `rate(vllm:generation_tokens_total[1m])` |

**典型值范围**：
- 8×H20 + Qwen2.5-72B (TP=8): 800-2000 tokens/s
- 单请求 batch: 50-100 tokens/s
- 高并发 (128 concurrent): 1500-2000 tokens/s

**告警阈值**：
- `< 100 tokens/s` 且有排队请求 → WARNING
- `= 0` 持续 30s → CRITICAL（服务可能挂了）

### 1.2 vllm:prompt_tokens_total

| 属性 | 值 |
|------|-----|
| 类型 | Counter |
| 含义 | 累计处理的 input prompt token 数 |
| 标签 | `model`, `instance` |
| 典型 PromQL | `rate(vllm:prompt_tokens_total[1m])` |

**业务含义**：
- Prompt throughput 反映 Prefill 阶段效率
- 如果 prompt 吞吐突然下降 → 可能 KV Cache 已满，Prefill 被抢占

### 1.3 vllm:request_success_total / vllm:request_failure_total

| 属性 | 值 |
|------|-----|
| 类型 | Counter |
| 含义 | 成功/失败的请求计数 |
| 典型 PromQL | `rate(vllm:request_failure_total[5m]) / rate(vllm:request_success_total[5m])` |

**告警阈值**：
- 错误率 > 1% → WARNING
- 错误率 > 5% → CRITICAL

---

## 2. 延迟指标 (Latency)

### 2.1 vllm:time_to_first_token_seconds

| 属性 | 值 |
|------|-----|
| 类型 | Histogram |
| 含义 | 从收到请求到第一个 token 生成的耗时 |
| Buckets | 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30 |
| 核心 PromQL | `histogram_quantile(0.99, rate(vllm:time_to_first_token_seconds_bucket[5m]))` |

**典型值范围（Qwen2.5-72B, TP=8）**：
- P50: 200-500ms (取决于 prompt 长度)
- P90: 500ms-2s
- P99: 1-5s
- 短 prompt (< 100 tokens): P50 < 200ms
- 长 prompt (8K+ tokens): P50 可达 2-3s

**影响因素**：
1. Prompt 长度（线性关系）
2. 排队时间（KV Cache 满时显著增大）
3. Prefix Cache 命中（命中时 TTFT 大幅缩短）
4. Tensor Parallel 通信开销

**告警阈值**：
- P99 > 5s → WARNING
- P99 > 10s → CRITICAL
- P50 > 2s → WARNING（系统整体变慢）

### 2.2 vllm:time_per_output_token_seconds

| 属性 | 值 |
|------|-----|
| 类型 | Histogram |
| 含义 | 每个 output token 的 inter-token 延迟 |
| 核心 PromQL | `histogram_quantile(0.99, rate(vllm:time_per_output_token_seconds_bucket[5m]))` |

**典型值范围**：
- P50: 15-30ms/token
- P90: 30-50ms/token
- P99: 50-100ms/token
- 高并发时 TPOT 会随 batch 增大而略微增加

**TPOT vs 用户体验**：
- < 30ms → 流畅 streaming（用户感觉 "飞快"）
- 30-60ms → 可接受
- 60-100ms → 用户开始感觉 "慢"
- > 100ms → 需要优化

### 2.3 vllm:e2e_request_latency_seconds

| 属性 | 值 |
|------|-----|
| 类型 | Histogram |
| 含义 | 端到端请求延迟（从进入队列到最后一个 token 生成） |
| 计算 | ≈ Queue Time + TTFT + (output_len × TPOT) |

**注意**：E2E latency 受 output 长度影响大，作为 SLO 不如 TTFT + TPOT 精确。

### 2.4 vllm:model_execute_time_seconds

| 属性 | 值 |
|------|-----|
| 类型 | Histogram |
| 含义 | 模型前向推理耗时（不含调度/采样开销） |
| 用途 | 隔离模型本身 vs 框架开销 |

**诊断用法**：
- `e2e_latency >> model_execute_time` → 调度/排队是瓶颈
- `e2e_latency ≈ model_execute_time` → 模型本身是瓶颈

---

## 3. 调度指标 (Scheduling)

### 3.1 vllm:num_requests_running

| 属性 | 值 |
|------|-----|
| 类型 | Gauge |
| 含义 | 当前正在执行（GPU 上活跃）的请求数 |
| 意义 | 反映 continuous batching 的实际 batch size |

**典型值范围**：
- 空闲: 0
- 正常负载: 8-64
- 高负载: 64-256（取决于 max_num_seqs 配置）

### 3.2 vllm:num_requests_waiting

| 属性 | 值 |
|------|-----|
| 类型 | Gauge |
| 含义 | 排队等待调度的请求数 |
| 意义 | > 0 表示系统已满载 |

**告警阈值**：
- > 10 持续 1 分钟 → WARNING（开始排队）
- > 50 持续 1 分钟 → CRITICAL（需要扩容）

### 3.3 vllm:num_requests_swapped

| 属性 | 值 |
|------|-----|
| 类型 | Gauge |
| 含义 | KV Cache 被 swap 到 CPU 内存的请求数 |
| 意义 | > 0 意味着 GPU 显存不够，正在 "颠簸" |

**告警**：
- swapped > 0 持续 5 分钟 → WARNING（性能已显著下降）

### 3.4 vllm:num_preemptions_total

| 属性 | 值 |
|------|-----|
| 类型 | Counter |
| 含义 | 累计 preemption（抢占）次数 |
| 意义 | Preemption = KV Cache 满，必须暂停某些请求释放空间 |

**深入理解 Preemption**：
```
正常运行: Request A, B, C 同时在 GPU 上
                     ↓ 新请求 D 进来，KV Cache 满了
Preemption: 暂停 C（swap 或 recompute），腾出空间给 D
                     ↓ 一段时间后
Resume: C 重新调度，继续生成（TTFT 增加、E2E 增加）
```

**PromQL**：
```promql
# Preemption 速率
rate(vllm:num_preemptions_total[5m])

# 有 preemption 说明 KV Cache 是瓶颈
rate(vllm:num_preemptions_total[5m]) > 0
  AND
vllm:gpu_cache_usage_perc > 0.9
```

### 3.5 vllm:num_batched_tokens

| 属性 | 值 |
|------|-----|
| 类型 | Gauge |
| 含义 | 当前 batch 中的总 token 数 |
| 意义 | 直接影响 GPU 利用率 |

---

## 4. KV Cache 指标

### 4.1 vllm:gpu_cache_usage_perc

| 属性 | 值 |
|------|-----|
| 类型 | Gauge |
| 含义 | GPU 端 KV Cache 使用率 (0-1) |
| 关键性 | **最重要的容量指标**，决定能否接受新请求 |

**典型值与行为**：
- 0-0.7: 正常，可以自由接受新请求
- 0.7-0.9: 注意，开始接近容量上限
- 0.9-0.95: 告警，可能触发 preemption
- 0.95-1.0: 危险，新请求被阻塞或旧请求被抢占

**核心关联**：
```
gpu_cache_usage_perc ↑ → preemption ↑ → TTFT ↑ → 用户体验 ↓
```

**容量规划公式**：
```
KV Cache 所需显存 (bytes) = 
    num_layers × 2(K+V) × head_dim × num_kv_heads × max_seq_len × batch_size × dtype_size

Qwen2.5-72B (TP=8, FP16):
    80 layers × 2 × 128 × (8 kv_heads / 8 TP) × seq_len × batch × 2 bytes
    = 40,960 bytes × seq_len × batch_size
    
    例: max_seq_len=32768, batch=64:
    = 40,960 × 32,768 × 64 ≈ 82 GB → 需要预留足够显存
```

### 4.2 vllm:cpu_cache_usage_perc

| 属性 | 值 |
|------|-----|
| 类型 | Gauge |
| 含义 | CPU 端 KV Cache 使用率（swap 空间） |
| 意义 | > 0 表示正在发生 CPU offload（严重性能下降） |

### 4.3 vllm:prefix_cache_hit_rate

| 属性 | 值 |
|------|-----|
| 类型 | Gauge |
| 含义 | Prefix caching 命中率 |
| 前提 | 需要 `--enable-prefix-caching` |

**业务场景**：
- System prompt 复用: hit rate 应 > 0.8
- 多轮对话: hit rate 取决于 cache 淘汰策略
- 随机独立请求: hit rate ≈ 0

**优化效果**：
- Hit rate 0.8 时, 平均 TTFT 可降低 40-60%
- 同时节省 GPU 显存（不用重复存储相同的 KV）

### 4.4 vllm:num_gpu_blocks_total / vllm:num_gpu_blocks_used

| 属性 | 值 |
|------|-----|
| 类型 | Gauge |
| 含义 | KV Cache 的 Block 级别使用情况 |
| 计算 | `gpu_cache_usage_perc ≈ num_gpu_blocks_used / num_gpu_blocks_total` |

---

## 5. 资源与系统指标

### 5.1 vllm:gpu_memory_usage_bytes

| 属性 | 值 |
|------|-----|
| 类型 | Gauge |
| 含义 | vLLM 进程占用的 GPU 显存总量 |
| 注意 | 包括 Model Weights + KV Cache + Activation + Overhead |

**显存分布（Qwen2.5-72B, TP=8, FP16）**：
```
Total per GPU: ~88 GB (of 96 GB H20)
├── Model Weights: ~9 GB (72B params / 8 TP × 2 bytes)
├── KV Cache: ~70 GB (可配置 gpu_memory_utilization)
├── Activation Memory: ~4 GB
└── Framework Overhead: ~5 GB
```

### 5.2 vllm:avg_generation_throughput_toks_per_s

| 属性 | 值 |
|------|-----|
| 类型 | Gauge |
| 含义 | 滑动窗口平均生成吞吐 |
| 用途 | 比 `rate(generation_tokens_total)` 更平滑 |

### 5.3 vllm:avg_prompt_throughput_toks_per_s

| 属性 | 值 |
|------|-----|
| 类型 | Gauge |
| 含义 | 滑动窗口平均 Prompt 处理吞吐 |
| 典型值 | 8×H20: 5000-15000 tokens/s (Prefill) |

---

## 6. 指标间关联关系

### 6.1 容量瓶颈诊断决策树

```
系统变慢?
├── TTFT 增大?
│   ├── gpu_cache_usage_perc > 0.9?
│   │   └── YES → KV Cache 满，需要扩容或减小 max_model_len
│   ├── num_requests_waiting > 0?
│   │   └── YES → 排队严重，增加实例或优化调度
│   └── prefix_cache_hit_rate 下降?
│       └── YES → Cache 被频繁淘汰，增大 cache 或调整淘汰策略
├── TPOT 增大?
│   ├── num_requests_running 增大?
│   │   └── YES → Decode batch 太大，限制 max_num_seqs
│   ├── DCGM_FI_DEV_CLOCK_THROTTLE_REASONS > 0?
│   │   └── YES → GPU 限频（温度/功率），检查散热
│   └── NVLink 带宽利用率 > 80%?
│       └── YES → TP 通信瓶颈，考虑减少 TP 或使用更长的 Pipeline
└── 吞吐量下降?
    ├── request_failure_total 增加?
    │   └── YES → 服务内部错误，查日志
    └── GPU SM Active 下降?
        └── YES → GPU 问题（ECC/限频/驱动），查 DCGM
```

### 6.2 黄金信号映射

| 黄金信号 | vLLM 指标 | SLO 建议 |
|----------|-----------|----------|
| Latency | TTFT P99 < 5s, TPOT P99 < 80ms | 按业务需求定义 |
| Traffic | generation_tokens_total rate | N/A (容量限制) |
| Errors | request_failure_total rate < 0.1% | 99.9% 成功率 |
| Saturation | gpu_cache_usage_perc < 0.9 | 留 10% 余量 |

---

## 7. 常用 PromQL 查询模板

### 7.1 实时吞吐概览

```promql
# 生成吞吐 (tokens/s) per instance
rate(vllm:generation_tokens_total[1m])

# 集群总吞吐
sum(rate(vllm:generation_tokens_total[1m]))

# 按模型分组吞吐
sum by (model) (rate(vllm:generation_tokens_total[1m]))
```

### 7.2 延迟分析

```promql
# TTFT P50 / P90 / P99
histogram_quantile(0.50, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))
histogram_quantile(0.90, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))
histogram_quantile(0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))

# TPOT P99 per instance
histogram_quantile(0.99, 
  sum by (le, instance) (rate(vllm:time_per_output_token_seconds_bucket[5m]))
)

# 延迟变化趋势（5 分钟内 P99 变化）
deriv(
  histogram_quantile(0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))[10m:]
)
```

### 7.3 KV Cache 深度分析

```promql
# Cache 使用率趋势
vllm:gpu_cache_usage_perc

# Cache 压力指数 = usage × (1 + preemption_rate)
vllm:gpu_cache_usage_perc * (1 + rate(vllm:num_preemptions_total[5m]))

# 预测 cache 何时满（线性外推）
predict_linear(vllm:gpu_cache_usage_perc[30m], 3600)

# Prefix cache 节省的计算量估算
vllm:prefix_cache_hit_rate * rate(vllm:prompt_tokens_total[5m])
```

### 7.4 调度健康度

```promql
# 调度积压比 = waiting / (running + 1)
vllm:num_requests_waiting / (vllm:num_requests_running + 1)

# Batch 效率 = 实际 batch tokens / 理论最大 batch tokens
vllm:num_batched_tokens / (vllm:num_requests_running * 2048)

# Preemption 与 TTFT 关联
rate(vllm:num_preemptions_total[5m]) > 0
  AND
histogram_quantile(0.99, rate(vllm:time_to_first_token_seconds_bucket[5m])) > 5
```

---

## 8. 指标采集最佳实践

### 8.1 Scrape 配置建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| scrape_interval | 15s | 与 DCGM 保持一致 |
| scrape_timeout | 10s | 避免超时丢数据 |
| honor_labels | true | 保留 vLLM 原始 label |
| metrics_path | /metrics | 默认路径 |

### 8.2 Label 设计

```yaml
# 建议在 relabeling 中添加的标签
- model: "qwen2.5-72b"           # 模型名
- tp_size: "8"                    # Tensor Parallel 规模
- instance_group: "inference-01"  # 实例组
- cluster: "prod-gpu-01"         # 集群
- region: "cn-east"              # 区域
```

### 8.3 Histogram Bucket 调优

vLLM 默认 bucket 在高延迟场景可能不够精确。如需自定义：
```python
# 在 vLLM 源码中调整（一般不需要）
TTFT_BUCKETS = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 10, 15, 20, 30]
TPOT_BUCKETS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.5]
```
