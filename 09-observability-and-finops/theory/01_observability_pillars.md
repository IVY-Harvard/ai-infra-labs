# 01 - 可观测性三支柱：AI Infra 场景的落地

## 从传统微服务到 AI Infra：范式转变

### 传统可观测性

在传统微服务架构中，可观测性三支柱（Metrics、Logs、Traces）已经是成熟实践：

```
Metrics: CPU/Memory/Disk/Network → Prometheus
Logs:    Application logs → ELK/Loki
Traces:  Request tracing → Jaeger/Zipkin
```

Prometheus + Grafana 是好的起点，但 AI Infra 的可观测性有本质差异。

### AI Infra 的特殊性

| 维度 | 传统微服务 | AI Infra |
|------|-----------|---------|
| 核心资源 | CPU/Memory | GPU/HBM/NVLink |
| 延迟构成 | 网络 + 计算 | Prefill + Decode + KV Cache |
| 批处理 | 通常不批处理 | Continuous Batching 是关键 |
| 资源共享 | 容易隔离 | MIG/MPS/时分复用复杂 |
| 成本量级 | $0.01/请求 | $0.001-$1/请求（差异巨大） |
| 状态管理 | 无状态为主 | KV Cache 是热状态 |
| 性能指标 | QPS/Latency | TTFT/TPOT/Throughput |
| 故障模式 | Pod 重启 | GPU 显存溢出/NVLink 故障/Thermal Throttling |

---

## 第一支柱：Metrics

### 传统 Metrics 的不足

你可能已经在采集这些指标：

```yaml
# 你可能在用的 nvidia-smi 指标
nvidia_smi_gpu_utilization_percentage    # ← 误导性指标
nvidia_smi_memory_used_bytes
nvidia_smi_temperature_celsius
nvidia_smi_power_draw_watts
```

**问题**：`gpu_utilization_percentage` 只反映 SM（Streaming Multiprocessor）在一个采样周期内"有东西在跑"的时间占比，不反映计算效率。SM 可能只跑了 1 个线程也算"活跃"。

### AI Infra Metrics 分层模型

```
Level 4: Business Metrics
  ├── cost_per_token, cost_per_request
  ├── revenue_per_gpu_hour
  └── capacity_utilization_percentage

Level 3: Service Metrics (SLI)
  ├── ttft_seconds{quantile="0.5|0.99"}
  ├── tpot_seconds{quantile="0.5|0.99"}
  ├── throughput_tokens_per_second
  ├── queue_wait_time_seconds
  └── kv_cache_hit_rate

Level 2: Infrastructure Metrics
  ├── DCGM: sm_occupancy, tensor_active, memory_throughput
  ├── NVLink: bandwidth_utilization, error_count
  ├── PCIe: throughput, replay_count
  └── Fabric Manager: topology_health

Level 1: Hardware Metrics
  ├── temperature, power_draw, ecc_errors
  ├── pcie_link_width, nvlink_link_status
  └── fan_speed, thermal_violation_count
```

### 关键 Metrics 采集架构

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  DCGM        │    │ vLLM/TGI     │    │ Custom      │
│  Exporter    │    │ /metrics     │    │ Exporters   │
│  :9400       │    │ :8000        │    │ :9xxx       │
└──────┬───────┘    └──────┬───────┘    └──────┬──────┘
       │                   │                   │
       └───────────┬───────┴───────┬───────────┘
                   │               │
            ┌──────▼──────┐ ┌─────▼──────┐
            │ Prometheus  │ │ Prometheus │
            │ (GPU pool)  │ │ (Service)  │
            └──────┬──────┘ └─────┬──────┘
                   │              │
            ┌──────▼──────────────▼──────┐
            │      Thanos / Cortex       │
            │   (Long-term + Federation) │
            └──────────────┬─────────────┘
                           │
                    ┌──────▼──────┐
                    │   Grafana   │
                    └─────────────┘
```

### 企业级 Metrics 实践

**多集群联邦**：当你有多个 GPU 集群时，单 Prometheus 不够用：

```yaml
# Thanos Sidecar 配置
apiVersion: v1
kind: ConfigMap
metadata:
  name: thanos-sidecar-config
data:
  bucket.yaml: |
    type: S3
    config:
      bucket: "gpu-metrics-longterm"
      endpoint: "s3.amazonaws.com"
      access_key: "${AWS_ACCESS_KEY}"
      secret_key: "${AWS_SECRET_KEY}"
```

**Recording Rules**：预计算复杂指标避免查询时 CPU 爆炸：

```yaml
groups:
  - name: gpu_efficiency
    interval: 30s
    rules:
      - record: gpu:sm_efficiency:ratio
        expr: |
          DCGM_FI_PROF_SM_ACTIVE / DCGM_FI_PROF_SM_OCCUPANCY
      - record: gpu:memory_bandwidth_utilization:ratio
        expr: |
          DCGM_FI_PROF_DRAM_ACTIVE
      - record: inference:cost_per_token:dollars
        expr: |
          (gpu_power_draw_watts * 0.00012)  # $/watt-hour
          / rate(vllm_tokens_generated_total[5m])
```

---

## 第二支柱：Logs

### AI Infra 日志的特殊挑战

1. **高吞吐量**：vLLM 在高 QPS 下每秒产生数千条日志
2. **GPU 内核日志**：CUDA 错误、ECC 纠正、XID 错误
3. **多层日志**：应用层 + 框架层 + 驱动层 + 硬件层
4. **结构化需求**：需要关联 request_id 到 GPU 信息

### 日志分层策略

```
Application Logs (vLLM/TGI)
  ├── request_id, model, prompt_tokens, completion_tokens
  ├── ttft, tpot, queue_time
  └── batch_size, kv_cache_usage

Framework Logs (PyTorch/CUDA)
  ├── CUDA out of memory events
  ├── NCCL communication errors
  └── cuBLAS/cuDNN warnings

Driver Logs (NVIDIA)
  ├── XID errors (critical GPU faults)
  ├── ECC single/double bit errors
  ├── Thermal throttling events
  └── NVLink errors

System Logs (kernel/dmesg)
  ├── PCIe errors
  ├── IOMMU faults
  └── Memory errors
```

### XID 错误速查（GPU 特有）

```
XID 13: Graphics Engine Exception → GPU 硬件故障，需要 RMA
XID 31: GPU memory page fault → 可能是驱动 bug 或应用 bug
XID 43: GPU stopped processing → GPU hang，需要 reset
XID 48: Double Bit ECC Error → 严重硬件故障
XID 63: ECC page retirement → HBM 坏块，需要监控坏块数量
XID 74: NVLink Error → NVLink 通道故障
XID 79: GPU has fallen off the bus → PCIe 故障，物理检查
XID 94: Contained ECC Error → ECC 纠正成功，暂时安全
```

### 结构化日志标准

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "service": "vllm-inference",
  "gpu_id": 3,
  "gpu_uuid": "GPU-abc123",
  "request_id": "req-789xyz",
  "trace_id": "trace-456",
  "span_id": "span-012",
  "event": "inference_complete",
  "model": "qwen2-72b",
  "prompt_tokens": 1024,
  "completion_tokens": 256,
  "ttft_ms": 45.2,
  "tpot_ms": 12.8,
  "batch_size": 8,
  "kv_cache_usage_pct": 72.5,
  "gpu_util_pct": 85.3,
  "gpu_memory_used_gb": 58.2
}
```

---

## 第三支柱：Traces

### 推理请求的全链路

一个推理请求经过的完整路径远比你想象的复杂：

```
Client Request
  │
  ├── [Span 1] API Gateway (认证/限流/路由)
  │     Duration: 2-5ms
  │
  ├── [Span 2] Load Balancer (选择后端实例)
  │     Duration: 0.5-2ms
  │     Attributes: selected_backend, lb_algorithm
  │
  ├── [Span 3] Request Queue (等待调度)
  │     Duration: 0-500ms (高峰期可能更长)
  │     Attributes: queue_depth, wait_time
  │
  ├── [Span 4] Tokenization
  │     Duration: 1-10ms
  │     Attributes: token_count, tokenizer
  │
  ├── [Span 5] KV Cache Lookup (Prefix Caching)
  │     Duration: 0.1-1ms
  │     Attributes: cache_hit, matched_prefix_length
  │
  ├── [Span 6] Prefill Phase (并行处理所有输入 tokens)
  │     Duration: 10-200ms
  │     Attributes: prompt_tokens, gpu_id, batch_position
  │
  ├── [Span 7] Decode Phase (逐 token 生成)
  │     Duration: 50-5000ms
  │     Attributes: generated_tokens, avg_tpot
  │
  ├── [Span 8] Detokenization
  │     Duration: 0.5-2ms
  │
  └── [Span 9] Response Streaming
        Duration: depends on network
        Attributes: chunks_sent, bytes_sent
```

### 为什么 AI 追踪和传统追踪不同

1. **长尾效应更严重**：P99 可能是 P50 的 10-50 倍（传统微服务通常 2-5 倍）
2. **批处理依赖**：你的延迟取决于你和谁一起被批处理
3. **资源竞争**：GPU 显存是硬约束，不像 CPU 可以超卖
4. **生成长度不确定**：输出 token 数量不可预知，导致延迟方差大

---

## OpenTelemetry：统一标准

### 为什么选 OpenTelemetry

```
Before OpenTelemetry:
  Metrics → Prometheus (自己的格式)
  Logs → ELK/Loki (各自格式)
  Traces → Jaeger/Zipkin (各自格式)
  关联 → 手动拼接 request_id，痛苦

After OpenTelemetry:
  Metrics + Logs + Traces → 统一 SDK
  自动关联 → trace_id 贯穿三个支柱
  Vendor-neutral → 后端可换
  Auto-instrumentation → 减少侵入
```

### AI Infra 的 OTel 语义约定（Semantic Conventions）

目前 OpenTelemetry 社区正在建立 GenAI 的语义约定：

```python
# 推理请求的标准属性
ATTR_GEN_AI_SYSTEM = "gen_ai.system"           # "vllm", "tgi", "triton"
ATTR_GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
ATTR_GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
ATTR_GEN_AI_RESPONSE_FINISH_REASON = "gen_ai.response.finish_reasons"
ATTR_GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
ATTR_GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

# GPU 特有属性（自定义扩展）
ATTR_GPU_ID = "gpu.id"
ATTR_GPU_MODEL = "gpu.model"                   # "H20"
ATTR_GPU_MEMORY_TOTAL = "gpu.memory.total"
ATTR_GPU_SM_OCCUPANCY = "gpu.sm.occupancy"
```

### 架构选型建议

```
                    ┌──────────────────────────┐
                    │    OpenTelemetry SDK      │
                    │  (Metrics+Logs+Traces)    │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │   OTel Collector          │
                    │  (采集、处理、路由)        │
                    └──┬────────┬────────┬─────┘
                       │        │        │
              ┌────────▼──┐ ┌──▼─────┐ ┌▼──────────┐
              │Prometheus │ │ Loki   │ │ Tempo/     │
              │(Metrics)  │ │(Logs)  │ │ Jaeger     │
              └────────┬──┘ └──┬─────┘ │(Traces)   │
                       │       │       └┬──────────┘
                       └───────┼────────┘
                          ┌────▼────┐
                          │ Grafana │
                          │ (统一面板)│
                          └─────────┘
```

---

## 企业级可观测性成熟度模型

### Level 1: 基础监控（你可能在这里）

- [x] Prometheus + Grafana 部署
- [x] nvidia-smi 指标采集
- [x] vLLM 基本指标展示
- [x] 基础告警（GPU 温度、显存）

### Level 2: 标准化监控

- [ ] DCGM Exporter 全量指标
- [ ] 结构化日志 + 日志聚合
- [ ] SLI/SLO 定义和跟踪
- [ ] 分级告警策略

### Level 3: 可观测性

- [ ] 分布式追踪全链路覆盖
- [ ] Metrics/Logs/Traces 关联
- [ ] 容量规划自动化
- [ ] 成本可观测性

### Level 4: 智能运维

- [ ] 异常检测（ML 驱动）
- [ ] 自动根因分析
- [ ] 自愈机制
- [ ] 预测性运维

### Level 5: FinOps 驱动

- [ ] 精确成本分摊
- [ ] ROI 可视化
- [ ] 自动化 Spot 策略
- [ ] 业务指标和技术指标关联

**本模块的目标是带你从 Level 1 到 Level 4，并触及 Level 5。**

---

## 关键原则

### 1. 先定义 SLO，再选择指标

不要"能采集什么就采集什么"，而是"SLO 需要什么就采集什么"。

### 2. 关联性 > 数据量

10 个相互关联的指标 > 100 个孤立的指标。
`trace_id` 把 Metrics、Logs、Traces 串起来是关键。

### 3. 分层采集，按需聚合

```
高频（1s）：GPU 硬件指标（温度、功耗）
中频（15s）：推理服务指标（延迟、吞吐）
低频（1m）：业务指标（成本、容量）
```

### 4. 可观测性即代码（Observability as Code）

所有 Dashboard、Alert Rule、Recording Rule 都应该在 Git 里，不要手动在 Grafana UI 上配置。

---

## 下一步

→ 进入 [02_gpu_metrics_truth.md](02_gpu_metrics_truth.md) 了解 GPU 指标的真相
