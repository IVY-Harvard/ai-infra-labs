# LogQL 查询模板与故障排查手册

## 概述

本手册提供 GPU 推理场景的 LogQL 查询模板，
覆盖故障排查、性能分析、成本审计三大场景。

LogQL 核心语法:
```
{label_selector} |= "text_match" | json | line_format "{{.field}}"
```

---

## 1. GPU 硬件故障排查

### 1.1 XID Error 查询

```logql
# 所有 XID Error (严重级别)
{job="syslog"} |~ "NVRM.*Xid"

# 特定危险 XID 错误码 (31=内存页错误, 48=ECC DBE, 63=行重映射, 79=GPU掉线)
{job="syslog"} |~ "Xid.*(?:31|48|63|79)"

# XID Error 按错误码统计 (过去 1 小时)
count_over_time(
  {job="syslog"} |~ "NVRM.*Xid" | regexp `Xid.*?:\s+(?P<xid_code>\d+)` [1h]
) by (xid_code)

# 特定 GPU 的 XID Error
{job="syslog"} |~ "NVRM.*Xid" |~ "GPU 3"

# XID Error 趋势 (每 5 分钟计数)
sum(count_over_time({job="syslog"} |~ "NVRM.*Xid" [5m]))
```

### 1.2 ECC 内存错误

```logql
# ECC 错误 (SBE + DBE)
{job="syslog"} |~ "(?i)ecc.*error|SBE|DBE"

# 不可纠正 ECC 错误 (紧急!)
{job="syslog"} |~ "(?i)DBE|uncorrectable|double.bit"

# ECC 错误按节点统计
count_over_time({job="syslog"} |~ "(?i)ecc" [24h]) by (hostname)
```

### 1.3 NVLink 问题

```logql
# NVLink 错误
{job="syslog"} |~ "(?i)nvlink.*error|nvlink.*fail"

# NVLink CRC 错误
{job="syslog"} |~ "(?i)nvlink.*crc"

# NVSwitch 错误
{job="syslog"} |~ "(?i)nvswitch"
```

### 1.4 GPU 功率/温度异常

```logql
# Thermal throttle 事件
{job="syslog"} |~ "(?i)thermal.*throttl|clock.*throttl"

# GPU 掉线 (fallen off bus)
{job="syslog"} |~ "(?i)fallen off|GPU.*lost|pci.*error"
```

---

## 2. 推理服务故障排查

### 2.1 CUDA 错误

```logql
# 所有 CUDA 错误
{namespace="inference", container="vllm"} |~ "(?i)cuda.*error|cuda.*fail"

# CUDA OOM
{namespace="inference", container="vllm"} |= "CUDA out of memory"

# CUDA OOM 详情 (提取已用/总量)
{namespace="inference", container="vllm"}
  |= "CUDA out of memory"
  | regexp `(?P<allocated>[\d.]+)\s*GiB.*(?P<total>[\d.]+)\s*GiB`
  | line_format "Allocated: {{.allocated}} GiB / Total: {{.total}} GiB"

# CUDA 异步错误
{namespace="inference", container="vllm"} |~ "cudaErrorAssert|CUBLAS_STATUS"
```

### 2.2 NCCL 通信问题

```logql
# NCCL 超时 (TP 组通信卡住)
{namespace="inference", container="vllm"} |~ "(?i)nccl.*timeout"

# NCCL 初始化错误
{namespace="inference", container="vllm"} |~ "(?i)nccl.*init.*fail|nccl.*error"

# NCCL 环境变量和拓扑信息
{namespace="inference", container="vllm"} |~ "NCCL.*INFO" | head 50

# NCCL 使用的传输层 (NVLink vs PCIe)
{namespace="inference", container="vllm"} |~ "NCCL.*transport|NCCL.*NET"
```

### 2.3 vLLM 请求级日志

```logql
# 特定请求的完整日志 (通过 request_id)
{namespace="inference", container="vllm"} |~ "request_id=req-12345"

# 通过 trace_id 查找日志 (从 Jaeger 跳转过来)
{namespace="inference", container="vllm"} |~ "trace_id=abc123def456"

# 失败的请求
{namespace="inference", container="vllm", level="ERROR"} |~ "request.*fail|request.*error"

# 请求超时
{namespace="inference", container="vllm"} |~ "(?i)timeout|timed out"

# Preemption 日志
{namespace="inference", container="vllm"} |~ "(?i)preempt"

# KV Cache 相关日志
{namespace="inference", container="vllm"} |~ "(?i)kv.?cache|cache.*full|cache.*evict"
```

### 2.4 模型加载与启动

```logql
# 模型加载进度
{namespace="inference", container="vllm"} |~ "Loading model|loading weights|model.*loaded"

# 模型加载耗时
{namespace="inference", container="vllm"}
  |~ "Model.*loaded|Loading complete"
  | json
  | line_format "Load time: {{.duration}}"

# 引擎启动配置
{namespace="inference", container="vllm"} |~ "engine.*config|EngineArgs"

# Worker 初始化
{namespace="inference", container="vllm"} |~ "Worker.*init|worker.*started|rank.*init"
```

---

## 3. 性能分析

### 3.1 延迟分析

```logql
# 慢请求 (TTFT > 5s)
{namespace="inference", container="vllm"}
  | json
  | ttft_ms > 5000
  | line_format "req={{.request_id}} ttft={{.ttft_ms}}ms tokens={{.prompt_tokens}}"

# TPOT 异常 (> 100ms)
{namespace="inference", container="vllm"}
  | json
  | tpot_ms > 100
  | line_format "req={{.request_id}} tpot={{.tpot_ms}}ms batch={{.batch_size}}"

# 延迟最高的 Top 10 请求 (过去 1 小时)
topk(10,
  {namespace="inference", container="vllm"}
    | json
    | unwrap ttft_ms [1h]
) by (request_id)

# P99 TTFT 从日志计算 (与 Prometheus 指标交叉验证)
quantile_over_time(0.99,
  {namespace="inference", container="vllm"}
    | json
    | unwrap ttft_ms [5m]
)
```

### 3.2 KV Cache 压力分析

```logql
# Cache 使用率 > 90% 的日志
{namespace="inference", container="vllm"}
  | json
  | cache_usage > 0.9
  | line_format "time={{.timestamp}} cache={{.cache_usage}} waiting={{.requests_waiting}}"

# Preemption 发生时的上下文 (前后 10 行)
{namespace="inference", container="vllm"} |~ "preempt"

# Cache 满导致的拒绝
{namespace="inference", container="vllm"} |~ "(?i)cache.*full|no.*free.*block|block.*exhausted"
```

### 3.3 Batch 效率分析

```logql
# Batch size 分布
{namespace="inference", container="vllm"}
  | json
  | line_format "batch_size={{.batch_size}}"
  | regexp `batch_size=(?P<batch>\d+)`

# 大 batch (> 128) 的日志
{namespace="inference", container="vllm"}
  | json
  | batch_size > 128

# Scheduler 决策日志
{namespace="inference", container="vllm"} |~ "schedule.*step|batch.*scheduled"
```

---

## 4. Kubernetes 事件排查

### 4.1 Pod 异常

```logql
# OOM Kill 事件
{job="kubernetes-events", reason="OOMKilling"}

# Pod 重启 (CrashLoopBackOff)
{job="kubernetes-events", reason="BackOff"} |~ "inference"

# Pod 被驱逐 (节点资源不足)
{job="kubernetes-events", reason="Evicted"} |~ "inference"

# Pod 启动失败
{job="kubernetes-events", reason="Failed"} |~ "inference"
```

### 4.2 节点问题

```logql
# GPU 节点 NotReady
{job="kubernetes-events", reason=~"NodeNotReady|NodePressure"} |~ "gpu"

# 节点磁盘压力 (影响日志和 checkpoint)
{job="kubernetes-events", reason="DiskPressure"}
```

---

## 5. 安全与合规审计

### 5.1 访问审计

```logql
# API 访问日志 (按 client_id 统计)
sum by (client_id) (
  count_over_time(
    {namespace="inference", container="vllm"}
      | json
      | line_format "{{.client_id}}" [1h]
  )
)

# 异常大量请求的客户端 (DoS 检测)
# 某 client 1 分钟内请求 > 100 次
sum by (client_id) (
  count_over_time(
    {namespace="inference", container="vllm"}
      | json [1m]
  )
) > 100

# 非法输入尝试 (prompt injection 日志)
{namespace="inference", container="vllm"}
  |~ "(?i)content.*filter|safety.*block|refused|harmful"
```

---

## 6. 复合查询 (多信号关联)

### 6.1 GPU 故障 → 推理影响

```logql
# Step 1: 找到 GPU XID Error 时间点
{job="syslog"} |~ "NVRM.*Xid.*48" | line_format "XID at {{.timestamp}}"

# Step 2: 在同一时间段查看推理错误
{namespace="inference", container="vllm", level="ERROR"}

# 注意: 在 Grafana 中使用 Split View 并排展示
```

### 6.2 Preemption 根因链

```logql
# 1. 查看 preemption 日志
{namespace="inference", container="vllm"} |~ "preempt"

# 2. 查看同时间段的 KV Cache 日志
{namespace="inference", container="vllm"} |~ "cache_usage"

# 3. 查看是否有异常长的请求占用 cache
{namespace="inference", container="vllm"}
  | json
  | prompt_tokens > 8000
  | line_format "Long prompt: req={{.request_id}} tokens={{.prompt_tokens}}"
```

### 6.3 OOM 事前分析

```logql
# OOM 前 5 分钟的内存使用日志
{namespace="inference", container="vllm"}
  | json
  | line_format "gpu_mem={{.gpu_memory_used_gb}}GB cache={{.cache_usage}}"

# 同时查看 Kubernetes OOM 事件
{job="kubernetes-events", reason="OOMKilling"}
```

---

## 7. 日志量管理

### 7.1 日志量监控

```logql
# 各容器日志速率 (bytes/s)
sum by (container) (bytes_rate({namespace="inference"} [5m]))

# 各日志级别占比
sum by (level) (
  count_over_time({namespace="inference", container="vllm"} | json [1h])
)

# DEBUG 日志占比 (如果太高需要调整)
sum(count_over_time({namespace="inference", level="DEBUG"} [1h]))
/
sum(count_over_time({namespace="inference"} [1h]))
```

### 7.2 日志优化建议

| 场景 | 建议 |
|------|------|
| DEBUG 日志占 > 50% 总量 | 生产环境设置 LOG_LEVEL=INFO |
| 单个 Pod 日志 > 10 MB/min | 检查是否有循环错误 |
| trace_id 未关联 | 确认 OTel Context 注入正常 |
| syslog 无 GPU 日志 | 检查 NVIDIA Driver 日志配置 |

---

## 8. Grafana 集成

### 8.1 Explore 使用

1. 打开 Grafana → Explore
2. 选择 Loki 数据源
3. 使用上述 LogQL 查询
4. 点击日志行中的 `trace_id` → 跳转到 Jaeger 查看完整 Trace
5. 使用 "Split" 模式同时查看 Loki 日志和 Prometheus 指标

### 8.2 Dashboard 中嵌入日志面板

在 Grafana Dashboard 中添加 "Logs" 面板类型:
- 数据源选择 Loki
- 查询: `{namespace="inference", container="vllm", level=~"ERROR|WARNING"}`
- 与同一仪表盘的 Metrics 面板时间同步
- 支持点击展开日志详情

### 8.3 Derived Fields (跨信号跳转)

在 Loki 数据源设置中配置 Derived Fields:
```
Name: TraceID
Regex: trace_id=(\w+)
URL: http://jaeger:16686/trace/$${__value.raw}
Datasource: Jaeger
```

这样日志中的 trace_id 会自动变成可点击的链接，
直接跳转到 Jaeger 查看对应的 Trace。
