# 04 - AI 场景的分布式追踪

## 为什么推理服务需要分布式追踪

### 监控 vs 追踪

```
监控（Metrics）告诉你：
  "TTFT P99 从 200ms 涨到 800ms 了"

追踪（Traces）告诉你：
  "这个请求慢是因为排队等了 500ms，因为当时有一个 100K token 的长请求
   占着 GPU，而它被调度到 GPU-3 是因为 GPU-0/1/2 的 KV Cache 已满"
```

### 推理服务的追踪复杂性

传统微服务追踪：A 调 B，B 调 C，链路清晰。

推理服务追踪：
1. **Continuous Batching**：你的请求和其他请求被动态组 batch，延迟互相影响
2. **Tensor Parallelism**：一个请求分布在多张 GPU 上并行计算
3. **Prefix Caching**：KV Cache 命中与否导致截然不同的路径
4. **Preemption**：低优先级请求可能被抢占再恢复
5. **Streaming**：响应是逐 token 流式返回的

---

## 推理请求全链路追踪设计

### 完整 Span 层次结构

```
[Root Span] inference_request (trace_id=abc123)
│
├── [Span] api_gateway
│   ├── attribute: client_ip, api_key_hash
│   ├── attribute: rate_limit_remaining
│   └── event: authentication_success
│
├── [Span] load_balancer
│   ├── attribute: algorithm="least_connections"
│   ├── attribute: selected_backend="vllm-worker-2"
│   ├── attribute: candidate_backends=4
│   └── event: backend_selected
│
├── [Span] request_queue
│   ├── attribute: queue_depth_on_arrival=12
│   ├── attribute: estimated_wait_ms=50
│   ├── attribute: priority="normal"
│   └── event: dequeued
│
├── [Span] tokenization
│   ├── attribute: tokenizer="qwen2"
│   ├── attribute: prompt_tokens=2048
│   └── attribute: special_tokens_added=3
│
├── [Span] kv_cache_lookup
│   ├── attribute: prefix_cache_enabled=true
│   ├── attribute: prefix_matched_tokens=512
│   ├── attribute: cache_hit_ratio=0.25
│   └── event: partial_cache_hit
│
├── [Span] scheduler
│   ├── attribute: batch_size=6
│   ├── attribute: batch_total_tokens=15360
│   ├── attribute: gpu_memory_available_pct=42
│   ├── attribute: scheduling_policy="fcfs"
│   └── event: added_to_batch
│
├── [Span] prefill
│   ├── attribute: tokens_to_process=1536  (2048-512 cached)
│   ├── attribute: gpu_ids=[0,1,2,3,4,5,6,7]  (TP=8)
│   ├── attribute: compute_time_ms=45
│   ├── attribute: tensor_core_utilization=0.72
│   └── link: batch_span_id  (指向同 batch 其他请求)
│
├── [Span] decode_loop
│   ├── attribute: total_tokens_generated=256
│   ├── attribute: avg_tpot_ms=28
│   ├── attribute: min_tpot_ms=22
│   ├── attribute: max_tpot_ms=65
│   ├── attribute: batch_size_changes=[6,7,5,6]
│   ├── attribute: preempted=false
│   ├── attribute: dram_utilization_avg=0.78
│   │
│   ├── [Span] decode_step_1
│   │   ├── attribute: batch_size=6
│   │   ├── attribute: step_time_ms=28
│   │   └── attribute: token_id=15234
│   │   ... (通常不追踪每一步，除非做性能分析)
│   │
│   └── event: eos_token_generated
│
├── [Span] detokenization
│   ├── attribute: output_tokens=256
│   ├── attribute: output_text_length=1024
│   └── attribute: detokenize_time_ms=1.2
│
└── [Span] response_streaming
    ├── attribute: chunks_sent=64
    ├── attribute: total_bytes=4096
    ├── attribute: stream_duration_ms=7168
    └── event: stream_complete
```

### Span 之间的关系

```
ChildOf 关系：
  inference_request → api_gateway → load_balancer → ...

FollowsFrom 关系：
  request_queue → scheduler（排队结束后才调度）

Links（关联）：
  prefill → batch_span（同 batch 的其他请求）
  decode_step → gpu_metrics_snapshot（GPU 指标快照）
```

---

## OpenTelemetry 集成实现

### 基础 SDK 设置

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

# 资源定义
resource = Resource.create({
    "service.name": "vllm-inference",
    "service.version": "0.4.0",
    "deployment.environment": "production",
    "gpu.count": 8,
    "gpu.model": "H20",
    "model.name": "qwen2-72b",
    "host.name": socket.gethostname(),
})

# Tracer Provider
provider = TracerProvider(resource=resource)

# OTLP Exporter（发送到 Jaeger/Tempo）
exporter = OTLPSpanExporter(
    endpoint="http://tempo:4317",
    insecure=True,
)

# Batch Processor（批量发送，减少开销）
processor = BatchSpanProcessor(
    exporter,
    max_queue_size=2048,
    max_export_batch_size=512,
    schedule_delay_millis=5000,
)

provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("vllm-inference", "1.0.0")
```

### 推理请求追踪中间件

```python
import time
from opentelemetry import trace, context
from opentelemetry.trace import StatusCode, SpanKind

tracer = trace.get_tracer("inference-service")

class InferenceTracer:
    """推理请求全链路追踪器"""

    def trace_request(self, request):
        """追踪完整的推理请求"""
        with tracer.start_as_current_span(
            "inference_request",
            kind=SpanKind.SERVER,
            attributes={
                "gen_ai.system": "vllm",
                "gen_ai.request.model": request.model,
                "gen_ai.request.max_tokens": request.max_tokens,
                "gen_ai.request.temperature": request.temperature,
            }
        ) as root_span:

            try:
                # 1. Tokenization
                tokens = self._trace_tokenization(request)

                # 2. KV Cache Lookup
                cached_prefix = self._trace_cache_lookup(tokens)

                # 3. Scheduling
                batch_info = self._trace_scheduling(tokens, cached_prefix)

                # 4. Prefill
                self._trace_prefill(tokens, cached_prefix, batch_info)

                # 5. Decode
                output = self._trace_decode(batch_info)

                # 6. Detokenization
                text = self._trace_detokenization(output)

                # 设置最终属性
                root_span.set_attribute("gen_ai.usage.input_tokens", len(tokens))
                root_span.set_attribute("gen_ai.usage.output_tokens", len(output))
                root_span.set_attribute("gen_ai.response.finish_reasons", ["stop"])

                return text

            except Exception as e:
                root_span.set_status(StatusCode.ERROR, str(e))
                root_span.record_exception(e)
                raise

    def _trace_tokenization(self, request):
        with tracer.start_as_current_span("tokenization") as span:
            start = time.perf_counter()
            tokens = self.tokenizer.encode(request.prompt)
            duration = (time.perf_counter() - start) * 1000

            span.set_attribute("tokenizer.name", self.tokenizer.name)
            span.set_attribute("tokenizer.prompt_tokens", len(tokens))
            span.set_attribute("tokenizer.duration_ms", duration)

            return tokens

    def _trace_cache_lookup(self, tokens):
        with tracer.start_as_current_span("kv_cache_lookup") as span:
            matched = self.prefix_cache.lookup(tokens)

            span.set_attribute("cache.enabled", True)
            span.set_attribute("cache.matched_tokens", matched)
            span.set_attribute("cache.hit_ratio", matched / len(tokens))
            span.add_event("cache_lookup_complete", {
                "hit": matched > 0,
                "matched_tokens": matched,
            })

            return matched

    def _trace_scheduling(self, tokens, cached_prefix):
        with tracer.start_as_current_span("scheduler") as span:
            queue_entry_time = time.perf_counter()

            # 等待调度
            batch_info = self.scheduler.schedule(tokens, cached_prefix)

            queue_wait = (time.perf_counter() - queue_entry_time) * 1000

            span.set_attribute("scheduler.queue_wait_ms", queue_wait)
            span.set_attribute("scheduler.batch_size", batch_info.batch_size)
            span.set_attribute("scheduler.gpu_ids", batch_info.gpu_ids)
            span.set_attribute("scheduler.policy", "fcfs")

            return batch_info

    def _trace_prefill(self, tokens, cached_prefix, batch_info):
        with tracer.start_as_current_span("prefill") as span:
            tokens_to_compute = len(tokens) - cached_prefix
            start = time.perf_counter()

            self.engine.prefill(tokens, cached_prefix, batch_info)

            duration = (time.perf_counter() - start) * 1000

            span.set_attribute("prefill.tokens_computed", tokens_to_compute)
            span.set_attribute("prefill.duration_ms", duration)
            span.set_attribute("prefill.tokens_per_second",
                             tokens_to_compute / (duration / 1000))

    def _trace_decode(self, batch_info):
        with tracer.start_as_current_span("decode_loop") as span:
            output_tokens = []
            tpot_values = []
            start = time.perf_counter()

            for token in self.engine.decode_stream(batch_info):
                step_time = time.perf_counter()
                output_tokens.append(token)

                if len(output_tokens) > 1:
                    tpot = (step_time - prev_step_time) * 1000
                    tpot_values.append(tpot)

                prev_step_time = step_time

            total_time = (time.perf_counter() - start) * 1000

            span.set_attribute("decode.total_tokens", len(output_tokens))
            span.set_attribute("decode.total_time_ms", total_time)
            span.set_attribute("decode.avg_tpot_ms",
                             sum(tpot_values) / len(tpot_values) if tpot_values else 0)
            span.set_attribute("decode.min_tpot_ms", min(tpot_values) if tpot_values else 0)
            span.set_attribute("decode.max_tpot_ms", max(tpot_values) if tpot_values else 0)

            return output_tokens

    def _trace_detokenization(self, tokens):
        with tracer.start_as_current_span("detokenization") as span:
            start = time.perf_counter()
            text = self.tokenizer.decode(tokens)
            duration = (time.perf_counter() - start) * 1000

            span.set_attribute("detokenize.tokens", len(tokens))
            span.set_attribute("detokenize.text_length", len(text))
            span.set_attribute("detokenize.duration_ms", duration)

            return text
```

---

## Trace 采样策略

### 为什么需要采样

```
8 张 GPU，QPS = 50 req/s
每个请求 ~10 个 span
= 500 spans/s = 43M spans/day

全量采集问题：
  - 存储成本巨大（每天数 GB traces）
  - 对性能有影响（序列化 + 网络传输）
  - 大部分 trace 是 "正常的"，分析价值低
```

### 采样策略

```python
from opentelemetry.sdk.trace.sampling import (
    TraceIdRatioBased,
    ParentBased,
)

# 策略 1: 基于概率采样（简单但会丢掉有价值的 trace）
sampler = TraceIdRatioBased(0.1)  # 10% 采样

# 策略 2: 基于父级（保持 trace 完整性）
sampler = ParentBased(root=TraceIdRatioBased(0.1))

# 策略 3: 尾部采样（推荐）— 在 OTel Collector 中实现
# 先 100% 采集，看完整个 trace 再决定是否保留
# 保留条件：有错误、延迟异常、特定标签
```

### OTel Collector 尾部采样配置

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317

processors:
  # 尾部采样：基于完整 trace 决定是否保留
  tail_sampling:
    decision_wait: 10s  # 等待 10s 收集完整 trace
    num_traces: 100000
    expected_new_traces_per_sec: 100
    policies:
      # 始终保留：有错误的 trace
      - name: errors
        type: status_code
        status_code:
          status_codes: [ERROR]

      # 始终保留：延迟超过 SLO 的 trace
      - name: high-latency
        type: latency
        latency:
          threshold_ms: 1000

      # 始终保留：特定标签（VIP 客户）
      - name: vip-customers
        type: string_attribute
        string_attribute:
          key: customer.tier
          values: ["enterprise", "vip"]

      # 对正常 trace 进行概率采样
      - name: normal-sampling
        type: probabilistic
        probabilistic:
          sampling_percentage: 5  # 正常请求只保留 5%

exporters:
  otlp:
    endpoint: tempo:4317
    tls:
      insecure: true

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [tail_sampling]
      exporters: [otlp]
```

---

## Jaeger / Tempo 部署与查询

### Grafana Tempo 部署（推荐）

```yaml
# tempo-config.yaml
server:
  http_listen_port: 3200

distributor:
  receivers:
    otlp:
      protocols:
        grpc:
          endpoint: 0.0.0.0:4317

storage:
  trace:
    backend: s3
    s3:
      bucket: inference-traces
      endpoint: s3.amazonaws.com
      region: us-east-1
    wal:
      path: /var/tempo/wal
    block:
      bloom_filter_false_positive: 0.05
      v2_index_downsample_bytes: 1000
      v2_encoding: zstd

  # 保留策略
  compactor:
    compaction:
      block_retention: 720h  # 30 天

# 查询优化
querier:
  max_concurrent_queries: 20
  search:
    prefer_self: 10  # 优先本地查询

# 指标生成（从 traces 自动产出 RED metrics）
metrics_generator:
  registry:
    external_labels:
      source: tempo
  storage:
    path: /var/tempo/generator/wal
    remote_write:
      - url: http://prometheus:9090/api/v1/write
  traces_storage:
    path: /var/tempo/generator/traces
  processor:
    service_graphs:
      dimensions:
        - gen_ai.request.model
        - gpu.model
    span_metrics:
      dimensions:
        - gen_ai.request.model
        - gen_ai.response.finish_reasons
```

### TraceQL 查询示例

```
# 查找 TTFT > 1s 的请求
{ span.name = "inference_request" && duration > 1s }

# 查找使用特定模型的慢请求
{ span.gen_ai.request.model = "qwen2-72b" && span.name = "prefill" && duration > 500ms }

# 查找 KV Cache 命中率低的请求
{ span.name = "kv_cache_lookup" && span.cache.hit_ratio < 0.1 }

# 查找被抢占的请求
{ span.name = "decode_loop" && span.decode.preempted = true }

# 查找有错误的 trace
{ status = error }

# 按模型聚合延迟（需要 Tempo metrics generator）
{ span.name = "inference_request" } | rate() by (span.gen_ai.request.model)
```

---

## Traces + Metrics + Logs 关联

### 通过 Exemplars 实现 Metrics → Traces

```python
# 在 Prometheus 指标中嵌入 trace_id
from prometheus_client import Histogram

ttft_histogram = Histogram(
    'inference_ttft_seconds',
    'Time to first token',
    ['model', 'prompt_length_bucket'],
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
)

# 记录指标时带上 exemplar
span = trace.get_current_span()
trace_id = format(span.get_span_context().trace_id, '032x')

ttft_histogram.labels(
    model="qwen2-72b",
    prompt_length_bucket="medium"
).observe(
    ttft_value,
    exemplar={'trace_id': trace_id}  # Grafana 可以直接跳转到 trace
)
```

### Grafana 中的关联配置

```yaml
# grafana datasources provisioning
datasources:
  - name: Prometheus
    type: prometheus
    url: http://prometheus:9090
    jsonData:
      exemplarTraceIdDestinations:
        - name: trace_id
          datasourceUid: tempo

  - name: Tempo
    type: tempo
    url: http://tempo:3200
    jsonData:
      tracesToLogs:
        datasourceUid: loki
        tags: ['request_id', 'gpu_id']
        mappedTags: [{ key: 'service.name', value: 'service' }]
        spanStartTimeShift: '-1h'
        spanEndTimeShift: '1h'
        filterByTraceID: true
        filterBySpanID: true
      tracesToMetrics:
        datasourceUid: prometheus
        tags: [{ key: 'gen_ai.request.model', value: 'model' }]
        queries:
          - name: 'GPU Utilization'
            query: 'DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{gpu_id="$${__tags.gpu_id}"}'
      serviceMap:
        datasourceUid: prometheus

  - name: Loki
    type: loki
    url: http://loki:3100
    jsonData:
      derivedFields:
        - datasourceUid: tempo
          matcherRegex: '"trace_id":"(\w+)"'
          name: TraceID
          url: '$${__value.raw}'
```

---

## 性能影响评估

### 追踪开销

```
场景：8×H20, vLLM, QPS=50

100% 采样 + 全 span：
  CPU 开销：~2% (序列化 + 发送)
  内存开销：~100MB (span buffer)
  网络开销：~5MB/s (到 collector)

尾部采样（推荐）：
  CPU 开销：~0.5%
  内存开销：~50MB
  网络开销：~0.5MB/s

结论：开销可接受，推荐在生产环境开启
```

---

## 下一步

→ 进入 [05_capacity_planning.md](05_capacity_planning.md) 了解如何基于 SLO 进行 GPU 容量规划
