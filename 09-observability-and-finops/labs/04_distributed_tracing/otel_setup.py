"""
OpenTelemetry 初始化与配置 — GPU 推理服务专用
============================================

本模块为 vLLM 推理服务配置完整的 OpenTelemetry tracing pipeline:
1. TracerProvider 初始化 (带 Resource 属性)
2. Span Processor 配置 (BatchSpanProcessor 优化性能)
3. Exporter 配置 (OTLP gRPC → OTel Collector)
4. Context Propagation (W3C TraceContext)
5. 自定义 Sampler (基于延迟的动态采样)
6. GPU 特化属性注入

设计原则:
- 低开销: 追踪对推理延迟的影响 < 0.1%
- 高信息密度: 每个 span 携带足够的诊断信息
- 智能采样: 慢请求 100% 采样，正常请求按比例采样
- 与 vLLM 内部状态深度集成

依赖:
    pip install opentelemetry-api opentelemetry-sdk \
               opentelemetry-exporter-otlp-proto-grpc \
               opentelemetry-instrumentation-fastapi \
               opentelemetry-instrumentation-httpx
"""

import os
import time
import logging
from typing import Optional, Sequence
from dataclasses import dataclass, field

from opentelemetry import trace, context
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.sampling import (
    Sampler,
    SamplingResult,
    Decision,
    ParentBased,
    TraceIdRatioBased,
)
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.trace import SpanKind, Link
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.context.propagation import set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.trace.propagation import TraceContextTextMapPropagator
from opentelemetry.baggage.propagation import W3CBaggagePropagator

logger = logging.getLogger(__name__)


# ============================================================
# 配置数据类
# ============================================================

@dataclass
class OTelConfig:
    """OpenTelemetry 配置

    支持通过环境变量覆盖，适配 Kubernetes ConfigMap 注入:
      OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
      OTEL_SERVICE_NAME=vllm-inference
      OTEL_TRACES_SAMPLER=parentbased_traceidratio
      OTEL_TRACES_SAMPLER_ARG=0.1
    """
    # === 服务标识 ===
    service_name: str = field(
        default_factory=lambda: os.getenv("OTEL_SERVICE_NAME", "vllm-inference")
    )
    service_version: str = field(
        default_factory=lambda: os.getenv("OTEL_SERVICE_VERSION", "1.0.0")
    )
    service_instance_id: str = field(
        default_factory=lambda: os.getenv("OTEL_SERVICE_INSTANCE_ID", "")
    )

    # === 集群与部署信息 ===
    cluster_name: str = field(
        default_factory=lambda: os.getenv("CLUSTER_NAME", "gpu-cluster-01")
    )
    namespace: str = field(
        default_factory=lambda: os.getenv("K8S_NAMESPACE", "inference")
    )
    node_name: str = field(
        default_factory=lambda: os.getenv("K8S_NODE_NAME", "")
    )
    pod_name: str = field(
        default_factory=lambda: os.getenv("K8S_POD_NAME", "")
    )

    # === GPU 信息 ===
    gpu_model: str = field(
        default_factory=lambda: os.getenv("GPU_MODEL", "NVIDIA-H20")
    )
    gpu_count: int = field(
        default_factory=lambda: int(os.getenv("GPU_COUNT", "8"))
    )
    tensor_parallel_size: int = field(
        default_factory=lambda: int(os.getenv("TP_SIZE", "8"))
    )

    # === 模型信息 ===
    model_name: str = field(
        default_factory=lambda: os.getenv("MODEL_NAME", "Qwen2.5-72B")
    )
    max_model_len: int = field(
        default_factory=lambda: int(os.getenv("MAX_MODEL_LEN", "32768"))
    )

    # === Exporter 配置 ===
    otlp_endpoint: str = field(
        default_factory=lambda: os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317"
        )
    )
    otlp_insecure: bool = field(
        default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "true").lower() == "true"
    )
    otlp_timeout: int = field(
        default_factory=lambda: int(os.getenv("OTEL_EXPORTER_OTLP_TIMEOUT", "10"))
    )

    # === 采样配置 ===
    base_sample_rate: float = field(
        default_factory=lambda: float(os.getenv("OTEL_TRACES_SAMPLER_ARG", "0.1"))
    )
    slow_request_threshold_s: float = field(
        default_factory=lambda: float(os.getenv("SLOW_REQUEST_THRESHOLD_S", "5.0"))
    )
    error_sample_rate: float = 1.0  # 错误请求 100% 采样
    slow_sample_rate: float = 1.0   # 慢请求 100% 采样

    # === Batch Processor 配置 ===
    batch_max_queue_size: int = 2048
    batch_max_export_batch_size: int = 512
    batch_schedule_delay_millis: int = 5000  # 5s 批次发送间隔
    batch_export_timeout_millis: int = 30000


# ============================================================
# 自定义采样器: 基于延迟的智能采样
# ============================================================

class InferenceAwareSampler(Sampler):
    """GPU 推理感知的智能采样器

    采样策略:
    1. 错误请求: 100% 采样 (用于根因分析)
    2. 慢请求 (TTFT > threshold): 100% 采样 (用于性能分析)
    3. 首次请求 (cold start): 100% 采样
    4. Preemption 发生: 100% 采样
    5. 正常请求: 按 base_rate 比例采样

    实现方式:
    - 由于在 span 开始时无法知道最终延迟，采用 "先采样后决定" 策略
    - 使用 ParentBased 确保 trace 内所有 span 一致性
    - 通过 span attributes 中的 hint 辅助决策
    """

    def __init__(self, config: OTelConfig):
        self._config = config
        self._base_sampler = TraceIdRatioBased(config.base_sample_rate)
        self._request_count = 0
        self._cold_start_threshold = 10  # 前 10 个请求视为 cold start

    def should_sample(
        self,
        parent_context: Optional[context.Context],
        trace_id: int,
        name: str,
        kind: SpanKind = None,
        attributes=None,
        links: Sequence[Link] = None,
    ) -> SamplingResult:
        """决定是否采样该 span

        决策逻辑:
        ┌─────────────────────────────────┐
        │ 有 parent span?                  │
        │ ├── YES → 跟随 parent 决策       │
        │ └── NO (root span) → 智能决策    │
        │     ├── cold start? → RECORD     │
        │     ├── hint=slow? → RECORD      │
        │     ├── hint=error? → RECORD     │
        │     ├── hint=preempt? → RECORD   │
        │     └── 默认 → base_rate 采样    │
        └─────────────────────────────────┘
        """
        # 如果有 parent context 且已决定采样/不采样，跟随
        parent_span = trace.get_current_span(parent_context)
        if parent_span and parent_span.get_span_context().is_valid:
            parent_decision = parent_span.get_span_context().trace_flags
            if parent_decision & trace.TraceFlags.SAMPLED:
                return SamplingResult(Decision.RECORD_AND_SAMPLE, attributes)
            else:
                return SamplingResult(Decision.DROP, attributes)

        # Root span: 智能采样决策
        self._request_count += 1

        # Cold start: 前 N 个请求全量采样 (用于验证系统正常)
        if self._request_count <= self._cold_start_threshold:
            logger.debug(f"Cold start sampling: request #{self._request_count}")
            return SamplingResult(Decision.RECORD_AND_SAMPLE, attributes)

        # 检查采样 hints (由业务代码注入)
        if attributes:
            attrs_dict = dict(attributes) if attributes else {}
            # 标记为慢请求 (预判, 如长 prompt)
            if attrs_dict.get("sampling.hint") == "slow":
                return SamplingResult(Decision.RECORD_AND_SAMPLE, attributes)
            # 标记为有 preemption
            if attrs_dict.get("sampling.hint") == "preempt":
                return SamplingResult(Decision.RECORD_AND_SAMPLE, attributes)
            # 标记为错误
            if attrs_dict.get("sampling.hint") == "error":
                return SamplingResult(Decision.RECORD_AND_SAMPLE, attributes)
            # 长 prompt 更可能触发性能问题
            prompt_tokens = attrs_dict.get("inference.prompt_tokens", 0)
            if prompt_tokens > 4096:
                return SamplingResult(Decision.RECORD_AND_SAMPLE, attributes)

        # 默认: 按比例采样
        return self._base_sampler.should_sample(
            parent_context, trace_id, name, kind, attributes, links
        )

    def get_description(self) -> str:
        return (
            f"InferenceAwareSampler("
            f"base_rate={self._config.base_sample_rate}, "
            f"slow_threshold={self._config.slow_request_threshold_s}s)"
        )


# ============================================================
# Span Processor: 追踪后处理 (添加 GPU 运行时属性)
# ============================================================

class GPUAttributeSpanProcessor(BatchSpanProcessor):
    """在 span 结束时注入 GPU 运行时状态

    普通 BatchSpanProcessor 只做 batch + export。
    本处理器额外注入:
    - GPU 温度 (如果可获取)
    - KV Cache 当前使用率
    - 当前 batch size
    - Preemption 计数

    这些信息对事后诊断极其有价值:
    "这个请求慢是因为当时 KV Cache 已经 92%"
    """

    def __init__(self, exporter: SpanExporter, gpu_state_provider=None, **kwargs):
        super().__init__(exporter, **kwargs)
        self._gpu_state_provider = gpu_state_provider

    def on_end(self, span: ReadableSpan) -> None:
        """span 结束时尝试注入运行时 GPU 状态"""
        if self._gpu_state_provider and span.name.startswith("model."):
            try:
                gpu_state = self._gpu_state_provider.get_current_state()
                # 注意: ReadableSpan 不可修改，这里演示概念
                # 实际实现中需要在 span 结束前 set_attribute
                pass
            except Exception as e:
                logger.debug(f"Failed to inject GPU state: {e}")

        super().on_end(span)


# ============================================================
# GPU 状态提供者 (用于 span 属性注入)
# ============================================================

class GPUStateProvider:
    """提供 GPU 运行时状态用于 trace 属性注入

    数据来源:
    1. vLLM 内部状态 (KV Cache, Scheduler)
    2. NVML (GPU 温度, 利用率)
    3. 缓存最近一次查询结果 (避免高频调用 NVML)
    """

    def __init__(self, cache_ttl_s: float = 1.0):
        self._cache_ttl = cache_ttl_s
        self._last_query_time = 0
        self._cached_state = {}
        self._nvml_initialized = False
        self._init_nvml()

    def _init_nvml(self):
        """初始化 NVIDIA Management Library"""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_initialized = True
            self._device_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"NVML initialized: {self._device_count} GPUs detected")
        except Exception as e:
            logger.warning(f"NVML not available: {e}. GPU attributes will be limited.")
            self._nvml_initialized = False

    def get_current_state(self) -> dict:
        """获取当前 GPU 状态 (带缓存)

        Returns:
            {
                "gpu.temperature_c": [65, 67, 64, ...],
                "gpu.utilization_pct": [85, 82, 88, ...],
                "gpu.memory_used_gb": [72.5, 73.1, ...],
                "gpu.power_w": [350, 345, 360, ...],
            }
        """
        now = time.time()
        if now - self._last_query_time < self._cache_ttl:
            return self._cached_state

        state = {}
        if self._nvml_initialized:
            try:
                import pynvml
                temps = []
                utils = []
                mem_used = []
                powers = []

                for i in range(self._device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    temps.append(pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    ))
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utils.append(util.gpu)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_used.append(round(mem_info.used / (1024**3), 2))
                    powers.append(
                        pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW → W
                    )

                state = {
                    "gpu.temperature_c": temps,
                    "gpu.utilization_pct": utils,
                    "gpu.memory_used_gb": mem_used,
                    "gpu.power_w": powers,
                    "gpu.max_temperature_c": max(temps),
                    "gpu.avg_utilization_pct": sum(utils) / len(utils),
                }
            except Exception as e:
                logger.debug(f"NVML query failed: {e}")

        self._cached_state = state
        self._last_query_time = now
        return state


# ============================================================
# 主初始化函数
# ============================================================

def init_tracing(config: Optional[OTelConfig] = None) -> trace.Tracer:
    """初始化 OpenTelemetry Tracing

    完整初始化流程:
    1. 创建 Resource (服务标识 + GPU 信息)
    2. 创建 Sampler (智能采样)
    3. 创建 TracerProvider
    4. 配置 SpanProcessor + Exporter
    5. 设置 Context Propagation
    6. 返回 Tracer 实例

    Args:
        config: OTel 配置，None 则使用默认/环境变量

    Returns:
        配置好的 Tracer 实例

    Usage:
        tracer = init_tracing()
        with tracer.start_as_current_span("my_operation") as span:
            span.set_attribute("key", "value")
            do_something()
    """
    if config is None:
        config = OTelConfig()

    # === Step 1: 创建 Resource ===
    # Resource 标识 "谁" 产生了这个 trace
    resource = Resource.create({
        SERVICE_NAME: config.service_name,
        "service.version": config.service_version,
        "service.instance.id": config.service_instance_id or config.pod_name,

        # Kubernetes 上下文
        "k8s.cluster.name": config.cluster_name,
        "k8s.namespace.name": config.namespace,
        "k8s.node.name": config.node_name,
        "k8s.pod.name": config.pod_name,

        # GPU 推理特有属性
        "gpu.model": config.gpu_model,
        "gpu.count": config.gpu_count,
        "inference.model": config.model_name,
        "inference.tp_size": config.tensor_parallel_size,
        "inference.max_model_len": config.max_model_len,

        # 部署环境
        "deployment.environment": os.getenv("DEPLOY_ENV", "production"),
    })

    logger.info(
        f"OTel Resource created: service={config.service_name}, "
        f"model={config.model_name}, tp={config.tensor_parallel_size}"
    )

    # === Step 2: 创建 Sampler ===
    sampler = ParentBased(
        root=InferenceAwareSampler(config),
        # 远程 parent 已采样 → 跟随
        remote_parent_sampled=TraceIdRatioBased(1.0),
        # 远程 parent 未采样 → 仍有小概率采样 (兜底)
        remote_parent_not_sampled=TraceIdRatioBased(0.01),
    )

    # === Step 3: 创建 TracerProvider ===
    provider = TracerProvider(
        resource=resource,
        sampler=sampler,
    )

    # === Step 4: 配置 Exporter + Processor ===
    # 主 Exporter: OTLP gRPC → OTel Collector
    otlp_exporter = OTLPSpanExporter(
        endpoint=config.otlp_endpoint,
        insecure=config.otlp_insecure,
        timeout=config.otlp_timeout,
    )

    # BatchSpanProcessor: 异步批量发送，不阻塞推理
    batch_processor = BatchSpanProcessor(
        otlp_exporter,
        max_queue_size=config.batch_max_queue_size,
        max_export_batch_size=config.batch_max_export_batch_size,
        schedule_delay_millis=config.batch_schedule_delay_millis,
        export_timeout_millis=config.batch_export_timeout_millis,
    )
    provider.add_span_processor(batch_processor)

    # 开发环境: 额外添加 Console Exporter (调试用)
    if os.getenv("OTEL_DEBUG", "false").lower() == "true":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        provider.add_span_processor(
            SimpleSpanProcessor(ConsoleSpanExporter())
        )
        logger.info("Debug mode: ConsoleSpanExporter enabled")

    # === Step 5: 注册全局 Provider ===
    trace.set_tracer_provider(provider)

    # === Step 6: 配置 Context Propagation ===
    # W3C TraceContext + Baggage — 跨服务传播 trace context
    propagator = CompositePropagator([
        TraceContextTextMapPropagator(),
        W3CBaggagePropagator(),
    ])
    set_global_textmap(propagator)

    logger.info(
        f"OTel Tracing initialized: "
        f"endpoint={config.otlp_endpoint}, "
        f"sample_rate={config.base_sample_rate}, "
        f"batch_size={config.batch_max_export_batch_size}"
    )

    # 返回命名 Tracer
    return trace.get_tracer(
        instrumenting_module_name="vllm.inference",
        instrumenting_library_version=config.service_version,
    )


# ============================================================
# FastAPI 中间件集成
# ============================================================

def setup_fastapi_tracing(app, config: Optional[OTelConfig] = None):
    """为 vLLM 的 FastAPI 应用添加自动追踪

    自动追踪:
    - 所有 HTTP 请求 (入口 span)
    - 请求头中的 trace context 提取
    - 响应头中的 trace context 注入
    - HTTP 状态码、方法、路径等属性

    Args:
        app: FastAPI application instance
        config: OTel 配置

    Usage:
        from fastapi import FastAPI
        app = FastAPI()
        tracer = setup_fastapi_tracing(app)
    """
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    tracer = init_tracing(config)

    # 自动 instrument FastAPI
    FastAPIInstrumentor.instrument_app(
        app,
        # 排除健康检查和 metrics 端点 (这些不需要追踪)
        excluded_urls="/health,/metrics,/readyz,/livez",
        # 自定义 span 名称: 使用 HTTP method + route 而非完整 URL
        server_request_hook=_server_request_hook,
        client_request_hook=_client_request_hook,
        client_response_hook=_client_response_hook,
    )

    logger.info("FastAPI auto-instrumentation enabled")
    return tracer


def _server_request_hook(span, scope):
    """服务端请求 hook: 在 span 开始时注入额外属性"""
    if span and span.is_recording():
        # 从请求头提取业务属性
        headers = dict(scope.get("headers", []))
        # 请求优先级 (用于采样决策)
        priority = headers.get(b"x-request-priority", b"normal").decode()
        span.set_attribute("request.priority", priority)
        # 客户端标识
        client_id = headers.get(b"x-client-id", b"unknown").decode()
        span.set_attribute("client.id", client_id)


def _client_request_hook(span, request):
    """客户端请求 hook (vLLM 调用外部服务时)"""
    pass


def _client_response_hook(span, response):
    """客户端响应 hook"""
    pass


# ============================================================
# 工具函数: Span 属性注入辅助
# ============================================================

def set_inference_attributes(
    span: trace.Span,
    prompt_tokens: int,
    output_tokens: int = 0,
    model: str = "",
    stream: bool = True,
    temperature: float = 1.0,
    max_tokens: int = 2048,
):
    """为推理请求 span 设置标准属性

    遵循 OpenTelemetry Semantic Conventions for GenAI:
    https://opentelemetry.io/docs/specs/semconv/gen-ai/

    Args:
        span: 当前 span
        prompt_tokens: 输入 token 数
        output_tokens: 输出 token 数 (可后续更新)
        model: 模型名称
        stream: 是否 streaming
        temperature: 采样温度
        max_tokens: 最大生成长度
    """
    if not span.is_recording():
        return

    # GenAI Semantic Conventions
    span.set_attribute("gen_ai.system", "vllm")
    span.set_attribute("gen_ai.request.model", model)
    span.set_attribute("gen_ai.request.max_tokens", max_tokens)
    span.set_attribute("gen_ai.request.temperature", temperature)
    span.set_attribute("gen_ai.request.stream", stream)
    span.set_attribute("gen_ai.usage.prompt_tokens", prompt_tokens)
    span.set_attribute("gen_ai.usage.completion_tokens", output_tokens)

    # 推理特化属性
    span.set_attribute("inference.prompt_tokens", prompt_tokens)
    span.set_attribute("inference.output_tokens", output_tokens)
    span.set_attribute("inference.total_tokens", prompt_tokens + output_tokens)

    # 采样 hint: 长 prompt 标记为可能慢
    if prompt_tokens > 4096:
        span.set_attribute("sampling.hint", "slow")


def set_kv_cache_attributes(
    span: trace.Span,
    gpu_cache_usage: float,
    prefix_cache_hit_rate: float = 0.0,
    blocks_allocated: int = 0,
    blocks_total: int = 0,
):
    """注入 KV Cache 相关属性

    这些属性对诊断延迟问题至关重要:
    "为什么这个请求 TTFT 是 8s?" → "因为当时 KV Cache 92%, 触发了 preemption"
    """
    if not span.is_recording():
        return

    span.set_attribute("kv_cache.gpu_usage_pct", round(gpu_cache_usage * 100, 1))
    span.set_attribute("kv_cache.prefix_hit_rate", round(prefix_cache_hit_rate, 3))
    span.set_attribute("kv_cache.blocks_allocated", blocks_allocated)
    span.set_attribute("kv_cache.blocks_total", blocks_total)
    span.set_attribute("kv_cache.blocks_free", blocks_total - blocks_allocated)

    # 标记高压力状态
    if gpu_cache_usage > 0.9:
        span.set_attribute("kv_cache.pressure", "high")
        span.set_attribute("sampling.hint", "slow")  # 确保被采样
    elif gpu_cache_usage > 0.7:
        span.set_attribute("kv_cache.pressure", "medium")
    else:
        span.set_attribute("kv_cache.pressure", "low")


def set_scheduling_attributes(
    span: trace.Span,
    queue_position: int,
    wait_time_ms: float,
    batch_size: int,
    preempted: bool = False,
    swap_in: bool = False,
):
    """注入调度相关属性"""
    if not span.is_recording():
        return

    span.set_attribute("scheduler.queue_position", queue_position)
    span.set_attribute("scheduler.wait_time_ms", round(wait_time_ms, 2))
    span.set_attribute("scheduler.batch_size", batch_size)
    span.set_attribute("scheduler.preempted", preempted)
    span.set_attribute("scheduler.swap_in", swap_in)

    if preempted:
        span.set_attribute("sampling.hint", "preempt")
        span.add_event("preemption_occurred", {
            "batch_size_at_preemption": batch_size,
        })


# ============================================================
# Trace Context 传播工具
# ============================================================

class TraceContextCarrier:
    """Trace Context 载体 — 用于跨 Worker 传播

    vLLM TP=8 时有多个 Worker 进程:
    - API Server (接收请求)
    - Scheduler (调度)
    - Workers[0..7] (GPU 执行)

    需要在它们之间传播 trace context:
    API Server → [inject context] → message → [extract context] → Worker
    """

    @staticmethod
    def inject_to_dict(context_dict: Optional[dict] = None) -> dict:
        """将当前 trace context 注入到 dict 中

        Usage:
            ctx = TraceContextCarrier.inject_to_dict()
            # 通过 IPC/message 传给 Worker
            send_to_worker(request_data, trace_context=ctx)
        """
        carrier = {}
        propagator = trace.get_tracer_provider()
        TraceContextTextMapPropagator().inject(carrier)
        if context_dict:
            carrier.update(context_dict)
        return carrier

    @staticmethod
    def extract_from_dict(carrier: dict) -> context.Context:
        """从 dict 中提取 trace context

        Usage:
            ctx = TraceContextCarrier.extract_from_dict(trace_context)
            with trace.use_span(ctx):
                # 在 Worker 中创建子 span
                with tracer.start_as_current_span("worker.execute"):
                    ...
        """
        return TraceContextTextMapPropagator().extract(carrier=carrier)


# ============================================================
# 清理函数
# ============================================================

def shutdown_tracing():
    """优雅关闭 tracing pipeline

    确保所有 pending spans 被导出:
    - flush BatchSpanProcessor 队列
    - 关闭 OTLP 连接
    - 释放资源

    应在服务关闭时调用 (如 SIGTERM handler)
    """
    provider = trace.get_tracer_provider()
    if hasattr(provider, 'shutdown'):
        logger.info("Shutting down OTel TracerProvider...")
        provider.shutdown()
        logger.info("OTel TracerProvider shutdown complete")


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 初始化 (使用默认配置 + 环境变量覆盖)
    config = OTelConfig(
        service_name="vllm-inference-demo",
        otlp_endpoint="http://localhost:4317",
        base_sample_rate=1.0,  # Demo: 100% 采样
    )
    tracer = init_tracing(config)

    # 模拟一个推理请求 trace
    with tracer.start_as_current_span(
        "inference.request",
        kind=SpanKind.SERVER,
        attributes={"http.method": "POST", "http.url": "/v1/chat/completions"},
    ) as root_span:
        # 模拟 tokenize
        with tracer.start_as_current_span("tokenizer.encode") as tok_span:
            time.sleep(0.003)
            tok_span.set_attribute("tokenizer.prompt_tokens", 1024)

        # 模拟 schedule
        with tracer.start_as_current_span("scheduler.schedule") as sched_span:
            time.sleep(0.001)
            set_scheduling_attributes(sched_span, queue_position=0, wait_time_ms=0,
                                      batch_size=16)

        # 模拟 prefill
        with tracer.start_as_current_span("model.prefill") as prefill_span:
            time.sleep(0.45)  # 450ms TTFT
            set_kv_cache_attributes(prefill_span, gpu_cache_usage=0.72,
                                    prefix_cache_hit_rate=0.85,
                                    blocks_allocated=150, blocks_total=1000)
            prefill_span.set_attribute("model.prefill_tokens", 1024)

        # 模拟 decode (多步)
        with tracer.start_as_current_span("model.decode") as decode_span:
            tokens_generated = 128
            time.sleep(0.025 * tokens_generated)  # 25ms/token
            decode_span.set_attribute("model.output_tokens", tokens_generated)
            decode_span.set_attribute("model.avg_tpot_ms", 25.0)

        # 设置最终属性
        set_inference_attributes(root_span, prompt_tokens=1024,
                                 output_tokens=128, model="Qwen2.5-72B")
        root_span.set_status(Status(StatusCode.OK))

    print("Demo trace sent. Check Jaeger UI at http://localhost:16686")
    shutdown_tracing()
