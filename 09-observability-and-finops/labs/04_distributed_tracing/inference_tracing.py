"""
vLLM 推理链路追踪实现 — 深度集成 Continuous Batching
====================================================

本模块实现对 vLLM 推理引擎的细粒度追踪:
1. 请求级 Trace: 完整的 API → Tokenize → Schedule → Prefill → Decode → Response
2. 调度级 Trace: Scheduler 每一步的决策 (add/preempt/swap)
3. 引擎级 Trace: Model Forward、KV Cache 管理、Sampling
4. 跨 Worker Trace: TP 组内多 GPU 协同

核心挑战:
- vLLM 使用 Continuous Batching, 一个 GPU step 内处理多个请求
  → 需要将 GPU 执行时间正确归因到各个请求
- Prefill 和 Decode 在同一个 step 中混合执行 (Chunked Prefill)
  → 需要区分不同类型的计算
- Preemption 可能中断请求并稍后恢复
  → 需要记录 preemption event 并关联前后 span

依赖:
    pip install opentelemetry-api opentelemetry-sdk
    本文件依赖 otel_setup.py 的初始化
"""

import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager

from opentelemetry import trace, context
from opentelemetry.trace import SpanKind, StatusCode, Status

from otel_setup import (
    init_tracing,
    set_inference_attributes,
    set_kv_cache_attributes,
    set_scheduling_attributes,
    TraceContextCarrier,
    OTelConfig,
)

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

class RequestPhase(Enum):
    """请求生命周期阶段"""
    QUEUED = "queued"          # 在等待队列中
    PREFILLING = "prefilling"  # Prefill 阶段 (处理 prompt)
    DECODING = "decoding"      # Decode 阶段 (生成 tokens)
    PREEMPTED = "preempted"    # 被抢占 (暂停)
    SWAPPED = "swapped"        # KV 被 swap 到 CPU
    FINISHED = "finished"      # 完成


@dataclass
class TracedRequest:
    """带追踪上下文的请求

    每个推理请求关联一组 spans:
    - root_span: 整个请求的入口 span
    - phase_spans: 各阶段的子 span (tokenize, schedule, prefill, decode, ...)
    - events: 关键事件 (preemption, swap, prefix cache hit, ...)
    """
    request_id: str
    trace_context: context.Context
    root_span: trace.Span

    # 请求元数据
    model: str = ""
    prompt_tokens: int = 0
    max_output_tokens: int = 2048
    stream: bool = True
    temperature: float = 1.0
    arrival_time: float = field(default_factory=time.time)

    # 追踪状态
    current_phase: RequestPhase = RequestPhase.QUEUED
    phase_spans: Dict[str, trace.Span] = field(default_factory=dict)
    output_tokens_generated: int = 0
    preemption_count: int = 0
    decode_steps: int = 0

    # 性能计时
    queue_start_time: float = 0
    prefill_start_time: float = 0
    first_token_time: float = 0
    decode_start_time: float = 0

    # KV Cache 快照 (记录关键时刻的 cache 状态)
    kv_cache_at_schedule: float = 0
    kv_cache_at_prefill: float = 0


# ============================================================
# 核心: 推理追踪器
# ============================================================

class InferenceTracer:
    """vLLM 推理链路追踪器

    与 vLLM Engine 的集成点:
    ┌──────────────────────────────────────────────────────────┐
    │                    vLLM Engine Loop                       │
    │                                                          │
    │  add_request() ──→ on_request_arrive()                   │
    │       │                                                  │
    │  scheduler.schedule() ──→ on_schedule_step()             │
    │       │                                                  │
    │  model.execute() ──→ on_model_forward_start/end()        │
    │       │                                                  │
    │  sampler.sample() ──→ on_tokens_generated()              │
    │       │                                                  │
    │  scheduler.preempt() ──→ on_preemption()                 │
    │       │                                                  │
    │  request.finish() ──→ on_request_finish()                │
    └──────────────────────────────────────────────────────────┘

    使用方式 (在 vLLM 源码中注入):
        tracer = InferenceTracer()
        # 在 engine.add_request 中:
        tracer.on_request_arrive(request_id, prompt_tokens, ...)
        # 在 scheduler.schedule 中:
        tracer.on_schedule_step(running, waiting, swapped, ...)
        # ...
    """

    def __init__(self, config: Optional[OTelConfig] = None):
        self._tracer = init_tracing(config)
        self._active_requests: Dict[str, TracedRequest] = {}
        self._step_count = 0

        # 统计
        self._total_requests = 0
        self._total_preemptions = 0

        logger.info("InferenceTracer initialized")

    @property
    def tracer(self) -> trace.Tracer:
        return self._tracer

    # ========================================================
    # 请求生命周期追踪
    # ========================================================

    def on_request_arrive(
        self,
        request_id: str,
        prompt_tokens: int,
        model: str = "Qwen2.5-72B",
        max_output_tokens: int = 2048,
        stream: bool = True,
        temperature: float = 1.0,
        client_trace_context: Optional[dict] = None,
    ) -> TracedRequest:
        """新推理请求到达

        创建 root span, 记录请求元数据, 开始 queue span。

        调用时机: vLLM engine.add_request()
        """
        self._total_requests += 1
        now = time.time()

        # 如果客户端传递了 trace context, 从中提取 parent
        parent_ctx = context.get_current()
        if client_trace_context:
            parent_ctx = TraceContextCarrier.extract_from_dict(client_trace_context)

        # 创建 root span (整个请求的生命周期)
        root_span = self._tracer.start_span(
            "inference.request",
            context=parent_ctx,
            kind=SpanKind.SERVER,
            attributes={
                "request.id": request_id,
                "gen_ai.system": "vllm",
                "gen_ai.request.model": model,
                "gen_ai.request.max_tokens": max_output_tokens,
                "gen_ai.request.temperature": temperature,
                "gen_ai.request.stream": stream,
                "inference.prompt_tokens": prompt_tokens,
            },
        )

        # 创建请求追踪对象
        traced = TracedRequest(
            request_id=request_id,
            trace_context=trace.set_span_in_context(root_span),
            root_span=root_span,
            model=model,
            prompt_tokens=prompt_tokens,
            max_output_tokens=max_output_tokens,
            stream=stream,
            temperature=temperature,
            arrival_time=now,
            queue_start_time=now,
        )

        # 开始 queue span (从到达到被调度)
        queue_span = self._tracer.start_span(
            "scheduler.queue",
            context=traced.trace_context,
            attributes={
                "request.id": request_id,
                "scheduler.queue_depth": len(self._active_requests),
            },
        )
        traced.phase_spans["queue"] = queue_span

        # 添加到活跃请求
        self._active_requests[request_id] = traced

        logger.debug(f"Trace started for request {request_id}: {prompt_tokens} prompt tokens")
        return traced

    def on_request_scheduled(
        self,
        request_id: str,
        action: str = "prefill",  # "prefill" | "decode" | "preempt" | "swap_out" | "swap_in"
        batch_size: int = 0,
        kv_cache_usage: float = 0.0,
    ):
        """请求被调度器选中

        调用时机: scheduler.schedule() 决定处理某个请求
        """
        traced = self._active_requests.get(request_id)
        if not traced:
            return

        now = time.time()

        # 结束 queue span (如果存在)
        queue_span = traced.phase_spans.pop("queue", None)
        if queue_span:
            wait_time_ms = (now - traced.queue_start_time) * 1000
            queue_span.set_attribute("scheduler.wait_time_ms", round(wait_time_ms, 2))
            queue_span.end()

        # 记录调度时刻的 KV Cache 状态
        traced.kv_cache_at_schedule = kv_cache_usage

        # 根据 action 创建对应阶段 span
        if action == "prefill":
            traced.current_phase = RequestPhase.PREFILLING
            traced.prefill_start_time = now
            span = self._tracer.start_span(
                "model.prefill",
                context=traced.trace_context,
                attributes={
                    "request.id": request_id,
                    "model.action": "prefill",
                    "inference.prompt_tokens": traced.prompt_tokens,
                    "scheduler.batch_size": batch_size,
                    "kv_cache.gpu_usage_pct": round(kv_cache_usage * 100, 1),
                },
            )
            traced.phase_spans["prefill"] = span

        elif action == "decode":
            traced.current_phase = RequestPhase.DECODING
            if traced.decode_start_time == 0:
                traced.decode_start_time = now
            # Decode span 在首次进入 decode 时创建, 持续到请求结束
            if "decode" not in traced.phase_spans:
                span = self._tracer.start_span(
                    "model.decode",
                    context=traced.trace_context,
                    attributes={
                        "request.id": request_id,
                        "model.action": "decode",
                    },
                )
                traced.phase_spans["decode"] = span

        elif action == "preempt":
            self.on_preemption(request_id, kv_cache_usage=kv_cache_usage,
                               batch_size=batch_size)

        elif action == "swap_out":
            self.on_swap_out(request_id)

        elif action == "swap_in":
            self.on_swap_in(request_id)

    def on_prefill_complete(
        self,
        request_id: str,
        prefix_cache_hit_rate: float = 0.0,
        computed_tokens: int = 0,
        cached_tokens: int = 0,
    ):
        """Prefill 阶段完成 (首个 token 即将生成)

        标记 TTFT 时刻 — 这是最关键的延迟指标之一。

        调用时机: 模型完成 prefill forward, 即将开始 decode
        """
        traced = self._active_requests.get(request_id)
        if not traced:
            return

        now = time.time()
        traced.first_token_time = now

        # 结束 prefill span
        prefill_span = traced.phase_spans.pop("prefill", None)
        if prefill_span:
            ttft_ms = (now - traced.arrival_time) * 1000
            prefill_compute_ms = (now - traced.prefill_start_time) * 1000

            prefill_span.set_attribute("model.ttft_ms", round(ttft_ms, 2))
            prefill_span.set_attribute("model.prefill_compute_ms", round(prefill_compute_ms, 2))
            prefill_span.set_attribute("model.prefix_cache_hit_rate", round(prefix_cache_hit_rate, 3))
            prefill_span.set_attribute("model.computed_tokens", computed_tokens)
            prefill_span.set_attribute("model.cached_tokens", cached_tokens)

            # Prefix Cache 事件
            if cached_tokens > 0:
                prefill_span.add_event("prefix_cache_hit", {
                    "cached_tokens": cached_tokens,
                    "computed_tokens": computed_tokens,
                    "hit_rate": round(prefix_cache_hit_rate, 3),
                    "ttft_savings_estimate_ms": round(
                        cached_tokens / max(computed_tokens, 1) * prefill_compute_ms, 1
                    ),
                })

            prefill_span.end()

        # Root span 记录 TTFT
        traced.root_span.set_attribute(
            "inference.ttft_ms", round((now - traced.arrival_time) * 1000, 2)
        )

        logger.debug(
            f"Prefill complete for {request_id}: "
            f"TTFT={round((now - traced.arrival_time)*1000, 1)}ms, "
            f"prefix_hit={prefix_cache_hit_rate:.2f}"
        )

    def on_token_generated(
        self,
        request_id: str,
        token_id: int,
        is_eos: bool = False,
        batch_size: int = 0,
        step_latency_ms: float = 0,
    ):
        """生成一个 output token

        记录每个 decode step 的关键信息:
        - Inter-token latency (TPOT)
        - 当前 batch size (影响 TPOT)
        - 是否结束 (EOS)

        调用时机: sampler 采样出 token 后
        """
        traced = self._active_requests.get(request_id)
        if not traced:
            return

        traced.output_tokens_generated += 1
        traced.decode_steps += 1

        # 在 decode span 中记录 step event (采样记录, 避免过多)
        decode_span = traced.phase_spans.get("decode")
        if decode_span and (
            traced.decode_steps <= 3 or         # 前 3 步
            traced.decode_steps % 50 == 0 or    # 每 50 步
            is_eos or                           # 最后一步
            step_latency_ms > 100               # 异常慢的步骤
        ):
            decode_span.add_event(
                f"decode_step_{traced.decode_steps}",
                {
                    "step": traced.decode_steps,
                    "token_id": token_id,
                    "batch_size": batch_size,
                    "step_latency_ms": round(step_latency_ms, 2),
                    "is_eos": is_eos,
                },
            )

    def on_preemption(
        self,
        request_id: str,
        reason: str = "kv_cache_full",
        kv_cache_usage: float = 0.0,
        batch_size: int = 0,
    ):
        """请求被抢占

        Preemption 是 vLLM 特有的关键事件:
        - KV Cache 满 → Scheduler 选择 "最不重要" 的请求暂停
        - 被抢占请求的 KV blocks 被回收
        - 稍后该请求会被重新调度 (recompute 或 swap-in)

        这是性能诊断的重要线索:
        "为什么这个请求总 E2E 是 30s? 因为中间被 preempt 了 3 次"

        调用时机: scheduler 决定 preempt 某个请求
        """
        traced = self._active_requests.get(request_id)
        if not traced:
            return

        traced.preemption_count += 1
        traced.current_phase = RequestPhase.PREEMPTED
        self._total_preemptions += 1

        # 在 root span 记录 preemption 事件
        traced.root_span.add_event(
            "preemption",
            {
                "preemption_count": traced.preemption_count,
                "reason": reason,
                "kv_cache_usage_pct": round(kv_cache_usage * 100, 1),
                "batch_size_at_preemption": batch_size,
                "tokens_generated_before_preempt": traced.output_tokens_generated,
            },
        )

        # 标记当前活跃 span
        for name, span in traced.phase_spans.items():
            if span.is_recording():
                span.add_event("preempted", {"reason": reason})

        # 创建 preemption span (记录暂停时间)
        preempt_span = self._tracer.start_span(
            "scheduler.preempted",
            context=traced.trace_context,
            attributes={
                "request.id": request_id,
                "scheduler.preemption_count": traced.preemption_count,
                "scheduler.preemption_reason": reason,
                "kv_cache.gpu_usage_pct": round(kv_cache_usage * 100, 1),
            },
        )
        traced.phase_spans["preempt"] = preempt_span

        logger.info(
            f"Request {request_id} preempted (count={traced.preemption_count}): "
            f"reason={reason}, kv_cache={kv_cache_usage:.2f}"
        )

    def on_swap_out(self, request_id: str):
        """请求 KV Cache 被 swap 到 CPU

        比 preemption 更严重的性能退化:
        - GPU → CPU 传输延迟 (PCIe 带宽限制)
        - CPU 内存占用增加
        - swap-in 时还需要 CPU → GPU 传输
        """
        traced = self._active_requests.get(request_id)
        if not traced:
            return

        traced.current_phase = RequestPhase.SWAPPED
        traced.root_span.add_event(
            "kv_cache_swap_out",
            {
                "tokens_in_cache": traced.output_tokens_generated + traced.prompt_tokens,
                "direction": "gpu_to_cpu",
            },
        )

        swap_span = self._tracer.start_span(
            "kv_cache.swap_out",
            context=traced.trace_context,
            attributes={
                "request.id": request_id,
                "kv_cache.swap_direction": "gpu_to_cpu",
            },
        )
        traced.phase_spans["swap_out"] = swap_span

    def on_swap_in(self, request_id: str):
        """请求 KV Cache 从 CPU swap 回 GPU"""
        traced = self._active_requests.get(request_id)
        if not traced:
            return

        # 结束 swap_out span
        swap_out_span = traced.phase_spans.pop("swap_out", None)
        if swap_out_span:
            swap_out_span.end()

        # 结束 preempt span
        preempt_span = traced.phase_spans.pop("preempt", None)
        if preempt_span:
            preempt_span.end()

        traced.root_span.add_event(
            "kv_cache_swap_in",
            {"direction": "cpu_to_gpu"},
        )

    def on_request_finish(
        self,
        request_id: str,
        finish_reason: str = "stop",  # "stop" | "length" | "error"
        error: Optional[str] = None,
    ):
        """请求完成

        结束所有活跃 spans, 设置最终属性和状态。

        调用时机: request 生成完毕或出错
        """
        traced = self._active_requests.pop(request_id, None)
        if not traced:
            return

        now = time.time()
        e2e_latency_ms = (now - traced.arrival_time) * 1000

        # 结束所有活跃子 spans
        for name, span in traced.phase_spans.items():
            if span.is_recording():
                span.end()

        # 计算最终指标
        avg_tpot_ms = 0
        if traced.output_tokens_generated > 0 and traced.first_token_time > 0:
            decode_time = now - traced.first_token_time
            avg_tpot_ms = (decode_time / traced.output_tokens_generated) * 1000

        ttft_ms = 0
        if traced.first_token_time > 0:
            ttft_ms = (traced.first_token_time - traced.arrival_time) * 1000

        # 设置 root span 最终属性
        root = traced.root_span
        set_inference_attributes(
            root,
            prompt_tokens=traced.prompt_tokens,
            output_tokens=traced.output_tokens_generated,
            model=traced.model,
            stream=traced.stream,
            temperature=traced.temperature,
            max_tokens=traced.max_output_tokens,
        )

        root.set_attribute("inference.e2e_latency_ms", round(e2e_latency_ms, 2))
        root.set_attribute("inference.ttft_ms", round(ttft_ms, 2))
        root.set_attribute("inference.avg_tpot_ms", round(avg_tpot_ms, 2))
        root.set_attribute("inference.output_tokens", traced.output_tokens_generated)
        root.set_attribute("inference.decode_steps", traced.decode_steps)
        root.set_attribute("inference.preemption_count", traced.preemption_count)
        root.set_attribute("inference.finish_reason", finish_reason)
        root.set_attribute("kv_cache.usage_at_schedule_pct",
                           round(traced.kv_cache_at_schedule * 100, 1))

        # Tokens per second (该请求的有效吞吐)
        if e2e_latency_ms > 0:
            tps = traced.output_tokens_generated / (e2e_latency_ms / 1000)
            root.set_attribute("inference.tokens_per_second", round(tps, 1))

        # 设置状态
        if error:
            root.set_status(Status(StatusCode.ERROR, error))
            root.record_exception(Exception(error))
        else:
            root.set_status(Status(StatusCode.OK))

        root.end()

        logger.debug(
            f"Trace complete for {request_id}: "
            f"e2e={e2e_latency_ms:.0f}ms, ttft={ttft_ms:.0f}ms, "
            f"tpot={avg_tpot_ms:.1f}ms, tokens={traced.output_tokens_generated}, "
            f"preemptions={traced.preemption_count}"
        )

    # ========================================================
    # 引擎级追踪 (Model Forward)
    # ========================================================

    def on_engine_step_start(
        self,
        step_id: int,
        running_request_ids: List[str],
        prefill_request_ids: List[str],
        total_tokens: int,
    ):
        """Engine 执行一个 step (可能包含多个请求的 prefill + decode)

        这是 Continuous Batching 的核心:
        一个 GPU forward 同时处理:
        - N 个请求的 decode (每个生成 1 token)
        - M 个新请求的 prefill (或 chunked prefill 的一个 chunk)

        调用时机: engine.step() 开始
        """
        self._step_count = step_id

        # 为这个 engine step 创建一个 span
        # 注意: 这个 span 没有 parent (因为它跨越多个请求)
        step_span = self._tracer.start_span(
            "engine.step",
            attributes={
                "engine.step_id": step_id,
                "engine.running_requests": len(running_request_ids),
                "engine.prefill_requests": len(prefill_request_ids),
                "engine.total_batch_tokens": total_tokens,
            },
        )
        return step_span

    def on_engine_step_end(
        self,
        step_span: trace.Span,
        forward_time_ms: float,
        sample_time_ms: float,
        schedule_time_ms: float,
    ):
        """Engine step 结束"""
        if step_span and step_span.is_recording():
            step_span.set_attribute("engine.forward_time_ms", round(forward_time_ms, 2))
            step_span.set_attribute("engine.sample_time_ms", round(sample_time_ms, 2))
            step_span.set_attribute("engine.schedule_time_ms", round(schedule_time_ms, 2))
            step_span.set_attribute("engine.overhead_ms",
                                    round(schedule_time_ms + sample_time_ms, 2))
            step_span.end()

    # ========================================================
    # 上下文管理器: 简化追踪代码
    # ========================================================

    @contextmanager
    def trace_tokenize(self, request_id: str, direction: str = "encode"):
        """追踪 tokenize 阶段

        Usage:
            with tracer.trace_tokenize(req_id, "encode") as span:
                tokens = tokenizer.encode(prompt)
                span.set_attribute("tokenizer.output_length", len(tokens))
        """
        traced = self._active_requests.get(request_id)
        if not traced:
            yield None
            return

        span = self._tracer.start_span(
            f"tokenizer.{direction}",
            context=traced.trace_context,
            attributes={"request.id": request_id},
        )
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        finally:
            span.end()

    @contextmanager
    def trace_sampling(self, request_id: str):
        """追踪采样阶段"""
        traced = self._active_requests.get(request_id)
        if not traced:
            yield None
            return

        span = self._tracer.start_span(
            "sampler.sample",
            context=traced.trace_context,
            attributes={
                "request.id": request_id,
                "sampler.temperature": traced.temperature,
            },
        )
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            span.end()

    # ========================================================
    # 诊断查询: 获取活跃请求追踪信息
    # ========================================================

    def get_active_traces_summary(self) -> List[Dict[str, Any]]:
        """获取当前所有活跃请求的追踪摘要

        用于运维仪表盘实时展示:
        - 哪些请求在执行?
        - 它们分别在什么阶段?
        - 有没有被 preempt 的?
        - 已经生成了多少 token?
        """
        now = time.time()
        summaries = []

        for req_id, traced in self._active_requests.items():
            elapsed_ms = (now - traced.arrival_time) * 1000
            summaries.append({
                "request_id": req_id,
                "phase": traced.current_phase.value,
                "elapsed_ms": round(elapsed_ms, 0),
                "prompt_tokens": traced.prompt_tokens,
                "output_tokens": traced.output_tokens_generated,
                "preemptions": traced.preemption_count,
                "model": traced.model,
                "trace_id": traced.root_span.get_span_context().trace_id,
            })

        return sorted(summaries, key=lambda x: x["elapsed_ms"], reverse=True)

    def get_stats(self) -> Dict[str, Any]:
        """获取追踪器统计信息"""
        return {
            "total_requests_traced": self._total_requests,
            "active_requests": len(self._active_requests),
            "total_preemptions": self._total_preemptions,
            "engine_steps": self._step_count,
        }


# ============================================================
# Async 版本: 适配 vLLM 异步引擎
# ============================================================

class AsyncInferenceTracer(InferenceTracer):
    """异步版推理追踪器

    vLLM 的 AsyncLLMEngine 使用 asyncio, 需要:
    - 异步安全的 span 管理
    - 正确的 context propagation (asyncio.Task 间)
    """

    async def trace_request(
        self,
        request_id: str,
        prompt_tokens: int,
        generate_fn,  # async generator: yields tokens
        **kwargs,
    ):
        """端到端异步请求追踪

        Usage:
            async for token in tracer.trace_request(
                req_id, 1024, engine.generate(prompt, params)
            ):
                yield token  # streaming response
        """
        # 开始追踪
        traced = self.on_request_arrive(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            **kwargs,
        )

        first_token = True
        try:
            async for output in generate_fn:
                if first_token:
                    self.on_prefill_complete(request_id)
                    first_token = False

                self.on_token_generated(
                    request_id=request_id,
                    token_id=output.get("token_id", 0),
                    is_eos=output.get("finish_reason") is not None,
                    batch_size=output.get("batch_size", 0),
                    step_latency_ms=output.get("step_latency_ms", 0),
                )
                yield output

            self.on_request_finish(
                request_id=request_id,
                finish_reason=output.get("finish_reason", "stop"),
            )

        except Exception as e:
            self.on_request_finish(
                request_id=request_id,
                finish_reason="error",
                error=str(e),
            )
            raise


# ============================================================
# Trace→Metric 桥接: 从 Traces 导出 RED Metrics
# ============================================================

class TraceMetricsBridge:
    """从 Trace spans 导出 Prometheus 指标

    Trace 中的信息可以转化为高维度 Metrics:
    - 按 client_id 统计 TTFT (Metrics 做不到, 因为 vLLM 不暴露 client 维度)
    - 按 prompt_length 分桶统计延迟 (自定义维度)
    - 按 preemption 状态统计 (preempted vs not preempted)

    这是 "Traces as structured logs" 理念的实现。
    """

    def __init__(self):
        try:
            from prometheus_client import Histogram, Counter, Gauge
            self._ttft_by_client = Histogram(
                "inference_ttft_by_client_seconds",
                "TTFT broken down by client",
                ["client_id", "model"],
                buckets=[0.1, 0.25, 0.5, 1, 2, 5, 10],
            )
            self._request_by_preempt = Counter(
                "inference_requests_by_preempt_total",
                "Requests with/without preemption",
                ["preempted", "model"],
            )
            self._tpot_by_prompt_bucket = Histogram(
                "inference_tpot_by_prompt_bucket_seconds",
                "TPOT by prompt length bucket",
                ["prompt_bucket", "model"],
                buckets=[0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15],
            )
            self._enabled = True
        except ImportError:
            logger.warning("prometheus_client not available, metrics bridge disabled")
            self._enabled = False

    def on_request_complete(
        self,
        client_id: str,
        model: str,
        ttft_seconds: float,
        avg_tpot_seconds: float,
        prompt_tokens: int,
        was_preempted: bool,
    ):
        """请求完成时更新 Prometheus 指标"""
        if not self._enabled:
            return

        # TTFT by client
        self._ttft_by_client.labels(
            client_id=client_id, model=model
        ).observe(ttft_seconds)

        # Preemption 计数
        self._request_by_preempt.labels(
            preempted=str(was_preempted).lower(), model=model
        ).inc()

        # TPOT by prompt length bucket
        prompt_bucket = self._get_prompt_bucket(prompt_tokens)
        self._tpot_by_prompt_bucket.labels(
            prompt_bucket=prompt_bucket, model=model
        ).observe(avg_tpot_seconds)

    @staticmethod
    def _get_prompt_bucket(tokens: int) -> str:
        """将 prompt 长度映射到可枚举的桶"""
        if tokens < 256:
            return "0-256"
        elif tokens < 1024:
            return "256-1K"
        elif tokens < 4096:
            return "1K-4K"
        elif tokens < 8192:
            return "4K-8K"
        elif tokens < 16384:
            return "8K-16K"
        else:
            return "16K+"


# ============================================================
# 模拟演示
# ============================================================

if __name__ == "__main__":
    import random

    logging.basicConfig(level=logging.INFO)

    # 初始化追踪器
    config = OTelConfig(
        service_name="vllm-trace-demo",
        otlp_endpoint="http://localhost:4317",
        base_sample_rate=1.0,
    )
    tracer = InferenceTracer(config)

    # ========== 模拟场景 1: 正常请求 ==========
    print("\n=== Scenario 1: Normal Request ===")
    req_id = "req-001"

    tracer.on_request_arrive(
        request_id=req_id,
        prompt_tokens=1024,
        model="Qwen2.5-72B",
    )

    # 被调度 (Prefill)
    time.sleep(0.01)  # 10ms queue time
    tracer.on_request_scheduled(req_id, action="prefill", batch_size=16,
                                 kv_cache_usage=0.65)

    # Prefill 完成
    time.sleep(0.4)   # 400ms prefill
    tracer.on_prefill_complete(req_id, prefix_cache_hit_rate=0.8,
                               computed_tokens=200, cached_tokens=824)

    # Decode 128 tokens
    tracer.on_request_scheduled(req_id, action="decode", batch_size=16)
    for i in range(128):
        time.sleep(0.002)  # 模拟 2ms (加速演示, 实际 ~25ms)
        tracer.on_token_generated(req_id, token_id=random.randint(0, 50000),
                                   is_eos=(i == 127), batch_size=16,
                                   step_latency_ms=25.0)

    tracer.on_request_finish(req_id, finish_reason="stop")

    # ========== 模拟场景 2: 被 Preempt 的请求 ==========
    print("\n=== Scenario 2: Preempted Request ===")
    req_id = "req-002"

    tracer.on_request_arrive(
        request_id=req_id,
        prompt_tokens=8192,  # 长 prompt
        model="Qwen2.5-72B",
    )

    # Prefill
    tracer.on_request_scheduled(req_id, action="prefill", batch_size=8,
                                 kv_cache_usage=0.85)
    time.sleep(0.5)
    tracer.on_prefill_complete(req_id, prefix_cache_hit_rate=0.0,
                               computed_tokens=8192, cached_tokens=0)

    # Decode 50 tokens then preempted
    tracer.on_request_scheduled(req_id, action="decode", batch_size=12)
    for i in range(50):
        time.sleep(0.001)
        tracer.on_token_generated(req_id, token_id=random.randint(0, 50000),
                                   batch_size=12, step_latency_ms=30.0)

    # Preemption!
    tracer.on_preemption(req_id, reason="kv_cache_full", kv_cache_usage=0.95,
                          batch_size=12)
    time.sleep(0.5)  # 被暂停 500ms

    # Resume (swap in)
    tracer.on_swap_in(req_id)
    tracer.on_request_scheduled(req_id, action="decode", batch_size=8)

    # Continue decode
    for i in range(78):
        time.sleep(0.001)
        tracer.on_token_generated(req_id, token_id=random.randint(0, 50000),
                                   is_eos=(i == 77), batch_size=8,
                                   step_latency_ms=28.0)

    tracer.on_request_finish(req_id, finish_reason="stop")

    # 输出统计
    print(f"\nTracer Stats: {tracer.get_stats()}")
    print("Demo complete. Check Jaeger UI at http://localhost:16686")
