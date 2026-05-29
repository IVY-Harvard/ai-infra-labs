# 10 - 生产级推理服务架构设计

## 核心问题

> 如何设计一个生产级的 LLM 推理服务？从请求入口到模型输出，经过哪些组件？

## 全景架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Production LLM Serving Architecture               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Clients (Web/App/API)                                               │
│       │                                                              │
│       ▼                                                              │
│  ┌──────────────────────┐                                           │
│  │    API Gateway        │  认证/限流/路由/协议转换                   │
│  │  (Kong/Nginx/Envoy)   │                                          │
│  └──────────┬───────────┘                                           │
│             │                                                        │
│             ▼                                                        │
│  ┌──────────────────────┐                                           │
│  │   Load Balancer       │  请求分发/健康检查/故障转移                │
│  │  (L4/L7 LB)          │                                          │
│  └──────────┬───────────┘                                           │
│             │                                                        │
│             ▼                                                        │
│  ┌──────────────────────┐                                           │
│  │   Model Router        │  模型选择/版本路由/A-B Test               │
│  │  (Custom Service)     │                                          │
│  └──────┬───────┬───────┘                                           │
│         │       │                                                    │
│         ▼       ▼                                                    │
│  ┌───────────┐ ┌───────────┐                                       │
│  │ Engine A  │ │ Engine B  │  推理引擎集群                           │
│  │ (vLLM)    │ │ (TRT-LLM) │  (vLLM / TRT-LLM / SGLang)           │
│  │ 8×H20    │ │ 8×H20    │                                       │
│  └─────┬─────┘ └─────┬─────┘                                       │
│        │              │                                              │
│        ▼              ▼                                              │
│  ┌──────────────────────────────────────┐                           │
│  │          Monitoring & Observability   │                           │
│  │   Prometheus → Grafana → Alerting     │                           │
│  │   Logging → ELK/Loki                  │                           │
│  │   Tracing → Jaeger/Tempo              │                           │
│  └──────────────────────────────────────┘                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Layer 1: API Gateway

```
职责:
  1. 认证与授权 (API Key / OAuth / JWT)
  2. 限流 (Rate Limiting) — 防止过载
  3. 协议转换 (HTTP/gRPC/WebSocket → 内部协议)
  4. 请求验证 (Schema 校验, 参数范围)
  5. 请求日志 (审计跟踪)

LLM 特有考虑:
  ┌────────────────────────────────────────────┐
  │ Token-Based Rate Limiting:                  │
  │   不能只限制请求数, 还要限制 token 数!      │
  │                                             │
  │   一个 128K prompt 的请求 = 100 个短请求     │
  │   → 按 tokens/minute 限流                   │
  │                                             │
  │ Streaming 支持:                             │
  │   SSE (Server-Sent Events) 长连接           │
  │   Gateway 需要支持长连接不超时               │
  │   设置合理的超时 (如 5 分钟)                │
  │                                             │
  │ Content Filtering:                          │
  │   输入/输出内容安全检查                     │
  │   敏感信息过滤 (PII detection)              │
  └────────────────────────────────────────────┘
```

## Layer 2: Load Balancer

```
LLM 负载均衡的特殊性:

1. 请求不等权:
   短 prompt + 短 output: 100ms
   长 prompt + 长 output: 60s
   → 不能简单 Round-Robin!

2. 有状态性:
   KV Cache 在特定 GPU 上
   → 多轮对话应该路由到同一实例 (Session Affinity)

3. GPU 利用率感知:
   某些实例 GPU 满载, 某些空闲
   → 需要根据 GPU 利用率/队列深度路由

推荐策略:

┌────────────────────────────────────────────┐
│  Least-Pending-Requests:                    │
│    路由到排队最少的实例                      │
│    → 自然平衡长短请求的负载                 │
│                                             │
│  Prefix-Aware Routing:                      │
│    相同 System Prompt → 路由到同一实例       │
│    → 利用 Prefix Caching, 避免重复计算       │
│                                             │
│  GPU-Utilization-Aware:                     │
│    监控每个实例的 GPU util / KV Cache 使用率 │
│    → 路由到资源最充裕的实例                  │
└────────────────────────────────────────────┘
```

## Layer 3: Model Router

```
多模型场景 (常见):

┌────────────────────────────────────────────────────┐
│  请求类型          → 路由目标                       │
├────────────────────────────────────────────────────┤
│  简单对话          → 小模型 (7B, 快速响应)          │
│  复杂推理          → 大模型 (70B, 高质量)           │
│  代码生成          → 代码模型 (CodeLlama)           │
│  图片理解          → 视觉模型 (LLaVA)               │
│  高并发低延迟      → 量化模型 (INT4)                │
│  高质量低并发      → 全精度模型 (FP16)              │
└────────────────────────────────────────────────────┘

路由策略:
  1. 基于规则: 请求参数 → 模型映射
  2. 基于模型: 用小模型分类请求 → 路由到合适的大模型
  3. 基于成本: 先尝试小模型, 质量不够 → 降级到大模型

降级策略 (Fallback):
  主模型超时/错误 → 备用模型
  大模型过载 → 降级到小模型 (带提示)
  所有 GPU 满载 → 排队 or 拒绝 (429)
```

## Layer 4: 推理引擎集群

```
8×H20 集群部署方案:

方案 A: 单实例 TP=8 (全部用于一个模型)
┌────────────────────────────────────────┐
│  Instance 0: LLaMA-70B (TP=8, 8×H20) │
│  所有 GPU 协同服务一个大模型           │
│  优点: 最大模型, 最低延迟              │
│  缺点: 无冗余, 灵活性低               │
└────────────────────────────────────────┘

方案 B: 多实例 (多个小模型)
┌────────────────────────────────────────┐
│  Instance 0: Model-7B (TP=1, GPU 0)  │
│  Instance 1: Model-7B (TP=1, GPU 1)  │
│  Instance 2: Model-13B (TP=2, GPU 2-3)│
│  Instance 3: Model-70B (TP=4, GPU 4-7)│
│  优点: 灵活, 多模型并存               │
│  缺点: 大模型性能受限 (TP=4)          │
└────────────────────────────────────────┘

方案 C: 多实例同模型 (高可用)
┌────────────────────────────────────────┐
│  Instance 0: LLaMA-70B (TP=4, GPU 0-3)│
│  Instance 1: LLaMA-70B (TP=4, GPU 4-7)│
│  优点: 冗余, 更高吞吐, 滚动更新      │
│  缺点: TP=4 延迟略高于 TP=8           │
└────────────────────────────────────────┘
```

## Layer 5: 监控与可观测性

### 核心指标

```
┌──────────────────────────────────────────────────────────────┐
│              LLM 推理核心监控指标                               │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  延迟指标:                                                    │
│  ├─ TTFT (Time To First Token): 首 token 延迟                │
│  │   P50 < 500ms, P99 < 2s (目标值)                          │
│  ├─ TPOT (Time Per Output Token): 每 token 延迟              │
│  │   P50 < 50ms, P99 < 100ms                                │
│  ├─ E2E Latency: 端到端延迟                                  │
│  │   取决于生成长度                                          │
│  └─ Queue Wait Time: 排队等待时间                             │
│      P50 < 100ms, P99 < 1s                                   │
│                                                               │
│  吞吐指标:                                                    │
│  ├─ Requests/s: 每秒完成的请求数                              │
│  ├─ Tokens/s (input): 每秒处理的输入 token                   │
│  ├─ Tokens/s (output): 每秒生成的输出 token                  │
│  └─ Batch Size (avg): 平均批处理大小                          │
│                                                               │
│  资源指标:                                                    │
│  ├─ GPU Utilization (%): GPU 计算利用率                       │
│  ├─ GPU Memory Used (%): 显存使用率                           │
│  ├─ KV Cache Utilization (%): KV Cache 使用率 ← 关键!       │
│  │   > 90% 要警惕 (可能导致抢占)                             │
│  ├─ Num Running Requests: 正在执行的请求数                    │
│  ├─ Num Waiting Requests: 排队中的请求数                      │
│  └─ Num Swapped Requests: 被抢占的请求数                      │
│                                                               │
│  业务指标:                                                    │
│  ├─ Success Rate (%): 请求成功率                              │
│  ├─ Error Rate (by type): 错误率和类型                        │
│  └─ Cost per Token: 每 token 成本                             │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Prometheus + Grafana 监控

```
vLLM 内置 Prometheus metrics:

# 延迟
vllm:time_to_first_token_seconds  (histogram)
vllm:time_per_output_token_seconds  (histogram)
vllm:e2e_request_latency_seconds  (histogram)

# 吞吐
vllm:generation_tokens_total  (counter)
vllm:prompt_tokens_total  (counter)

# 资源
vllm:gpu_cache_usage_perc  (gauge)  ← KV Cache 使用率
vllm:cpu_cache_usage_perc  (gauge)
vllm:num_requests_running  (gauge)
vllm:num_requests_waiting  (gauge)
vllm:num_requests_swapped  (gauge)

# 告警规则 (Prometheus AlertManager)
- KV Cache > 95% for 5min → 扩容或限流
- TTFT P99 > 5s → 检查 Prefill 性能
- Error Rate > 1% → 紧急告警
- Queue Depth > 100 for 10min → 扩容
```

## 高可用设计

```
┌──────────────────────────────────────────────────────┐
│  高可用策略                                           │
├──────────────────────────────────────────────────────┤
│                                                       │
│  1. 多实例部署:                                       │
│     至少 2 个推理实例 → 一个挂了另一个接管              │
│                                                       │
│  2. 健康检查:                                         │
│     /health 端点 → 检查 GPU 状态 + 模型加载           │
│     不健康 → LB 自动摘除                              │
│                                                       │
│  3. 优雅关闭 (Graceful Shutdown):                     │
│     收到 SIGTERM → 停止接受新请求                      │
│     等待进行中的请求完成 (drain)                       │
│     超时后强制关闭                                    │
│                                                       │
│  4. 滚动更新 (Rolling Update):                        │
│     新版本部署 → 逐个替换实例                          │
│     K8s 支持: maxUnavailable=0, maxSurge=1            │
│                                                       │
│  5. 模型预热 (Warmup):                                │
│     新实例启动后 → 发送 dummy 请求预热                 │
│     CUDA Graph 编译等一次性开销                        │
│     预热完成后才加入 LB                                │
│                                                       │
│  6. 降级策略:                                         │
│     全部过载 → 返回缓存结果 / 降级到小模型            │
│     GPU 故障 → 自动迁移到备用节点                     │
│                                                       │
└──────────────────────────────────────────────────────┘
```

## 成本优化

```
推理成本优化策略:

1. 右侧对齐 (Right-Sizing):
   不要用 70B 模型做简单任务
   简单 → 7B, 中等 → 13B, 复杂 → 70B
   成本差: 10x

2. 量化:
   FP16 → FP8: GPU 数量减半或吞吐翻倍
   FP16 → INT4: 更激进的节省

3. Prefix Caching:
   System Prompt 复用 → 减少重复 Prefill
   TTFT 降低 2-5x (for common prompts)

4. 动态扩缩容:
   高峰: scale out → 更多实例
   低谷: scale in → 节省成本
   K8s HPA 基于 GPU 利用率 / 队列深度

5. Spot/Preemptible Instances:
   非关键推理 → 用 spot GPU (便宜 60-70%)
   需要处理抢占 (save & resume)
```

## 知识要点框架

### "如何设计一个生产级 LLM 推理服务？"

```
"生产级 LLM 推理服务分五层:

1. API Gateway: 认证限流, 特别要做 token-based rate limiting
2. Load Balancer: 基于队列深度的负载均衡, 支持 Session Affinity
3. Model Router: 多模型路由 + 降级策略
4. Inference Engine: vLLM/TRT-LLM + PagedAttention + Continuous Batching
5. Monitoring: TTFT/TPOT/KV Cache 使用率是核心指标

关键设计点:
- 流式输出: SSE 全链路支持
- 高可用: 多实例 + 健康检查 + 优雅关闭
- 成本: 模型分级 + 量化 + 动态扩缩容
- 监控: KV Cache > 95% 是最关键的告警阈值"
```

## 小结

| 层次 | 组件 | 核心职责 | LLM 特有考虑 |
|------|------|----------|-------------|
| Gateway | Kong/Nginx | 认证限流 | Token-based 限流 |
| LB | L4/L7 | 请求分发 | Queue-depth 感知 |
| Router | Custom | 模型选择 | 多模型降级 |
| Engine | vLLM等 | 推理计算 | PA + CB |
| Monitor | Prometheus | 可观测性 | TTFT/TPOT/KV Cache |
