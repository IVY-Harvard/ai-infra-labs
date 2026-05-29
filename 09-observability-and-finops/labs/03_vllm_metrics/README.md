# Lab 03 - vLLM Metrics 全景监控：从指标到决策

## 目标

通过本实验，读者将：
1. 掌握 vLLM 暴露的所有核心指标及其业务含义
2. 构建生产级 Grafana Dashboard（4 个维度）
3. 设计分层告警规则（GPU / 推理 / 容量）
4. 理解 KV Cache、Scheduling、Preemption 指标的深层关联

## 前置条件

- vLLM 0.4+ 已部署并开启 Prometheus metrics endpoint
- Prometheus 已配置 scrape vLLM `/metrics` 端点
- Grafana 已连接 Prometheus 数据源
- 完成 Lab 01 / Lab 02（理解 DCGM 指标基础）

## 架构概览

```
┌───────────────────────────────────────────────────────────────┐
│  vLLM Engine (per instance)                                    │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  LLMEngine                                               │  │
│  │  ├── Scheduler ──────────── scheduling metrics           │  │
│  │  │   ├── running / waiting / swapped                     │  │
│  │  │   ├── preemption_count                                │  │
│  │  │   └── num_batched_tokens                              │  │
│  │  ├── KV Cache Manager ──── cache metrics                 │  │
│  │  │   ├── gpu_cache_usage_perc                            │  │
│  │  │   ├── cpu_cache_usage_perc                            │  │
│  │  │   └── cache_hit_rate (prefix caching)                 │  │
│  │  ├── Model Executor ────── latency metrics               │  │
│  │  │   ├── time_to_first_token                             │  │
│  │  │   ├── time_per_output_token                           │  │
│  │  │   ├── e2e_request_latency                             │  │
│  │  │   └── model_execute_time                              │  │
│  │  └── Token Counter ─────── throughput metrics            │  │
│  │      ├── generation_tokens_total                         │  │
│  │      ├── prompt_tokens_total                             │  │
│  │      └── request_success / request_failure               │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
│  :8000/metrics  ←── Prometheus scrape                          │
└───────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  Prometheus                                  │
│  ├── Recording Rules (预聚合)               │
│  └── Alert Rules (分层告警)                 │
│      ├── gpu_alerts.yaml                    │
│      ├── inference_alerts.yaml              │
│      └── capacity_alerts.yaml              │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  Grafana Dashboards                          │
│  ├── overview.json      (全局总览)          │
│  ├── per_model.json     (模型级分析)        │
│  ├── kv_cache.json      (KV Cache 深度)     │
│  └── latency.json       (延迟剖析)          │
└─────────────────────────────────────────────┘
```

## 实验内容

### Part 1: vLLM 指标体系 (metrics_catalog.md)

全面梳理 vLLM 暴露的指标，按功能域分类：
- **吞吐量指标**：tokens/s, requests/s
- **延迟指标**：TTFT, TPOT, E2E latency (histogram)
- **调度指标**：running/waiting/swapped, preemption
- **KV Cache 指标**：usage%, hit rate, eviction
- **资源指标**：GPU memory, batch utilization

### Part 2: Grafana Dashboard 构建 (grafana_dashboards/)

4 个生产级 Dashboard JSON 模板：
- `overview.json` — SRE/On-call 一眼看清系统状态
- `per_model.json` — 按模型对比性能与资源
- `kv_cache.json` — KV Cache 深度分析与调优指导
- `latency.json` — 延迟分位数、TTFT 趋势、异常检测

### Part 3: 分层告警规则 (alert_rules/)

基于严重度和职责分离的告警设计：
- `gpu_alerts.yaml` — GPU 硬件层（SRE/Infra 响应）
- `inference_alerts.yaml` — 推理服务层（ML Platform 响应）
- `capacity_alerts.yaml` — 容量规划层（提前 N 小时预警）

## vLLM Metrics 端点配置

```python
# vLLM 启动参数
# 默认在 :8000/metrics 暴露 Prometheus 格式指标
python -m vllm.entrypoints.openai.api_server \
    --model /models/qwen2.5-72b \
    --tensor-parallel-size 8 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --disable-log-requests       # 生产环境关闭请求日志
```

```yaml
# Prometheus scrape 配置 (ServiceMonitor)
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: vllm-metrics
  namespace: inference
spec:
  selector:
    matchLabels:
      app: vllm
  endpoints:
    - port: http
      path: /metrics
      interval: 15s
```

## 关键知识点

| 概念 | 说明 | 对应指标 |
|------|------|----------|
| TTFT | 从收到请求到第一个 token 生成的时间 | `vllm:time_to_first_token_seconds` |
| TPOT | 每个输出 token 的生成时间 | `vllm:time_per_output_token_seconds` |
| KV Cache 压力 | GPU cache 使用率高于 90% 时开始 preemption | `vllm:gpu_cache_usage_perc` |
| Preemption | 显存不足时暂停低优先级请求 | `vllm:num_preemptions_total` |
| Prefix Caching | 复用相同 prefix 的 KV Cache，减少重复计算 | `vllm:prefix_cache_hit_rate` |

## 注意事项

1. **版本差异**：vLLM 0.4 vs 0.5 部分指标名有变化，本 Lab 以 0.5+ 为准
2. **多实例聚合**：生产环境通常有多个 vLLM instance，需要 label 区分
3. **Histogram 精度**：TTFT/TPOT 用 histogram，bucket 配置影响 P99 精度
4. **性能影响**：metrics 采集本身开销可忽略（< 0.1% CPU）

## 预期产出

- 理解 vLLM 每个核心指标的含义和告警阈值
- 4 个可直接导入 Grafana 的 Dashboard JSON
- 3 套告警规则覆盖 GPU 硬件 / 推理服务 / 容量规划
- 能基于指标做出运维决策（扩容、调参、排障）
