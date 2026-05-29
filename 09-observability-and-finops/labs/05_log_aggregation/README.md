# Lab 05: GPU 推理服务日志聚合与分析

## 概述

在 GPU 推理平台中，日志是最原始但最详细的可观测信号。
当 Metrics 告诉你 "TTFT P99 突然升到 10s"，Traces 告诉你 "某个请求在 scheduler.queue 卡了 8s"，
Logs 告诉你 **"因为 GPU 3 出现了 XID Error 48 导致 Worker 重启"**。

本 Lab 基于 Grafana Loki + Promtail 构建 GPU 推理日志聚合系统。

---

## 架构图

```
┌────────────────────────────────────────────────────────────────────────┐
│                     GPU Inference Log Pipeline                         │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Log Sources (每个节点)                        │   │
│  │                                                                 │   │
│  │  vLLM Engine ──→ stdout/stderr ──→ /var/log/pods/vllm-*.log    │   │
│  │    • Request lifecycle events                                   │   │
│  │    • Scheduler decisions (prefill/decode/preempt)               │   │
│  │    • KV Cache operations                                       │   │
│  │    • Model loading progress                                     │   │
│  │                                                                 │   │
│  │  NVIDIA Driver ──→ dmesg/syslog ──→ /var/log/syslog            │   │
│  │    • XID Errors (31, 48, 63, 79...)                             │   │
│  │    • GPU Reset events                                           │   │
│  │    • NVLink errors                                              │   │
│  │    • ECC memory errors                                          │   │
│  │                                                                 │   │
│  │  NCCL (TP 通信) ──→ stderr ──→ /var/log/pods/vllm-*.log       │   │
│  │    • AllReduce timeout                                          │   │
│  │    • Ring/Tree topology                                         │   │
│  │    • Transport selection (NVLink/PCIe)                          │   │
│  │                                                                 │   │
│  │  Kubernetes ──→ /var/log/pods/*                                 │   │
│  │    • Pod lifecycle (start/stop/restart)                          │   │
│  │    • OOM Kill events                                            │   │
│  │    • Node pressure events                                       │   │
│  └──────────────────────────────┬──────────────────────────────────┘   │
│                                 │                                      │
│                                 ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    Promtail (DaemonSet)                           │  │
│  │                                                                  │  │
│  │  ┌─────────┐   ┌─────────────────┐   ┌─────────────────────┐   │  │
│  │  │ Discover │──→│ Pipeline Stages │──→│ Push to Loki        │   │  │
│  │  │ targets  │   │ • regex parse   │   │ • batch + compress  │   │  │
│  │  │ (K8s SD) │   │ • json parse    │   │ • retry on failure  │   │  │
│  │  │          │   │ • label extract  │   │                     │   │  │
│  │  │          │   │ • timestamp      │   │                     │   │  │
│  │  │          │   │ • multiline      │   │                     │   │  │
│  │  └─────────┘   └─────────────────┘   └─────────────────────┘   │  │
│  └──────────────────────────────┬───────────────────────────────────┘  │
│                                 │ HTTP Push                            │
│                                 ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    Grafana Loki (集群)                            │  │
│  │                                                                  │  │
│  │  ┌───────────┐  ┌──────────┐  ┌────────────┐  ┌─────────────┐  │  │
│  │  │Distributor│─→│ Ingester │─→│ Compactor  │─→│ Object Store│  │  │
│  │  │(接收+hash)│  │(WAL+内存)│  │(合并+压缩) │  │ (S3/MinIO)  │  │  │
│  │  └───────────┘  └──────────┘  └────────────┘  └─────────────┘  │  │
│  │        │                                             │          │  │
│  │        ▼                                             │          │  │
│  │  ┌───────────┐                                       │          │  │
│  │  │  Querier  │←──────────────────────────────────────┘          │  │
│  │  │(查询+聚合)│                                                  │  │
│  │  └───────────┘                                                  │  │
│  └──────────────────────────────┬───────────────────────────────────┘  │
│                                 │ LogQL                                │
│                                 ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Grafana                                                         │  │
│  │  • Log exploration (Explore → Loki)                              │  │
│  │  • Log → Trace 关联 (trace_id 跳转 Jaeger)                      │  │
│  │  • Log → Metric 关联 (同时间段 Prometheus 数据)                  │  │
│  │  • Alert on log patterns (Loki Alerting)                         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 文件结构

```
05_log_aggregation/
├── README.md                  ← 本文件
├── loki_setup.yaml           ← Loki + Promtail 完整部署
├── promtail_config.yaml      ← Promtail 采集规则 (GPU 推理特化)
└── log_query_patterns.md     ← LogQL 查询模板与故障排查手册
```

---

## 为什么 GPU 推理日志需要特殊处理？

### 1. 多源异构日志

| 日志源 | 格式 | 关键信息 | 挑战 |
|--------|------|---------|------|
| vLLM Python | JSON/Text 混合 | 请求ID、延迟、token 数 | 多行 traceback |
| NVIDIA Driver | syslog 格式 | XID Error、ECC | dmesg 时间格式 |
| NCCL | stderr 自定义 | AllReduce 状态 | 无结构化 |
| Kubernetes | JSON (CRI) | Pod 事件 | 标签关联 |

### 2. 高吞吐量

- 8 GPU × 数千 requests/min = 大量日志
- vLLM DEBUG 模式: 每个 decode step 一条日志 → 极高写入量
- 需要合理的日志级别策略和采样

### 3. 关联需求

- GPU XID Error → 哪些推理请求受影响？
- OOM Kill → 当时 KV Cache 使用率是多少？
- NCCL Timeout → 哪个 GPU 通信断开？

---

## 日志级别策略

```
┌─────────────────────────────────────────────────────────┐
│              GPU 推理服务日志级别建议                      │
├──────────┬──────────────────────────────────────────────┤
│  ERROR   │ • GPU 硬件错误 (XID, ECC DBE)               │
│          │ • 推理失败 (OOM, CUDA Error)                 │
│          │ • 服务不可用                                  │
├──────────┼──────────────────────────────────────────────┤
│  WARNING │ • KV Cache > 90%                             │
│          │ • Preemption 发生                             │
│          │ • TTFT > SLO 阈值                            │
│          │ • ECC SBE (可纠正错误)                        │
├──────────┼──────────────────────────────────────────────┤
│  INFO    │ • 请求到达/完成 (含关键指标)                  │
│          │ • 模型加载完成                                │
│          │ • 配置变更                                    │
├──────────┼──────────────────────────────────────────────┤
│  DEBUG   │ • 每个 decode step 详情                      │
│          │ • Scheduler 每步决策                          │
│          │ • KV Cache block 分配                         │
│          │ ⚠️ 生产环境建议关闭 (日志量极大)              │
└──────────┴──────────────────────────────────────────────┘
```

---

## 运行前提

- Kubernetes 集群
- Helm 3 (用于部署 Loki)
- Grafana 已部署
- 存储后端: MinIO / S3 (Loki 持久化)
