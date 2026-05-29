# Lab 02 - DCGM Exporter 深度配置：从默认到生产级

## 目标

通过本实验，读者将：
1. 掌握 DCGM Exporter 的完整部署流程（DaemonSet + ServiceMonitor）
2. 自定义 Profiling Metrics 采集清单，只采集对推理场景真正有用的指标
3. 编写高级 PromQL 查询实现多维度 GPU 分析
4. 理解采集频率与 GPU 开销的 trade-off

## 前置条件

- Kubernetes 集群已部署（推荐 1.26+）
- Prometheus Operator 已安装（kube-prometheus-stack）
- NVIDIA GPU Operator 或手动安装 DCGM 2.4+
- 至少 1 个 GPU 节点

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│  GPU Node                                                │
│  ┌──────────────┐    ┌───────────────────┐              │
│  │  DCGM Daemon │◄──►│  NVIDIA Driver    │              │
│  │  (nv-hostengine)  │  (535.x+)         │              │
│  └──────┬───────┘    └───────────────────┘              │
│         │ gRPC                                           │
│  ┌──────▼───────┐                                       │
│  │ DCGM Exporter│ ← custom_metrics.csv 控制采集哪些指标  │
│  │  :9400/metrics│                                      │
│  └──────┬───────┘                                       │
│         │ HTTP scrape                                    │
└─────────┼───────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────┐
│  Prometheus                                              │
│  ┌─────────────────┐    ┌──────────────┐                │
│  │ ServiceMonitor  │───►│  TSDB        │                │
│  │ (自动发现)       │    │  (15s scrape)│                │
│  └─────────────────┘    └──────┬───────┘                │
│                                │                         │
│  ┌─────────────────────────────▼──────────────────┐     │
│  │  Recording Rules (预计算常用聚合)                │     │
│  └─────────────────────────────┬──────────────────┘     │
└────────────────────────────────┼────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────┐
│  Grafana Dashboards                                      │
│  - GPU Real Utilization                                  │
│  - SM Occupancy Trends                                   │
│  - Health & Thermal                                      │
└──────────────────────────────────────────────────────────┘
```

## 实验内容

### Part 1: 自定义采集指标

默认 DCGM Exporter 采集 ~60 个指标，大部分对推理场景无用。
我们精简为 ~20 个核心指标，减少 Prometheus 存储压力。

参考 `custom_metrics.csv` 文件，它定义了：
- Profiling Metrics（性能关键）
- Health Metrics（健康监控）
- Clock/Memory Metrics（资源追踪）

### Part 2: Kubernetes 部署

参考 `dcgm_exporter_setup.yaml`，包含：
- DaemonSet（确保每个 GPU 节点都有 Exporter）
- ServiceMonitor（Prometheus Operator 自动发现）
- ConfigMap（挂载 custom_metrics.csv）
- Recording Rules（预计算常用聚合指标）

```bash
# 部署
kubectl apply -f dcgm_exporter_setup.yaml

# 验证
kubectl get pods -n gpu-monitoring -l app=dcgm-exporter
kubectl port-forward -n gpu-monitoring svc/dcgm-exporter 9400:9400
curl localhost:9400/metrics | grep DCGM_FI_PROF
```

### Part 3: 高级 PromQL 查询

参考 `advanced_queries.md`，包含：
- 真实计算效率计算
- 多 GPU 负载均衡分析
- 异常检测查询
- 容量规划查询

## 关键配置项

| 配置 | 默认值 | 推荐值 | 说明 |
|------|--------|--------|------|
| 采集间隔 | 30s | 15s | Profiling Metrics 需要更高频率 |
| Profiling 模式 | 关闭 | 开启 | 需要 DCGM 2.4+ |
| 指标数量 | ~60 | ~20 | 精简后减少 70% 存储 |
| GPU 开销 | < 1% | < 1% | Profiling 对 GPU 性能影响极小 |

## 注意事项

1. **Profiling Metrics 需要 DCGM 2.4+**：老版本只有 Health Metrics
2. **MIG 环境**：MIG 模式下 Profiling Metrics 按 GPU Instance 报告
3. **vGPU 环境**：某些 Profiling Metrics 在 vGPU 下不可用
4. **采集频率**：低于 10s 可能导致 DCGM 内部 buffer 竞争

## 预期产出

- 一个在 K8s 中运行的 DCGM Exporter，只采集推理相关指标
- Prometheus 中可查询到所有 Profiling Metrics
- 预计算的 Recording Rules 加速 Dashboard 查询
