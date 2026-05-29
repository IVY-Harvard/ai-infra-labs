# Module 09: Observability & FinOps for AI Infrastructure

## 定位：从基础监控到企业级可观测性

### 前置知识

本模块假设读者已具备：
- NVIDIA GPU 的实际运维经验
- Prometheus + Grafana 监控体系搭建能力
- vLLM 指标监控的基本经验
- 对 GPU 利用率监控的基本理解

### 本模块的提升方向

- `nvidia-smi` 的 utilization 为什么会误导人？如何获取真实利用率
- 如何从零散的指标采集升级为完整的 SLI/SLO 体系
- 如何实现 AI 推理服务的全链路追踪
- 如何建立系统的容量规划方法论
- 如何实现精细的多租户成本分摊

---

## 模块结构

### 理论篇 (`theory/`)

| # | 主题 | 核心要点 |
|---|------|---------|
| 01 | 可观测性三支柱 | Metrics/Logs/Traces 在 AI Infra 的特殊落地方式 |
| 02 | GPU 指标的真相 | nvidia-smi 的误导、DCGM 真指标体系 |
| 03 | 推理服务 SLI/SLO | TTFT/TPOT/吞吐量，行业标准参考 |
| 04 | 分布式追踪 | AI 推理全链路追踪实现 |
| 05 | 容量规划 | SLO 反推 GPU 需求，弹性策略 |
| 06 | AI FinOps | GPU 成本模型与分摊策略 |
| 07 | AIOps 自动化 | 用 AI 运维 AI Infra |

### 实验篇 (`labs/`)

| # | 实验 | 产出 |
|---|------|------|
| 01 | GPU 指标深潜 | 真实利用率计算、SM Occupancy 监控 |
| 02 | DCGM Exporter 高级配置 | 自定义指标采集、高级查询 |
| 03 | vLLM 指标全景 | 完整 Dashboard、告警规则体系 |
| 04 | 分布式追踪 | OpenTelemetry + Jaeger 集成 |
| 05 | 日志聚合 | Loki + Promtail 方案 |
| 06 | 异常检测 | 统计方法 + ML 方法 |
| 07 | 容量规划 | SLO→GPU 计算器、流量预测 |
| 08 | 成本核算 | GPU 成本模型、分摊计算 |
| 09 | Spot 策略 | Spot 中断处理、成本对比 |
| 10 | AIOps 自动化 | 自动修复、根因分析、ChatOps |

### 项目篇 (`project/`)

**AI 全栈可观测平台** — 整合所有 Lab 产出，构建一个可落地的企业级平台。

---

## 学习路径建议

```
Week 1: Theory 01-03 + Lab 01-03（夯实指标体系）
Week 2: Theory 04-05 + Lab 04-05 + Lab 07（追踪 + 容量）
Week 3: Theory 06-07 + Lab 06 + Lab 08-09（FinOps + 异常检测）
Week 4: Lab 10 + Project（整合为企业平台）
```

## 前置要求

- Linux 系统管理经验
- Prometheus + Grafana 使用经验
- Docker/Kubernetes 基础
- Python 编程能力
- vLLM 部署与运维经验

## 预期产出

完成本模块后，读者将能够：
1. 建立完整的 AI Infra 可观测性体系（不止监控）
2. 准确解读 GPU 指标，避免常见误区
3. 设计并实施推理服务 SLI/SLO 体系
4. 实现全链路分布式追踪
5. 进行科学的 GPU 容量规划
6. 建立多维度成本分摊模型
7. 实现 AIOps 自动化运维闭环
