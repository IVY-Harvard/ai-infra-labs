# Lab 06: GPU 推理指标异常检测

## 概述

传统阈值告警 (如 "TTFT > 5s") 只能捕捉已知问题。
异常检测 (Anomaly Detection) 能发现 **你不知道应该监控什么** 的问题。

例如:
- TTFT 从稳定 200ms 缓慢漂移到 800ms (没到 5s 阈值但已异常)
- 每天凌晨 3 点 GPU SM Active 出现 10 分钟周期性抖动
- 某个 client 的请求模式突然变化导致 KV Cache 效率下降

本 Lab 实现三层异常检测体系:
1. **统计检测** (statistical_detector.py): Z-score, MAD, EWMA — 快速、低资源
2. **ML 检测** (ml_detector.py): Isolation Forest, Autoencoder — 捕捉复杂模式
3. **关联分析** (correlation_analyzer.py): 多指标协同异常检测

---

## 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Anomaly Detection Pipeline                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Prometheus ──→ Metric Fetcher ──→ ┌─────────────────────────┐     │
│                                     │   Detection Engines      │     │
│                                     │                         │     │
│                 Time Series Data     │  ┌───────────────────┐ │     │
│                 (TTFT, TPOT, KV,    │  │ Statistical Layer │ │     │
│                  GPU metrics...)     │  │ • Z-Score         │ │     │
│                        │            │  │ • MAD             │ │     │
│                        │            │  │ • EWMA            │ │     │
│                        ▼            │  │ • Grubbs Test     │ │     │
│                  ┌──────────┐       │  └───────┬───────────┘ │     │
│                  │ Feature  │       │          │             │     │
│                  │ Engineer │──────→│  ┌───────▼───────────┐ │     │
│                  │          │       │  │    ML Layer       │ │     │
│                  └──────────┘       │  │ • Isolation Forest│ │     │
│                                     │  │ • Autoencoder     │ │     │
│                                     │  │ • DBSCAN          │ │     │
│                                     │  └───────┬───────────┘ │     │
│                                     │          │             │     │
│                                     │  ┌───────▼───────────┐ │     │
│                                     │  │ Correlation Layer │ │     │
│                                     │  │ • Cross-metric    │ │     │
│                                     │  │ • Temporal pattern│ │     │
│                                     │  │ • Causal graph    │ │     │
│                                     │  └───────┬───────────┘ │     │
│                                     └──────────┼─────────────┘     │
│                                                │                    │
│                                                ▼                    │
│                                     ┌───────────────────────┐      │
│                                     │  Anomaly Aggregator   │      │
│                                     │  • Score fusion       │      │
│                                     │  • Dedup & suppress   │      │
│                                     │  • Context enrich     │      │
│                                     └───────────┬───────────┘      │
│                                                 │                   │
│                                    ┌────────────┼────────────┐     │
│                                    ▼            ▼            ▼     │
│                              ┌──────────┐ ┌──────────┐ ┌────────┐ │
│                              │Alertmgr  │ │ Grafana  │ │ Slack  │ │
│                              │(severity)│ │(annotate)│ │(notify)│ │
│                              └──────────┘ └──────────┘ └────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 文件结构

```
06_anomaly_detection/
├── README.md                     ← 本文件
├── statistical_detector.py       ← 统计学异常检测 (Z-score, MAD, EWMA)
├── ml_detector.py                ← ML 异常检测 (Isolation Forest, Autoencoder)
└── correlation_analyzer.py       ← 多指标关联分析
```

---

## 检测方法对比

| 方法 | 优势 | 劣势 | 适用场景 |
|------|------|------|---------|
| Z-Score | 简单快速, 可解释 | 假设正态分布 | 稳态指标 (TPOT) |
| MAD | 对离群值鲁棒 | 计算量稍大 | 有尖峰的指标 |
| EWMA | 适应趋势变化 | 参数敏感 | 缓慢漂移检测 |
| Isolation Forest | 多维异常 | 需要训练 | 复合异常模式 |
| Autoencoder | 学习复杂模式 | 需要大量数据 | 时序形态异常 |
| 关联分析 | 发现因果关系 | 复杂度高 | 根因定位 |

---

## 关键概念

### GPU 推理指标的异常类型

```
1. 点异常 (Point Anomaly)
   正常: 200ms, 210ms, 195ms, 205ms
   异常:              ↓
   200ms, 210ms, 5000ms, 205ms
   → Z-Score / MAD 检测

2. 上下文异常 (Contextual Anomaly)
   工作日: TTFT P99 ≈ 500ms
   周末:   TTFT P99 ≈ 200ms
   异常: 周末 TTFT P99 = 500ms (工作日正常但周末异常)
   → 需要考虑时间上下文

3. 集体异常 (Collective Anomaly)
   正常: 200ms, 210ms, 195ms, ...
   异常: 400ms, 420ms, 380ms, 410ms, ... (持续偏高但未到阈值)
   → EWMA / Change Point Detection

4. 关联异常 (Correlated Anomaly)
   KV Cache ↑ + TTFT ↑ → 正常关联
   KV Cache ↑ + TTFT ↓ → 关联打破 → 异常!
   → Correlation Analyzer
```

---

## 运行前提

```bash
pip install numpy pandas scikit-learn torch prometheus-api-client
```
