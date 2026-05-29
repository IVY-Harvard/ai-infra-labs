# Lab 09: GPU Spot Instance 策略

## 概述

Spot Instance 可节省 60-70% GPU 成本, 但面临随时被中断的风险。
本 Lab 实现 GPU 推理服务的 Spot 策略:

1. **Spot Advisor**: 选择中断概率最低的实例类型/可用区
2. **Interruption Handler**: 优雅处理 Spot 中断 (保存 KV Cache, 迁移请求)
3. **Cost Comparison**: On-Demand vs Reserved vs Spot 混合策略

## 文件结构

```
09_spot_strategy/
├── README.md                  ← 本文件
├── spot_advisor.py           ← Spot 选型与可用区建议
├── interruption_handler.py   ← 中断处理与请求迁移
└── cost_comparison.py        ← 混合部署成本对比
```

## Spot 策略设计原则

```
┌─────────────────────────────────────────────────────────────┐
│                GPU Spot 策略分层                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Tier 1: On-Demand (保底容量)                                │
│  ├── 承载: 最低保障 QPS (SLO 关键业务)                      │
│  ├── 比例: 30-40% 总容量                                    │
│  └── 特点: 永不中断, 成本最高                                │
│                                                             │
│  Tier 2: Reserved Instance (主力容量)                        │
│  ├── 承载: 可预测的稳态负载                                  │
│  ├── 比例: 40-50% 总容量                                    │
│  └── 特点: 1-3 年承诺, 30-50% 折扣                          │
│                                                             │
│  Tier 3: Spot Instance (弹性容量)                            │
│  ├── 承载: 峰值溢出 + Batch 任务                            │
│  ├── 比例: 10-30% 总容量                                    │
│  └── 特点: 60-70% 折扣, 可能被中断                          │
│                                                             │
│  容量保障: Tier1 + Tier2 >= SLO 最低保障                     │
│  成本优化: Tier3 在有余量时承接 Standard/Batch 请求           │
└─────────────────────────────────────────────────────────────┘
```

## 中断处理流程

```
Spot 中断通知 (通常提前 2 分钟)
    │
    ├── 1. 停止接受新请求 (从 LB 摘除, 5s)
    │
    ├── 2. 等待短请求完成 (预计 < 30s 的请求)
    │
    ├── 3. 长请求迁移
    │   ├── 导出当前 context (request_id, prompt, generated_tokens)
    │   ├── 发送到健康实例继续生成
    │   └── 客户端透明 (streaming 短暂中断后恢复)
    │
    ├── 4. 清理资源 (释放 GPU 显存, 关闭连接)
    │
    └── 5. 确认 shutdown (允许实例回收)
```
