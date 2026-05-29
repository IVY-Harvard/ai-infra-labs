# Lab 10: AIOps 自动化运维

## 概述

AIOps = AI + Ops, 将机器学习应用于运维自动化:
1. **自动修复** (Auto-Remediation): 检测到异常后自动执行修复动作
2. **根因分析** (Root Cause Analysis): 从告警风暴中定位真正的根因
3. **ChatOps**: 通过自然语言与运维系统交互

## 文件结构

```
10_aiops_automation/
├── README.md                  ← 本文件
├── auto_remediation.py       ← 自动修复引擎 (规则+ML)
├── root_cause_analyzer.py    ← 告警根因分析
└── chatops_bot.py            ← ChatOps 机器人
```

## 自动修复决策树

```
告警触发
│
├── KV Cache > 95%
│   ├── 有排队请求?
│   │   ├── YES → Action: 限流 + 触发扩容
│   │   └── NO  → Action: 等待 (可能是长请求即将完成)
│   └── 持续 > 5min?
│       └── YES → Action: 强制限流 + 紧急扩容
│
├── TTFT P99 > 10s
│   ├── KV Cache > 90%?
│   │   └── YES → Root cause: KV Cache → Action: 扩容
│   ├── GPU Throttle?
│   │   └── YES → Root cause: 温度 → Action: 降低负载
│   └── Request spike?
│       └── YES → Root cause: 流量突增 → Action: 限流 + 扩容
│
├── GPU XID Error
│   ├── XID 48 (ECC DBE)?
│   │   └── Action: 立即摘除 GPU, 通知 Infra
│   ├── XID 79 (Fallen off bus)?
│   │   └── Action: 尝试 GPU Reset, 失败则摘除节点
│   └── 其他 XID?
│       └── Action: 记录 + 监控频率
│
├── 吞吐为 0 但有排队
│   └── Action: 检查 GPU 状态 → 重启 vLLM Worker
│
└── Spot 中断通知
    └── Action: 执行 graceful drain + 请求迁移
```

## 运行前提

```bash
pip install numpy aiohttp
```
