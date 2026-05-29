# Lab 01: Prompt Injection 攻防实验

## 实验目标

1. 理解 Prompt Injection 的攻击原理和分类
2. 实现常见的注入攻击演示（教育目的）
3. 掌握多种防御策略并评估其效果
4. 构建红队测试套件用于安全评估

## 前置知识

- 理论课 `02_prompt_injection.md` 内容
- Python 基础、HTTP API 调用经验
- 对 LLM 推理流程有基本了解

## 实验环境

```
硬件：8x NVIDIA H20 GPU
模型：Llama-3-8B-Instruct（通过 vLLM 部署）
框架：Python 3.10+, FastAPI, httpx
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `injection_demo.py` | 各类注入攻击的演示代码 |
| `defense_strategies.py` | 防御策略实现与评估 |
| `red_team_suite.py` | 自动化红队测试套件 |

## 实验步骤

### Step 1: 观察注入攻击效果

```bash
# 启动 vLLM 服务（假设已部署）
# python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3-8B-Instruct

# 运行注入演示
python injection_demo.py --target http://localhost:8000/v1
```

### Step 2: 部署防御策略

```bash
python defense_strategies.py --mode evaluate --target http://localhost:8000/v1
```

### Step 3: 红队测试

```bash
python red_team_suite.py --target http://localhost:8000/v1 --report results.json
```

## 注意事项

- 本实验仅用于安全研究和防御改进
- 不要对未授权的系统使用这些技术
- 红队测试结果应妥善保管，不要公开传播
- 实验中的攻击 payload 仅作为防御参考

## 预期输出

完成实验后，应能够：
1. 区分直接注入和间接注入
2. 评估不同防御策略的误报率和漏报率
3. 设计适合业务场景的防御组合方案
