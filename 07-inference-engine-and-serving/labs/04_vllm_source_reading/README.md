# Lab 04: vLLM 源码阅读指南

## 目标

从"会用 vLLM"进阶到"理解 vLLM"。通过结构化的源码阅读，理解核心设计决策。

## 阅读顺序

1. **architecture_map.md** — 模块关系图，先建立全局视角
2. **request_lifecycle.md** — 跟踪一个请求的完整生命周期
3. **scheduler_analysis.md** — Scheduler 核心逻辑分析
4. **block_manager_analysis.md** — BlockManager 源码分析

## 源码阅读建议

```bash
# 克隆 vLLM 源码
git clone https://github.com/vllm-project/vllm.git
cd vllm

# 建议使用 IDE (VS Code / PyCharm) 阅读
# 利用"跳转到定义"功能追踪调用链

# 核心文件 (按重要性排序):
# 1. vllm/core/scheduler.py          ← 最核心
# 2. vllm/core/block_manager.py      ← 最核心
# 3. vllm/engine/llm_engine.py       ← 引擎主循环
# 4. vllm/worker/model_runner.py     ← 模型执行
# 5. vllm/worker/worker.py           ← GPU Worker
```
