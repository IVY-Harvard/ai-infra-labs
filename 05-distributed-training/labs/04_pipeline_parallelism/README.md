# Lab 04: 流水线并行 — GPipe / 1F1B 调度

## 目标

1. 理解朴素流水线的 bubble 问题
2. 实现 GPipe 调度，观察 bubble 率
3. 实现 1F1B 调度，对比显存优势
4. 可视化不同调度策略的 GPU 利用率

## 前置知识

- 对应理论：[theory/04_pipeline_parallelism.md](../../theory/04_pipeline_parallelism.md)
- Bubble 率公式：`(p-1) / (m + p - 1)`

## 文件说明

| 文件 | 说明 |
|------|------|
| `naive_pipeline.py` | 朴素流水线：顺序执行，观察 bubble |
| `gpipe_schedule.py` | GPipe 调度：所有前向 → 所有反向 |
| `1f1b_schedule.py` | 1F1B 调度：warmup → steady → cooldown |
| `bubble_visualization.py` | 可视化各调度策略的时间线 |

## 实验步骤

```bash
# 朴素流水线（观察巨大 bubble）
torchrun --nproc_per_node=4 naive_pipeline.py --micro-batches 8

# GPipe
torchrun --nproc_per_node=4 gpipe_schedule.py --micro-batches 8

# 1F1B
torchrun --nproc_per_node=4 1f1b_schedule.py --micro-batches 8

# 可视化（在 rank 0 生成图表）
torchrun --nproc_per_node=4 bubble_visualization.py
```

## 思考题

1. GPipe 和 1F1B 的 bubble 率公式相同，那 1F1B 的优势在哪里？
2. micro-batch 数量 m 越大越好吗？会有什么副作用？
3. Interleaved 1F1B 如何进一步减少 bubble？代价是什么？
