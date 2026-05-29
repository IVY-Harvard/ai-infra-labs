# Lab 01: KV Cache Fundamentals

## 目标

- 亲手计算 KV Cache 的显存占用
- 理解不同模型参数对 KV Cache 大小的影响
- 理解为什么长上下文是推理的核心挑战

## 实验内容

1. **kv_cache_calculator.py** — KV Cache 显存计算器
   - 输入: 模型参数 (layers, heads, head_dim, dtype)
   - 输入: 推理参数 (seq_len, batch_size, tp_size)
   - 输出: KV Cache 占用、最大 batch size、GPU 利用分析

2. **naive_kv_cache.py** — 朴素 KV Cache 管理实现
   - 连续内存分配
   - 预分配 max_seq_len
   - 演示碎片化问题

## 运行方式

```bash
# 计算器 (不需要 GPU)
python kv_cache_calculator.py

# 朴素 KV Cache (需要 GPU)
python naive_kv_cache.py
```

## 关键思考

- 为什么 128K 上下文的单请求就能吃掉 40GB 显存？
- 为什么 GQA 对推理如此重要？
- 为什么 KV Cache 量化（FP8）能让 batch size 翻倍？
