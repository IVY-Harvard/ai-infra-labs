# Lab 08: 算子融合对比

## 实验目的

直观展示算子融合的效果：
1. 对比 unfused（多个独立 kernel）vs fused（一个合并 kernel）的性能
2. 手写一个简单的融合 kernel
3. 量化融合带来的 HBM 访问节省

## 前置要求

- PyTorch + Triton
- 已读 theory/06（算子融合原理）

## 运行

```bash
python unfused_vs_fused.py
python custom_fused_kernel.py
```

## 预期观察

- 简单 pointwise 融合: 2-5x 加速
- LayerNorm 融合: 1.5-3x 加速
- 加速倍数和数据量正相关（数据越大，HBM 访问越是瓶颈）
