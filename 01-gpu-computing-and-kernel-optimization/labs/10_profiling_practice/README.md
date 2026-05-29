# Lab 10: GPU Profiling 全套实战

## 实验目的

掌握 GPU 性能分析的完整工具链：
1. **NSight Compute** (ncu): kernel 级别的微观分析
2. **NSight Systems** (nsys): 系统级别的宏观分析
3. **Roofline 分析**: 判断瓶颈类型
4. **NVML 实时监控**: 运行时状态监控

## 前置要求

- NVIDIA CUDA Toolkit（含 NSight 工具）
- `pynvml`: `pip install pynvml`
- PyTorch

## 文件说明

- `nsight_guide.md` — NSight 工具使用指南
- `roofline_analysis.py` — 自动化 Roofline 分析
- `nvml_monitor.py` — 实时 GPU 状态监控

## 实验步骤

1. 阅读 `nsight_guide.md`，了解 profiling 工具的用法
2. 运行 `roofline_analysis.py`，对常见算子做 Roofline 分析
3. 运行 `nvml_monitor.py`，在另一个终端运行 GPU 负载，观察监控输出
