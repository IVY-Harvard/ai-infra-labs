# GPU Kernel Benchmark Suite

## 项目概述

一个企业级 GPU 基准测试工具套件，用于：
- 测试 GPU 计算性能（FP16/FP32/INT8 TFLOPS）
- 测试 HBM 显存带宽
- 测试 GPU 间 P2P 通信带宽
- 测试 NVLink 带宽
- 生成 HTML 对比报告
- 支持多卡并行测试

## 为什么需要这个工具？

当拥有多卡 GPU 时，需要回答：
1. 每张卡的实际算力是多少？（是否达标？有没有故障卡？）
2. 卡间通信带宽是多少？（NVLink 是否正常工作？）
3. 显存带宽是否正常？（是否有硬件降级？）
4. 不同量化精度下的性能差异？

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
# 运行完整基准测试
python -m src.cli benchmark --all

# 只测计算性能
python -m src.cli benchmark --compute

# 只测内存带宽
python -m src.cli benchmark --memory

# 测试 P2P/NVLink
python -m src.cli benchmark --p2p

# 生成报告
python -m src.cli report --output results/report.html

# 对比两次测试结果
python -m src.cli compare results/run1.json results/run2.json
```

## Docker

```bash
docker build -t gpu-benchmark .
docker run --gpus all gpu-benchmark benchmark --all
```

## 项目结构

```
gpu-kernel-benchmark-suite/
├── src/
│   ├── cli.py                 # CLI 入口
│   ├── kernels/
│   │   ├── compute_benchmark.py    # 算力测试
│   │   ├── memory_benchmark.py     # 显存带宽测试
│   │   ├── p2p_benchmark.py        # GPU 间 P2P
│   │   └── nvlink_benchmark.py     # NVLink 带宽
│   ├── profiler/
│   │   └── gpu_profiler.py         # NVML 封装
│   └── reporter/
│       ├── html_report.py          # HTML 报告
│       └── comparison.py           # 多卡对比
├── tests/
│   └── test_benchmarks.py
├── configs/
│   └── default.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

## 输出示例

```
GPU Benchmark Results
=====================
GPU 0: NVIDIA H20 (96GB HBM3)
  FP32 TFLOPS:     42.1 / 44.0 (95.7%)
  FP16 TFLOPS:    139.2 / 148.0 (94.1%)
  INT8 TOPS:      271.5 / 296.0 (91.7%)
  HBM Bandwidth:  3812 / 4000 GB/s (95.3%)
  
P2P Bandwidth Matrix (GB/s):
       GPU0  GPU1  GPU2  GPU3
GPU0:    -   412   410   415
GPU1:  411     -   413   412
GPU2:  409   413     -   414
GPU3:  414   411   413     -
```
