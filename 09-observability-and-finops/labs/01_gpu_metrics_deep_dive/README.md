# Lab 01 - GPU 指标深潜：超越 nvidia-smi 的真实利用率

## 目标

通过本实验，读者将：
1. 理解 nvidia-smi utilization 的误导性，学会用 DCGM Profiling Metrics 衡量真实 GPU 利用率
2. 实现基于 SM Occupancy、Tensor Core Active、DRAM Active 的多维度利用率计算
3. 构建一个生产级 GPU 健康度评分系统

## 前置条件

- 已安装 NVIDIA DCGM（datacenter-gpu-manager）
- Prometheus + Grafana 已部署
- 至少有 1 张 NVIDIA GPU（H20 / A100 / A800 等）
- Python 3.10+

## 实验内容

### Part 1: DCGM 指标采集基础

参考 `dcgm_metrics_guide.md` 了解所有关键指标的含义与采集方法。

### Part 2: 真实利用率计算

运行 `gpu_real_utilization.py`，该脚本：
- 从 Prometheus 拉取 DCGM Profiling Metrics
- 计算 Prefill / Decode 阶段的真实效率
- 生成综合利用率评分（区别于 nvidia-smi 的粗糙数字）

```bash
python gpu_real_utilization.py --prometheus-url http://localhost:9090 --interval 30
```

### Part 3: SM Occupancy 持续监控

运行 `sm_occupancy_monitor.py`，该脚本：
- 实时监控 SM Occupancy 趋势
- 当 Occupancy 持续低于阈值时触发告警
- 生成 Occupancy 与吞吐量的关联分析报告

```bash
python sm_occupancy_monitor.py --prometheus-url http://localhost:9090 --alert-threshold 0.3
```

## 预期产出

- 一份基于真实 DCGM 指标的 GPU 利用率报告（JSON 格式）
- SM Occupancy 趋势图与告警规则
- GPU 健康度评分仪表盘数据

## 关键收获

| 维度 | nvidia-smi 显示的 | DCGM 真正反映的 |
|------|---------------------|-------------------|
| 计算效率 | GPU-Util % | Tensor Core Active + SM Active |
| 内存效率 | Memory Used/Total | DRAM Active (带宽利用率) |
| 并行效率 | 无 | SM Occupancy |
| 通信效率 | 无 | NVLink TX/RX Bytes |
