# Lab 02：JuiceFS 部署与调优实战

## 实验目标

在多卡 GPU 环境中完整部署 JuiceFS，并针对 AI 工作负载进行缓存调优，实现模型加载加速和 Checkpoint 写入优化。

## 前置条件

- Docker 已安装
- 至少 500GB NVMe SSD 可用于缓存
- 网络连接正常

## 实验内容

### 实验 1：JuiceFS + Redis + MinIO 完整部署

按照 `setup_guide.md` 完成部署。

### 实验 2：Kubernetes CSI 集成

使用 `k8s_csi_setup.yaml` 配置 JuiceFS CSI Driver（如有 K8s 环境）。

### 实验 3：缓存调优与性能测试

运行 `cache_tuning_benchmark.py` 对比不同缓存参数下的性能。

## 运行方式

```bash
# 1. 部署基础设施（参考 setup_guide.md）
# 2. 运行缓存调优测试
pip install numpy
python cache_tuning_benchmark.py --jfs-mount /mnt/jfs --cache-dir /nvme/jfs-cache
```

## 调优维度

| 参数 | 测试值 | 影响 |
|------|--------|------|
| cache-size | 100G/300G/500G | 缓存命中率 |
| prefetch | 1/3/5 | 顺序读性能 |
| buffer-size | 1024/2048/4096 | 写性能 |
| max-uploads | 10/30/50 | 后台上传并发 |

## 文件列表

- `setup_guide.md` — 完整部署步骤
- `k8s_csi_setup.yaml` — Kubernetes CSI 配置
- `cache_tuning_benchmark.py` — 缓存参数调优测试
