# Lab 01：分布式文件系统对比测试

## 实验目标

在多卡 GPU 环境中，对比不同存储方案在 AI 典型工作负载下的实际性能表现，建立量化的性能基准。

## 前置条件

- 多卡 GPU 单节点
- 本地 NVMe SSD（至少 500GB 可用）
- NFS 挂载点可用
- MinIO 已部署或可部署
- Python 3.10+，PyTorch 2.3+

## 实验内容

### 实验 1：顺序读性能（模型加载场景）

测试不同存储在大文件顺序读下的吞吐表现。

### 实验 2：随机读 IOPS（数据加载场景）

测试小文件随机访问的 IOPS 和延迟。

### 实验 3：突发写性能（Checkpoint 场景）

测试大块数据突发写入的吞吐和延迟。

### 实验 4：多进程并发读

模拟 8 GPU DataLoader Worker 并发读取的场景。

## 运行方式

```bash
# 安装依赖
pip install numpy torch boto3

# 运行全部测试
python storage_benchmark.py --all

# 运行单项测试
python storage_benchmark.py --test sequential_read --path /mnt/nfs/test
python storage_benchmark.py --test random_read --path /local/nvme/test
python storage_benchmark.py --test burst_write --path /mnt/jfs/test
```

## 预期结论

| 存储方案 | 顺序读 | 随机 IOPS | 突发写 | 适合场景 |
|---------|--------|-----------|--------|---------|
| 本地 NVMe | 最快 | 最快 | 最快 | 所有（但容量有限）|
| NFS | 慢 | 差 | 差 | 小规模共享 |
| JuiceFS+缓存 | 接近本地 | 良好 | 良好 | 通用推荐 |
| MinIO 直连 | 中等 | 差 | 中等 | 对象存储场景 |

## 文件列表

- `storage_benchmark.py` — 基准测试工具
- `comparison_report.md` — 结果分析报告模板
