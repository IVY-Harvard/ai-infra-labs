# NSight 工具使用指南

## 两个核心工具

| 工具 | 视角 | 适用场景 |
|------|------|----------|
| NSight Systems (nsys) | 宏观/系统级 | 找到哪个 kernel 是瓶颈、timeline 分析 |
| NSight Compute (ncu) | 微观/kernel级 | 分析单个 kernel 的性能瓶颈 |

## NSight Systems (nsys)

### 基本用法

```bash
# 采集整个程序的 timeline
nsys profile -o report python your_script.py

# 指定采集范围
nsys profile --trace=cuda,nvtx,osrt -o report ./your_app

# 只采集特定时间段
nsys profile --delay 5 --duration 10 -o report python your_script.py
```

### 在 PyTorch 中标记感兴趣的区域

```python
import torch
import nvtx  # pip install nvtx

# 方法 1: NVTX range
with nvtx.annotate("forward_pass", color="blue"):
    output = model(input)

# 方法 2: PyTorch profiler (会自动加 NVTX 标记)
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    with_stack=True,
) as prof:
    output = model(input)
prof.export_chrome_trace("trace.json")
```

### 常见分析模式

```
1. Timeline 视图中看到大量空白 → CPU 是瓶颈（数据加载/预处理）
2. Kernel 之间有间隙 → Kernel launch overhead 或 同步等待
3. 某个 kernel 占比极高 → 重点优化这个 kernel
4. H2D/D2H 传输占时间多 → 需要异步传输或减少传输
```

## NSight Compute (ncu)

### 基本用法

```bash
# 分析所有 kernel（会很慢，因为每个 kernel replay 多次）
ncu python your_script.py

# 只分析特定 kernel
ncu --kernel-name "regex:matmul.*" python your_script.py

# 指定采集的 metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
python your_script.py

# 完整分析（所有 metrics，很慢但信息最全）
ncu --set full -o report python your_script.py

# 只分析第 N 次 kernel 调用
ncu --launch-skip 10 --launch-count 5 python your_script.py
```

### 关键 Metrics

```
性能指标:
├── sm__throughput.avg.pct_of_peak_sustained_elapsed
│   └── SM 计算吞吐利用率（高 → compute-bound）
├── dram__throughput.avg.pct_of_peak_sustained_elapsed
│   └── HBM 带宽利用率（高 → memory-bound）
├── sm__warps_active.avg.pct_of_peak_sustained_elapsed
│   └── Warp 占用率 (occupancy)
└── sm__pipe_tensor__cycles_active.avg.pct_of_peak_sustained_elapsed
    └── Tensor Core 利用率

内存指标:
├── dram__bytes_read.sum / dram__bytes_write.sum
│   └── 实际 HBM 读/写字节数
├── l2__read_throughput.avg.pct_of_peak_sustained_elapsed
│   └── L2 Cache 带宽利用率
└── smsp__shared_mem_bank_conflicts
    └── 共享内存 bank conflict 次数

效率指标:
├── smsp__thread_inst_executed_per_inst_executed.ratio
│   └── 每条指令的有效线程比（divergence 影响）
└── smsp__warps_issue_stalled_*
    └── Warp stall 原因分析
```

### 判断瓶颈类型

```
1. 查看 sm__throughput 和 dram__throughput:
   - sm 高, dram 低 → Compute-bound
   - sm 低, dram 高 → Memory-bound
   - 都低 → Latency-bound（occupancy 不够或其他问题）

2. Latency-bound 的常见原因:
   - Occupancy 太低（寄存器/共享内存用太多）
   - Warp divergence
   - 依赖链太长
   - Atomic 操作竞争
```

### 实用的 ncu 命令

```bash
# 快速判断 compute vs memory bound
ncu --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed \
  python script.py

# 检查 Tensor Core 利用率
ncu --metrics \
  sm__pipe_tensor__cycles_active.avg.pct_of_peak_sustained_elapsed \
  python script.py

# 检查 occupancy
ncu --metrics \
  sm__warps_active.avg.pct_of_peak_sustained_elapsed,\
  sm__maximum_warps_per_active_cycle \
  python script.py

# 检查内存效率
ncu --metrics \
  dram__bytes_read.sum,\
  dram__bytes_write.sum,\
  l2__read_throughput.avg.pct_of_peak_sustained_elapsed \
  python script.py
```

## PyTorch Profiler

### 基本用法

```python
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step in range(6):
        output = model(input)
        loss.backward()
        optimizer.step()
        prof.step()

# 打印摘要
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

### 输出解读

```
Name                    Self CPU    CPU total   Self CUDA   CUDA total
---------------------------------------------------------------------
aten::mm                  0.5ms       0.5ms       5.2ms       5.2ms
aten::addmm               0.3ms       0.8ms       3.1ms       3.1ms
aten::layer_norm          0.2ms       0.4ms       1.5ms       1.5ms

关注:
- CUDA total 最大的操作 → 优化目标
- Self CPU vs CPU total 的差距 → 函数调用开销
- 算子名称帮你定位到代码位置
```

## 实战 Workflow

```
Step 1: nsys 全局分析
└── 找到最耗时的 kernel 或阶段

Step 2: ncu 聚焦分析
└── 对目标 kernel 做详细 profiling

Step 3: 判断瓶颈
├── Compute-bound → Tensor Core? 更低精度?
├── Memory-bound → 融合? 减少访存?
└── Latency-bound → 提高 occupancy? 减少 divergence?

Step 4: 优化
└── 应用对应策略

Step 5: 验证
└── 再次 profile，确认改进
```

## 常见问题

### Q: profile 运行太慢怎么办？
```
- 用 --launch-skip N --launch-count M 只分析特定 kernel
- 用 --set basic 而非 --set full
- 减少输入数据量（但保持足够大以代表实际负载）
```

### Q: 看不到 kernel name？
```
- 确保没有 strip debug info
- 用 NVTX 标记代码区域
- PyTorch: 设置 TORCH_SHOW_DISPATCH_TRACE=1
```

### Q: 数据量太大？
```
- nsys: 用 --duration 限制采集时间
- ncu: 用 --launch-count 限制分析的 kernel 数
- 导出为 .qdrep / .ncu-rep 文件离线分析
```
