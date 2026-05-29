# 03 - GPU 共享方案全景

## 引言：为什么需要 GPU 共享

一张 H20 有 96GB HBM3 显存。如果一个推理服务只用 10GB 显存，独占整张卡就浪费了 90%。
在 Slurm 中你可能用过 `--gres=gpu:1 --mem-per-gpu` 来限制，但那只是资源预留，不是真正
的隔离。K8s 生态提供了多种 GPU 共享方案，各有适用场景。

## 1. GPU 共享方案全景图

```
┌─────────────────────────────────────────────────────────────────────┐
│                     GPU 共享方案                                     │
│                                                                     │
│  硬件级隔离              软件级复用                  虚拟化            │
│  ┌──────────┐      ┌──────────────────┐      ┌──────────────┐     │
│  │   MIG    │      │  MPS    Time-    │      │    vGPU      │     │
│  │          │      │        Slicing   │      │              │     │
│  │ 硬件分区  │      │ CUDA   上下文     │      │ hypervisor   │     │
│  │ 独立引擎  │      │ 多进程  时分复用   │      │ 级隔离       │     │
│  └──────────┘      └──────────────────┘      └──────────────┘     │
│                                                                     │
│  隔离性：强 ◄─────────────────────────────────────────► 弱          │
│  灵活性：低 ◄─────────────────────────────────────────► 高          │
│  性能损耗：低 ◄────────────────────────────────────────► 中          │
└─────────────────────────────────────────────────────────────────────┘
```

## 2. MIG (Multi-Instance GPU)

### 2.1 硬件原理

MIG 是 A100/H100/H20 才有的硬件特性，将一张 GPU **物理分区**为多个独立实例。

```
一张 H20 (96GB, 132 SM):

全卡模式：
┌────────────────────────────────────────────────────────┐
│              96GB HBM3, 132 SM, 全部 NVLink            │
│              一个用户独占                                │
└────────────────────────────────────────────────────────┘

MIG 分区后（示例：3g.48gb x 2）：
┌───────────────────────────────┬───────────────────────────────┐
│  MIG Instance 0 (3g.48gb)    │  MIG Instance 1 (3g.48gb)    │
│  48GB HBM3, 66 SM            │  48GB HBM3, 66 SM            │
│  独立内存控制器               │  独立内存控制器               │
│  独立 L2 Cache               │  独立 L2 Cache               │
│  独立 SM                     │  独立 SM                     │
│  独立 Copy Engine            │  独立 Copy Engine            │
└───────────────────────────────┴───────────────────────────────┘
  两个实例完全隔离：内存、计算、带宽互不影响
```

### 2.2 MIG Profile 详解

H20 支持的 MIG Profile（与 A100 类似但资源量不同）：

```
Profile 命名规则：{GPU Instance size}g.{Memory size}gb

常见 Profile:
┌─────────────┬──────┬────────┬────────────┬──────────────────┐
│   Profile   │  SM  │ Memory │ Max Count  │    适用场景       │
├─────────────┼──────┼────────┼────────────┼──────────────────┤
│  1g.12gb    │  16  │  12GB  │    7       │ 小模型推理        │
│  2g.24gb    │  33  │  24GB  │    3       │ 中等模型推理      │
│  3g.48gb    │  66  │  48GB  │    2       │ 大模型推理/微调   │
│  7g.96gb    │ 132  │  96GB  │    1       │ 等于全卡          │
└─────────────┴──────┴────────┴────────────┴──────────────────┘

组合示例（一张 H20）：
  方案 A: 7 x 1g.12gb  → 7 个小推理服务
  方案 B: 1 x 3g.48gb + 2 x 2g.24gb  → 1 个大服务 + 2 个中服务
  方案 C: 2 x 3g.48gb  → 2 个中大服务（剩余资源无法使用）
```

### 2.3 MIG 配置操作

```bash
# 1. 检查 MIG 支持
nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv
# Disabled

# 2. 启用 MIG（需要 GPU Reset 或重启）
sudo nvidia-smi -i 0 -mig 1
# Warning: MIG mode change requires GPU Reset or node reboot

# 3. 重置 GPU（会中断所有进程！）
sudo nvidia-smi -i 0 -r

# 4. 创建 GPU Instance
# 创建 2 个 3g.48gb 的 GPU Instance
sudo nvidia-smi mig -i 0 -cgi 9,9   # 9 = 3g.48gb profile ID

# 5. 在每个 GPU Instance 中创建 Compute Instance
sudo nvidia-smi mig -i 0 -cci       # 为所有 GI 创建默认 CI

# 6. 查看结果
nvidia-smi mig -i 0 -lgi
# +----+----------+-----+----------+--------+--------+
# | ID | Inst Num | Mem | SM Count | CE Num | Dec Num|
# +----+----------+-----+----------+--------+--------+
# |  0 |        0 | 48G |       66 |      3 |      3 |
# |  1 |        1 | 48G |       66 |      3 |      3 |
# +----+----------+-----+----------+--------+--------+
```

### 2.4 MIG 在 K8s 中的暴露

```yaml
# GPU Operator 自动将 MIG 实例暴露为不同的资源类型

# MIG Strategy: "single" (所有 MIG 实例同一类型时)
# 节点资源：
# nvidia.com/gpu: 7   (7 个 1g.12gb MIG 实例)

# MIG Strategy: "mixed" (MIG 实例类型不同时)
# 节点资源：
# nvidia.com/mig-1g.12gb: 2
# nvidia.com/mig-3g.48gb: 1

# Pod 请求 MIG 实例
apiVersion: v1
kind: Pod
metadata:
  name: inference-small
spec:
  containers:
  - name: model
    resources:
      limits:
        nvidia.com/mig-1g.12gb: 1  # 请求一个 1g.12gb MIG 实例
```

### 2.5 使用 GPU Operator 管理 MIG

```yaml
# ConfigMap 定义 MIG 配置
apiVersion: v1
kind: ConfigMap
metadata:
  name: mig-config
  namespace: gpu-operator
data:
  config.yaml: |
    version: v1
    mig-configs:
      all-1g.12gb:
        - device-filter: ["0x2339"]  # H20 PCI ID
          devices: all
          mig-enabled: true
          mig-devices:
            "1g.12gb": 7
      all-3g.48gb:
        - device-filter: ["0x2339"]
          devices: all
          mig-enabled: true
          mig-devices:
            "3g.48gb": 2
      mixed:
        - device-filter: ["0x2339"]
          devices: all
          mig-enabled: true
          mig-devices:
            "1g.12gb": 2
            "3g.48gb": 1
---
# 通过节点标签切换 MIG 配置
# kubectl label node gpu-node-1 nvidia.com/mig.config=all-1g.12gb
# MIG Manager 会自动执行分区操作
```

## 3. MPS (Multi-Process Service)

### 3.1 原理

MPS 是 CUDA 多进程服务，允许多个 CUDA 进程 **同时在一张 GPU 上执行**。

```
无 MPS：
┌──────────────────────────────┐
│            GPU               │
│  ┌─────┐    上下文切换       │
│  │Proc1│◄──────────►        │
│  └─────┘  ┌─────┐          │
│           │Proc2│          │
│           └─────┘          │
│  同一时刻只有一个进程执行     │
│  频繁上下文切换（昂贵！）    │
└──────────────────────────────┘

有 MPS：
┌──────────────────────────────┐
│            GPU               │
│  ┌─────┐ ┌─────┐           │
│  │Proc1│ │Proc2│           │
│  └─────┘ └─────┘           │
│  共享 CUDA Context           │
│  两个进程的 kernel 可以并发   │
│  减少上下文切换开销           │
└──────────────────────────────┘
```

### 3.2 MPS 配置

```bash
# 启动 MPS 控制守护进程
export CUDA_VISIBLE_DEVICES=0
nvidia-cuda-mps-control -d

# 设置活跃线程百分比限制（资源隔离）
echo "set_default_active_thread_percentage 50" | nvidia-cuda-mps-control
# 每个客户端最多使用 50% 的 SM

# 为特定 PID 设置限制
echo "set_active_thread_percentage <PID> 25" | nvidia-cuda-mps-control

# 关闭 MPS
echo quit | nvidia-cuda-mps-control
```

### 3.3 MPS 在 K8s 中的部署

```yaml
# NVIDIA Device Plugin 支持 MPS
apiVersion: v1
kind: ConfigMap
metadata:
  name: device-plugin-config
data:
  config.yaml: |
    version: v1
    sharing:
      mps:
        renameByDefault: false
        resources:
        - name: nvidia.com/gpu
          replicas: 4  # 每张 GPU 虚拟为 4 个 "设备"
          # 注意：replicas 只影响调度计数，不做真正的资源隔离
          # 需要配合 MPS 的 active_thread_percentage 做限制
```

### 3.4 MPS 的优缺点

```
优点：
  ✅ 多进程真正并发执行 GPU kernel
  ✅ 减少上下文切换开销
  ✅ 对应用透明，不需要修改代码
  ✅ 支持所有 NVIDIA GPU（不限于 A100/H20）

缺点：
  ❌ 无内存隔离！一个进程的 bug 可能 crash 整个 GPU
  ❌ 无显存限制（进程可以分配超出份额的显存）
  ❌ 错误传播：一个进程的 CUDA error 影响所有客户端
  ❌ 需要所有进程使用相同 CUDA Compute Capability
  ❌ 性能不可预测（取决于 kernel 的重叠程度）
```

## 4. Time-Slicing（时分复用）

### 4.1 原理

Time-Slicing 是最简单的 GPU 共享方式：多个容器 **轮流** 使用同一张 GPU。

```
Time-Slicing：
┌──────────────────────────────────────────────┐
│                    GPU                        │
│  ┌─────┐         ┌─────┐         ┌─────┐   │
│  │Pod1 │  idle   │Pod2 │  idle   │Pod1 │   │
│  │     │         │     │         │     │   │
│  └─────┘         └─────┘         └─────┘   │
│  t=0    t=1      t=2    t=3      t=4       │
│                                              │
│  类似 CPU 时间片调度，由 GPU 驱动管理          │
│  上下文切换开销：~1ms                         │
└──────────────────────────────────────────────┘
```

### 4.2 Time-Slicing 配置

```yaml
# Device Plugin ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: device-plugin-config
  namespace: gpu-operator
data:
  h20-shared: |
    version: v1
    sharing:
      timeSlicing:
        renameByDefault: false
        failRequestsGreaterThanOne: false
        resources:
        - name: nvidia.com/gpu
          replicas: 4  # 每张物理 GPU 允许 4 个 Pod 共享
---
# 通过节点标签应用配置
# kubectl label node gpu-node-1 nvidia.com/device-plugin.config=h20-shared
```

```yaml
# Pod 正常请求 GPU，调度器认为节点有 8*4=32 个 "GPU"
apiVersion: v1
kind: Pod
metadata:
  name: inference-1
spec:
  containers:
  - name: model
    image: my-inference:v1
    resources:
      limits:
        nvidia.com/gpu: 1  # 实际共享 1/4 的 GPU 时间
```

### 4.3 Time-Slicing 的关键限制

```
⚠️  Time-Slicing 不提供任何隔离！

1. 无显存隔离
   Pod A 可以分配 90GB 显存（H20 总共 96GB）
   Pod B 尝试分配时 OOM
   
2. 无计算隔离
   Pod A 运行重 kernel 时，Pod B 的延迟会飙升
   
3. 无故障隔离
   Pod A 的 CUDA 错误可能导致 GPU Reset
   影响所有共享该 GPU 的 Pod

4. 无法感知过载
   调度器以为有 32 个 GPU，实际只有 8 张卡
   可能导致严重 overcommit

适用场景：
  ✅ 开发测试环境（不在意性能隔离）
  ✅ 轻量推理（显存用量小、计算量小）
  ❌ 生产推理（延迟不可预测）
  ❌ 训练（性能严重下降）
```

## 5. vGPU（NVIDIA Virtual GPU）

### 5.1 架构

```
┌──────────────────────────────────────────┐
│           Hypervisor (ESXi/KVM)          │
│  ┌──────────┐  ┌──────────┐            │
│  │   VM 1   │  │   VM 2   │            │
│  │ ┌──────┐ │  │ ┌──────┐ │            │
│  │ │vGPU A│ │  │ │vGPU B│ │            │
│  │ │32GB  │ │  │ │64GB  │ │            │
│  │ └──────┘ │  │ └──────┘ │            │
│  └──────────┘  └──────────┘            │
│       │              │                  │
│       ▼              ▼                  │
│  ┌─────────────────────────────────┐   │
│  │  NVIDIA vGPU Manager            │   │
│  │  - 显存隔离（硬件强制）          │   │
│  │  - 计算隔离（时间片/MIG）       │   │
│  │  - QoS 控制                     │   │
│  └─────────────────────────────────┘   │
│       │                                 │
│  ┌────▼────────────────────────────┐   │
│  │     Physical GPU (H20 96GB)     │   │
│  └─────────────────────────────────┘   │
└──────────────────────────────────────────┘
```

### 5.2 vGPU 类型

```
C-Series (Compute): 适合 AI/ML
  H20-2C:  2GB  显存
  H20-4C:  4GB  显存
  H20-12C: 12GB 显存
  H20-24C: 24GB 显存
  H20-48C: 48GB 显存
  H20-96C: 96GB 显存（全卡）

Q-Series (Quadro): 适合图形渲染
  （H20 作为数据中心 GPU，通常不用 Q 系列）
```

### 5.3 vGPU 的优缺点

```
优点：
  ✅ 显存硬隔离（物理分区或严格限制）
  ✅ 支持热迁移（VM 级别）
  ✅ 安全性高（hypervisor 级隔离）
  ✅ 可与 VMware/OpenStack 集成

缺点：
  ❌ 需要 vGPU 许可证（昂贵！）
  ❌ 需要 hypervisor（裸金属不适用）
  ❌ 性能开销：3-8%（虚拟化开销）
  ❌ 不支持 P2P（GPU Direct RDMA/NVLink 不可用）
  ❌ 训练性能严重受限（无 NVLink/NVSwitch）

结论：在 AI 训练场景（你的 8xH20）中，vGPU 通常不适用。
      更适合多租户推理或企业桌面虚拟化。
```

## 6. 方案对比决策矩阵

### 6.1 核心维度对比

| 维度 | MIG | MPS | Time-Slicing | vGPU |
|------|-----|-----|-------------|------|
| **计算隔离** | 硬件级 ✅ | 软件级 (线程限制) | 无 ❌ | hypervisor 级 ✅ |
| **显存隔离** | 硬件级 ✅ | 无 ❌ | 无 ❌ | 硬件级 ✅ |
| **错误隔离** | 完全隔离 | 无（共享 context）| 无 | 完全隔离 |
| **性能开销** | <5% | <5% | 10-30% | 3-8% |
| **最大实例数** | 7 (1g) | 48 (per GPU) | 无限制 | 取决于 profile |
| **显存灵活性** | 固定 profile | 动态分配 | 动态分配 | 固定 profile |
| **GPU 要求** | A100/H100/H20 | 全系列 | 全系列 | 全系列(需许可) |
| **NVLink 支持** | 实例间不支持 | 支持 | 支持 | 不支持 |
| **部署复杂度** | 中（需规划 profile）| 低 | 低 | 高 |
| **K8s 集成** | GPU Operator | Device Plugin | Device Plugin | 需额外插件 |
| **动态调整** | 需 drain 节点 | 随时 | 随时 | 需重新配置 |
| **许可费用** | 免费 | 免费 | 免费 | 付费 |

### 6.2 场景推荐

```
场景：大规模训练 (8 GPU 全量使用)
  → 不需要共享，使用全卡模式

场景：多个推理服务，需要稳定延迟
  → MIG（如果是 A100/H100/H20）
  → 原因：硬件隔离保证延迟稳定

场景：多个推理服务，模型较小，追求吞吐
  → MPS
  → 原因：kernel 并发执行，吞吐高于 Time-Slicing

场景：开发测试，多人共享 GPU
  → Time-Slicing
  → 原因：配置最简单，开发环境不需要隔离

场景：多租户，安全要求高
  → vGPU 或 MIG
  → 原因：硬件级隔离
  → 如果是物理机 → MIG
  → 如果是虚拟化环境 → vGPU

场景：你的 8xH20 集群
  训练节点 (4-6 张卡): 不共享，全卡分配
  推理节点 (2-4 张卡): MIG，按推理模型大小规划 profile
  开发节点 (如有): Time-Slicing，4-8 倍 overcommit
```

### 6.3 混合策略

```yaml
# 在同一集群中，不同节点使用不同的共享策略

# 节点标签
# gpu-node-1,2,3,4: role=training (无共享)
# gpu-node-5,6:     role=inference-mig (MIG 分区)
# gpu-node-7,8:     role=dev-shared (Time-Slicing)

# 训练 Pod
spec:
  nodeSelector:
    role: training
  containers:
  - resources:
      limits:
        nvidia.com/gpu: 8  # 全卡

# 推理 Pod
spec:
  nodeSelector:
    role: inference-mig
  containers:
  - resources:
      limits:
        nvidia.com/mig-3g.48gb: 1  # MIG 实例

# 开发 Pod
spec:
  nodeSelector:
    role: dev-shared
  containers:
  - resources:
      limits:
        nvidia.com/gpu: 1  # Time-Slicing 共享
```

## 7. HAMi — 第三方 GPU 共享方案

除了 NVIDIA 官方方案，社区还有 HAMi (Heterogeneous AI Computing Virtualization Middleware)：

```
HAMi 的核心能力：
  - GPU 显存限制（通过 LD_PRELOAD hook CUDA API）
  - GPU 算力限制（通过限制 SM 使用百分比）
  - 支持多种 GPU（NVIDIA/AMD/华为昇腾）

示例：
apiVersion: v1
kind: Pod
metadata:
  name: gpu-limited
spec:
  containers:
  - name: app
    resources:
      limits:
        nvidia.com/gpu: 1
        nvidia.com/gpumem: 4096  # 限制 4GB 显存
        nvidia.com/gpucores: 25  # 限制 25% 算力

优点：细粒度资源限制，不需要 MIG 硬件
缺点：通过 hook 实现，有绕过风险；性能有额外开销
```

## 小结

```
GPU 共享不是银弹，关键是理解你的工作负载特征：

训练工作负载：
  - 需要完整 GPU + NVLink，不适合共享
  - 追求最大计算性能和通信带宽

推理工作负载：
  - 通常不需要整张 GPU
  - 关注延迟稳定性和资源利用率
  - MIG 是首选（有硬件支持的情况下）

开发环境：
  - 不需要性能保障
  - Time-Slicing 最简单

你的 8xH20 集群建议：
  1. 训练：不共享，按需分配 1/2/4/8 张全卡
  2. 推理：MIG 分区，根据模型大小选择 profile
  3. 评估 MPS 用于推理场景（如果对延迟抖动容忍度高）
  4. 保留 1-2 张卡做 Time-Slicing 给开发者用
```
