# Lab 02 - K8s Device Plugin 机制

## 目标

理解 K8s Device Plugin 框架如何让 kubelet 感知和分配 GPU 设备。
通过实现一个 Fake GPU Device Plugin，掌握完整的设备注册、发现、分配流程。

## 背景知识

```
K8s Device Plugin 架构：

kubelet                          Device Plugin (如 nvidia-device-plugin)
  │                                          │
  │    1. 注册 (/var/lib/kubelet/device-plugins/kubelet.sock)
  │◄─────── Register(ResourceName, Endpoint, Options) ──────│
  │                                          │
  │    2. 发现设备                             │
  │─────── ListAndWatch() ──────────────────►│
  │◄─────── DeviceList [gpu-0, gpu-1, ...] ──│
  │                                          │
  │    3. Pod 请求 GPU                        │
  │─────── Allocate(deviceIDs) ─────────────►│
  │◄─────── AllocateResponse(envs, mounts) ──│
  │                                          │
  │    4. 设备容器级别挂载                     │
  │    (设置 NVIDIA_VISIBLE_DEVICES 等)       │

关键接口 (gRPC)：
  - Register: 向 kubelet 注册资源类型
  - ListAndWatch: 上报可用设备列表，持续推送变更
  - Allocate: kubelet 分配设备时回调，返回挂载/环境变量
  - GetPreferredAllocation: (可选) 返回推荐的设备组合
```

## 前置条件

- Go 1.21+
- K8s 集群（kind 即可，不需要真 GPU）
- kubectl

## 实验内容

### 实验 1：分析 NVIDIA Device Plugin 的行为

```bash
# 1. 查看节点上报的 GPU 资源
kubectl describe node <gpu-node> | grep -A 5 "Capacity"
# 应该看到：nvidia.com/gpu: 8

# 2. 查看 device plugin 的 socket 文件
ls /var/lib/kubelet/device-plugins/
# nvidia.sock  kubelet.sock

# 3. 查看 device plugin Pod 的日志
kubectl logs -n kube-system -l app=nvidia-device-plugin-daemonset

# 4. 查看 kubelet 的设备分配状态
kubectl get --raw /api/v1/nodes/<node>/proxy/pods | \
  jq '.items[].spec.containers[].resources'
```

### 实验 2：部署 Fake GPU Device Plugin

详见：
- [fake_gpu_plugin.go](./fake_gpu_plugin.go) — 完整的 Fake GPU Plugin 实现
- [daemonset.yaml](./daemonset.yaml) — DaemonSet 部署清单
- [test_pod.yaml](./test_pod.yaml) — 测试用 Pod

```bash
# 构建 Fake GPU Plugin
cd fake-gpu-plugin
go build -o fake-gpu-plugin .

# 构建容器镜像
docker build -t fake-gpu-plugin:v1 .

# 部署到集群
kubectl apply -f daemonset.yaml

# 验证 — 节点应该多出 fake.com/gpu 资源
kubectl describe node | grep "fake.com/gpu"

# 创建测试 Pod
kubectl apply -f test_pod.yaml
kubectl describe pod fake-gpu-test
```

### 实验 3：观察设备分配流程

```bash
# 启用 kubelet verbose 日志
# 在 kubelet 启动参数中加 --v=6

# 提交测试 Pod 后观察 kubelet 日志
journalctl -u kubelet | grep -i "device"
# 应该看到：
#   Allocate request: [fake-gpu-0 fake-gpu-1]
#   Allocate response: envs={FAKE_GPU_IDS: "0,1"}, mounts=[/dev/fake-gpu]
```

## 思考题

1. 为什么 NVIDIA Device Plugin 需要 privileged 权限？
2. 如果一个 GPU 发生 ECC 错误，Device Plugin 如何通知 kubelet？
3. `GetPreferredAllocation` 在 GPU 场景中有什么用途？
