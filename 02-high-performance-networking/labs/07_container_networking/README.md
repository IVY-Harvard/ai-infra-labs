# Lab 07: Container High-Performance Networking

## Overview
Kubernetes container networking for HPC/AI workloads requires bypassing the default CNI to achieve bare-metal performance. This lab covers three key technologies:

## 1. SR-IOV CNI
Single Root I/O Virtualization allows a single physical NIC to present multiple virtual functions (VFs) directly to pods, bypassing the kernel network stack.

- **Performance**: Near line-rate throughput, <2μs latency
- **Use case**: MPI workloads, distributed training, low-latency inference
- **Requirement**: SR-IOV capable NIC (Mellanox ConnectX-5/6/7, Intel E810)

## 2. Multus CNI
Meta-CNI plugin that enables attaching multiple network interfaces to pods. Essential for separating data-plane (RDMA/RoCE) from control-plane (Kubernetes default) traffic.

- Primary interface: Kubernetes pod network (Calico/Flannel)
- Secondary interface: SR-IOV VF for high-speed data transfer
- Optional: Additional interfaces for storage, management

## 3. RDMA Device Plugin
Exposes RDMA-capable devices (InfiniBand HCA, RoCE NIC) as Kubernetes extended resources, allowing pods to request RDMA access.

- Supports InfiniBand and RoCE v2
- Integrates with NVIDIA GPU Operator for GPUDirect RDMA
- Enables kernel-bypass networking inside containers

## Lab Files
| File | Description |
|------|-------------|
| `sriov_cni_setup.yaml` | SR-IOV NetworkAttachmentDefinition |
| `multus_config.yaml` | Multus multi-NIC pod configuration |
| `rdma_device_plugin.yaml` | RDMA Device Plugin DaemonSet |
| `network_test_pod.yaml` | Test pod for validating network setup |

## Prerequisites
```bash
# Install SR-IOV Network Operator
kubectl apply -f https://github.com/k8snetworkplumbingwg/sriov-network-operator/releases/latest/download/sriov-network-operator.yaml

# Verify SR-IOV capable NICs
kubectl get sriovnetworknodestates -n sriov-network-operator
```
