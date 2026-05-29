/*
 * GPU 拓扑感知调度插件 Demo
 *
 * 功能：在 Score 阶段对节点打分时，优先选择 GPU 拓扑连接更好的节点。
 * 例如：如果 Pod 请求 4 个 GPU，优先选择有 4 个通过 NVLink 互连的空闲 GPU 的节点。
 *
 * 编译说明：
 *   go mod init gpu-topo-scheduler
 *   go mod tidy
 *   go build -o gpu-topo-scheduler .
 */

package main

import (
	"context"
	"fmt"
	"math"
	"os"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/cmd/kube-scheduler/app"
)

const (
	// PluginName 是插件名称
	PluginName = "GPUTopologyScore"

	// GPU 拓扑类型的分数权重
	NVLinkScore  = 100  // NVLink 连接的 GPU 组合最优
	PCIeScore    = 60   // 同一 PCIe Switch 下的 GPU
	CrossSocket  = 30   // 跨 CPU Socket 的 GPU
)

// GPUTopologyScore 实现 framework.ScorePlugin 接口
type GPUTopologyScore struct {
	handle framework.Handle
}

// GPUTopology 描述节点的 GPU 拓扑信息
type GPUTopology struct {
	TotalGPUs     int
	AllocatedGPUs int
	NVLinkGroups  [][]int  // NVLink 互连的 GPU 组
	PCIeGroups    [][]int  // 同一 PCIe Switch 下的 GPU 组
}

var _ framework.ScorePlugin = &GPUTopologyScore{}

// Name 返回插件名称
func (pl *GPUTopologyScore) Name() string {
	return PluginName
}

// Score 对单个节点打分
func (pl *GPUTopologyScore) Score(
	ctx context.Context,
	state *framework.CycleState,
	pod *v1.Pod,
	nodeName string,
) (int64, *framework.Status) {

	// 1. 获取 Pod 请求的 GPU 数量
	requestedGPUs := getRequestedGPUs(pod)
	if requestedGPUs == 0 {
		// 非 GPU Pod，不影响调度
		return framework.MaxNodeScore, nil
	}

	// 2. 获取节点信息
	nodeInfo, err := pl.handle.SnapshotSharedLister().NodeInfos().Get(nodeName)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("获取节点信息失败: %v", err))
	}

	// 3. 解析节点 GPU 拓扑（从节点 annotation 中读取）
	topology := parseGPUTopology(nodeInfo.Node())
	if topology == nil {
		// 节点没有 GPU 拓扑信息，给基础分
		return 50, nil
	}

	// 4. 计算拓扑感知分数
	score := calculateTopologyScore(topology, requestedGPUs)

	return score, nil
}

// ScoreExtensions 返回 nil，表示不需要 Normalize
func (pl *GPUTopologyScore) ScoreExtensions() framework.ScoreExtensions {
	return pl
}

// NormalizeScore 将分数标准化到 [0, MaxNodeScore] 范围
func (pl *GPUTopologyScore) NormalizeScore(
	ctx context.Context,
	state *framework.CycleState,
	pod *v1.Pod,
	scores framework.NodeScoreList,
) *framework.Status {
	var maxScore int64 = 0
	for _, s := range scores {
		if s.Score > maxScore {
			maxScore = s.Score
		}
	}
	if maxScore == 0 {
		return nil
	}
	for i := range scores {
		scores[i].Score = scores[i].Score * framework.MaxNodeScore / maxScore
	}
	return nil
}

// getRequestedGPUs 从 Pod Spec 中获取请求的 GPU 数量
func getRequestedGPUs(pod *v1.Pod) int64 {
	var totalGPUs int64
	for _, container := range pod.Spec.Containers {
		if gpuQty, ok := container.Resources.Limits["nvidia.com/gpu"]; ok {
			totalGPUs += gpuQty.Value()
		}
	}
	for _, initContainer := range pod.Spec.InitContainers {
		if gpuQty, ok := initContainer.Resources.Limits["nvidia.com/gpu"]; ok {
			if gpuQty.Value() > totalGPUs {
				totalGPUs = gpuQty.Value()
			}
		}
	}
	return totalGPUs
}

// parseGPUTopology 从节点 Annotations 中解析 GPU 拓扑
// 预期 annotation 格式：
//   gpu.topology/nvlink-groups: "[[0,1,2,3],[4,5,6,7]]"
//   gpu.topology/pcie-groups: "[[0,1],[2,3],[4,5],[6,7]]"
//   gpu.topology/total-gpus: "8"
//   gpu.topology/allocated-gpus: "3"
func parseGPUTopology(node *v1.Node) *GPUTopology {
	annotations := node.Annotations
	if annotations == nil {
		return nil
	}

	totalStr, ok := annotations["gpu.topology/total-gpus"]
	if !ok {
		return nil
	}

	// 简化示例：实际实现需要 JSON 解析
	// 这里用硬编码模拟一个典型的 8-GPU NVLink 拓扑（如 DGX H100）
	total := parseInt(totalStr)
	allocated := parseInt(annotations["gpu.topology/allocated-gpus"])

	return &GPUTopology{
		TotalGPUs:     total,
		AllocatedGPUs: allocated,
		NVLinkGroups:  [][]int{{0, 1, 2, 3}, {4, 5, 6, 7}},
		PCIeGroups:    [][]int{{0, 1}, {2, 3}, {4, 5}, {6, 7}},
	}
}

// calculateTopologyScore 计算拓扑感知分数
func calculateTopologyScore(topo *GPUTopology, requestedGPUs int64) int64 {
	available := topo.TotalGPUs - topo.AllocatedGPUs

	// 可用 GPU 不够
	if int64(available) < requestedGPUs {
		return 0
	}

	// 情况 1：请求的 GPU 可以完全放在一个 NVLink 组内
	for _, group := range topo.NVLinkGroups {
		if int64(len(group)) >= requestedGPUs {
			return NVLinkScore
		}
	}

	// 情况 2：请求的 GPU 可以放在同一个 PCIe Switch 下
	for _, group := range topo.PCIeGroups {
		if int64(len(group)) >= requestedGPUs {
			return PCIeScore
		}
	}

	// 情况 3：需要跨 Socket
	return CrossSocket
}

// parseInt 简单的字符串转 int
func parseInt(s string) int {
	var n int
	fmt.Sscanf(s, "%d", &n)
	return n
}

// New 创建插件实例
func New(obj runtime.Object, handle framework.Handle) (framework.Plugin, error) {
	return &GPUTopologyScore{handle: handle}, nil
}

func main() {
	// 注册插件到调度器框架
	command := app.NewSchedulerCommand(
		app.WithPlugin(PluginName, New),
	)

	if err := command.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "调度器启动失败: %v\n", err)
		os.Exit(1)
	}
}

/*
 * 对应的 KubeSchedulerConfiguration:
 *
 * apiVersion: kubescheduler.config.k8s.io/v1
 * kind: KubeSchedulerConfiguration
 * profiles:
 * - schedulerName: gpu-topo-scheduler
 *   plugins:
 *     score:
 *       enabled:
 *       - name: GPUTopologyScore
 *         weight: 10
 *       disabled:
 *       - name: NodeResourcesBalancedAllocation
 */
