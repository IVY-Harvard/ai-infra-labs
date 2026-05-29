/*
 * Fake GPU Device Plugin
 *
 * 实现一个模拟的 GPU Device Plugin，用于理解 K8s Device Plugin 框架。
 * 注册资源类型 "fake.com/gpu"，模拟 8 个 GPU 设备。
 *
 * 编译：go build -o fake-gpu-plugin .
 * 运行：./fake-gpu-plugin
 */

package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"google.golang.org/grpc"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
)

const (
	// 资源名称
	resourceName = "fake.com/gpu"

	// Socket 路径
	socketName = "fake-gpu.sock"
	socketDir  = "/var/lib/kubelet/device-plugins"

	// 模拟的 GPU 数量
	numGPUs = 8
)

// FakeGPUPlugin 实现 DevicePluginServer 接口
type FakeGPUPlugin struct {
	pluginapi.UnimplementedDevicePluginServer
	devices []*pluginapi.Device
	server  *grpc.Server
	stop    chan struct{}
}

// NewFakeGPUPlugin 创建 Fake GPU Plugin 实例
func NewFakeGPUPlugin() *FakeGPUPlugin {
	devices := make([]*pluginapi.Device, numGPUs)
	for i := 0; i < numGPUs; i++ {
		devices[i] = &pluginapi.Device{
			ID:     fmt.Sprintf("fake-gpu-%d", i),
			Health: pluginapi.Healthy,
		}
	}
	return &FakeGPUPlugin{
		devices: devices,
		stop:    make(chan struct{}),
	}
}

// GetDevicePluginOptions 返回插件选项
func (p *FakeGPUPlugin) GetDevicePluginOptions(
	ctx context.Context,
	req *pluginapi.Empty,
) (*pluginapi.DevicePluginOptions, error) {
	return &pluginapi.DevicePluginOptions{
		PreStartRequired:                false,
		GetPreferredAllocationAvailable: true,
	}, nil
}

// ListAndWatch 持续向 kubelet 上报设备列表
func (p *FakeGPUPlugin) ListAndWatch(
	_ *pluginapi.Empty,
	stream pluginapi.DevicePlugin_ListAndWatchServer,
) error {
	log.Printf("ListAndWatch 被调用，上报 %d 个 GPU 设备\n", len(p.devices))

	// 首次上报所有设备
	resp := &pluginapi.ListAndWatchResponse{Devices: p.devices}
	if err := stream.Send(resp); err != nil {
		return err
	}

	// 持续监听设备状态变化
	// 实际 NVIDIA Plugin 会定期检查 nvidia-smi/DCGM
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-p.stop:
			return nil
		case <-ticker.C:
			// 模拟健康检查 — 每 30 秒重新上报
			// 真实场景：检查 ECC 错误、温度、XID 错误等
			log.Println("健康检查：所有 GPU 正常")
			if err := stream.Send(resp); err != nil {
				return err
			}
		}
	}
}

// Allocate 当 Pod 请求 GPU 时被调用
func (p *FakeGPUPlugin) Allocate(
	ctx context.Context,
	req *pluginapi.AllocateRequest,
) (*pluginapi.AllocateResponse, error) {
	responses := &pluginapi.AllocateResponse{}

	for _, containerReq := range req.ContainerRequests {
		deviceIDs := containerReq.DevicesIDs
		log.Printf("Allocate 请求：设备 IDs = %v\n", deviceIDs)

		// 构造容器的设备响应
		containerResp := &pluginapi.ContainerAllocateResponse{
			// 注入环境变量（类似 NVIDIA_VISIBLE_DEVICES）
			Envs: map[string]string{
				"FAKE_GPU_IDS":     joinIDs(deviceIDs),
				"FAKE_GPU_COUNT":   fmt.Sprintf("%d", len(deviceIDs)),
				"FAKE_CUDA_VERSION": "12.2",
			},
			// 挂载设备文件（模拟）
			Mounts: []*pluginapi.Mount{
				{
					ContainerPath: "/dev/fake-gpu",
					HostPath:      "/dev/null",
					ReadOnly:      true,
				},
			},
			// 设备节点（模拟）
			Devices: []*pluginapi.DeviceSpec{
				{
					ContainerPath: "/dev/fake-nvidia0",
					HostPath:      "/dev/null",
					Permissions:   "rw",
				},
			},
		}
		responses.ContainerResponses = append(responses.ContainerResponses, containerResp)
	}

	return responses, nil
}

// GetPreferredAllocation 返回推荐的设备分配组合
// 用于实现拓扑感知分配 — 优先分配 NVLink 互连的 GPU
func (p *FakeGPUPlugin) GetPreferredAllocation(
	ctx context.Context,
	req *pluginapi.PreferredAllocationRequest,
) (*pluginapi.PreferredAllocationResponse, error) {
	resp := &pluginapi.PreferredAllocationResponse{}

	for _, containerReq := range req.ContainerRequests {
		available := containerReq.AvailableDeviceIDs
		count := int(containerReq.AllocationSize)

		log.Printf("GetPreferredAllocation: 从 %v 中选 %d 个\n", available, count)

		// 拓扑感知策略：优先选择编号连续的 GPU（模拟 NVLink 组）
		// 真实场景：查询 nvidia-smi topo -m 获取拓扑信息
		preferred := available[:count] // 简化：选前 N 个

		resp.ContainerResponses = append(resp.ContainerResponses,
			&pluginapi.ContainerPreferredAllocationResponse{
				DeviceIDs: preferred,
			})
	}

	return resp, nil
}

// PreStartContainer 在容器启动前调用（可选）
func (p *FakeGPUPlugin) PreStartContainer(
	ctx context.Context,
	req *pluginapi.PreStartContainerRequest,
) (*pluginapi.PreStartContainerResponse, error) {
	return &pluginapi.PreStartContainerResponse{}, nil
}

// Start 启动 gRPC server
func (p *FakeGPUPlugin) Start() error {
	socketPath := filepath.Join(socketDir, socketName)

	// 清理旧 socket
	os.Remove(socketPath)

	listener, err := net.Listen("unix", socketPath)
	if err != nil {
		return fmt.Errorf("监听 socket 失败: %v", err)
	}

	p.server = grpc.NewServer()
	pluginapi.RegisterDevicePluginServer(p.server, p)

	go func() {
		log.Printf("Fake GPU Plugin gRPC server 启动: %s\n", socketPath)
		if err := p.server.Serve(listener); err != nil {
			log.Printf("gRPC server 错误: %v\n", err)
		}
	}()

	// 等待 server 就绪
	time.Sleep(time.Second)

	// 向 kubelet 注册
	return p.register()
}

// register 向 kubelet 注册 Device Plugin
func (p *FakeGPUPlugin) register() error {
	kubeletSocket := filepath.Join(socketDir, "kubelet.sock")
	conn, err := grpc.Dial(
		"unix://"+kubeletSocket,
		grpc.WithInsecure(),
		grpc.WithBlock(),
		grpc.WithTimeout(5*time.Second),
	)
	if err != nil {
		return fmt.Errorf("连接 kubelet 失败: %v", err)
	}
	defer conn.Close()

	client := pluginapi.NewRegistrationClient(conn)
	_, err = client.Register(context.Background(), &pluginapi.RegisterRequest{
		Version:      pluginapi.Version,
		Endpoint:     socketName,
		ResourceName: resourceName,
		Options: &pluginapi.DevicePluginOptions{
			PreStartRequired:                false,
			GetPreferredAllocationAvailable: true,
		},
	})
	if err != nil {
		return fmt.Errorf("注册到 kubelet 失败: %v", err)
	}

	log.Printf("成功注册资源 %s 到 kubelet\n", resourceName)
	return nil
}

// Stop 停止插件
func (p *FakeGPUPlugin) Stop() {
	close(p.stop)
	if p.server != nil {
		p.server.Stop()
	}
}

func joinIDs(ids []string) string {
	result := ""
	for i, id := range ids {
		if i > 0 {
			result += ","
		}
		result += id
	}
	return result
}

func main() {
	log.Println("启动 Fake GPU Device Plugin...")

	plugin := NewFakeGPUPlugin()

	if err := plugin.Start(); err != nil {
		log.Fatalf("启动失败: %v", err)
	}

	// 等待终止信号
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	log.Println("收到终止信号，清理中...")
	plugin.Stop()
}
