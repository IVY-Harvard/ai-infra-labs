"""
GPU 节点故障检测与处理控制器

功能：
  1. 定期检查所有 GPU 节点的健康状态
  2. 检测 ECC 错误、温度异常、GPU 挂死等问题
  3. 自动给故障节点添加 Taint 隔离
  4. 触发 Pod 迁移和告警

在 K8s 中以 Deployment 方式运行，需要 RBAC 权限操作 Node Taint。

使用方式：
    python node_failure_handler.py --kubeconfig=/path/to/kubeconfig
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

try:
    from kubernetes import client, config, watch
    from kubernetes.client.rest import ApiException
except ImportError:
    print("需要安装 kubernetes 库: pip install kubernetes")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("gpu-fault-handler")


class GPUHealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"      # 性能下降但仍可用
    UNHEALTHY = "unhealthy"    # 需要隔离
    UNKNOWN = "unknown"


@dataclass
class GPUHealthMetrics:
    """单个 GPU 的健康指标"""
    gpu_index: int
    temperature: float = 0.0
    power_usage: float = 0.0
    ecc_single_bit_errors: int = 0
    ecc_double_bit_errors: int = 0
    memory_used_mb: int = 0
    memory_total_mb: int = 0
    gpu_utilization: float = 0.0
    xid_errors: list[int] = field(default_factory=list)
    nvidia_smi_responsive: bool = True


@dataclass
class NodeGPUHealth:
    """节点级别的 GPU 健康状态"""
    node_name: str
    gpus: list[GPUHealthMetrics]
    overall_status: GPUHealthStatus = GPUHealthStatus.HEALTHY
    last_check_time: Optional[datetime] = None
    failure_reason: str = ""


class GPUHealthChecker:
    """GPU 健康检查器"""

    def __init__(
        self,
        ecc_dbe_threshold: int = 5,       # Double-bit ECC 错误阈值
        temp_threshold: float = 85.0,      # 温度阈值 (°C)
        unresponsive_timeout: int = 30,    # nvidia-smi 无响应超时 (秒)
    ):
        self.ecc_dbe_threshold = ecc_dbe_threshold
        self.temp_threshold = temp_threshold
        self.unresponsive_timeout = unresponsive_timeout

    def check_node_health(self, node_name: str, metrics: list[GPUHealthMetrics]) -> NodeGPUHealth:
        """检查节点所有 GPU 的健康状态"""
        health = NodeGPUHealth(
            node_name=node_name,
            gpus=metrics,
            last_check_time=datetime.now(),
        )

        unhealthy_reasons = []
        degraded_reasons = []

        for gpu in metrics:
            # 检查 1: nvidia-smi 是否响应
            if not gpu.nvidia_smi_responsive:
                unhealthy_reasons.append(
                    f"GPU {gpu.gpu_index}: nvidia-smi 无响应"
                )
                continue

            # 检查 2: ECC Double-bit 错误（不可纠正）
            if gpu.ecc_double_bit_errors >= self.ecc_dbe_threshold:
                unhealthy_reasons.append(
                    f"GPU {gpu.gpu_index}: ECC DBE={gpu.ecc_double_bit_errors} >= {self.ecc_dbe_threshold}"
                )

            # 检查 3: 温度过高
            if gpu.temperature >= self.temp_threshold:
                degraded_reasons.append(
                    f"GPU {gpu.gpu_index}: 温度 {gpu.temperature}°C >= {self.temp_threshold}°C"
                )

            # 检查 4: XID 错误（GPU 内部错误）
            critical_xid = {31, 43, 45, 48, 61, 62, 63, 64, 68, 69, 74, 79, 92, 94, 95, 119}
            if set(gpu.xid_errors) & critical_xid:
                unhealthy_reasons.append(
                    f"GPU {gpu.gpu_index}: 严重 XID 错误 {gpu.xid_errors}"
                )

        # 确定整体状态
        if unhealthy_reasons:
            health.overall_status = GPUHealthStatus.UNHEALTHY
            health.failure_reason = "; ".join(unhealthy_reasons)
        elif degraded_reasons:
            health.overall_status = GPUHealthStatus.DEGRADED
            health.failure_reason = "; ".join(degraded_reasons)
        else:
            health.overall_status = GPUHealthStatus.HEALTHY

        return health


class NodeTaintManager:
    """管理节点 Taint"""

    TAINT_KEY_UNHEALTHY = "nvidia.com/gpu-unhealthy"
    TAINT_KEY_DEGRADED = "nvidia.com/gpu-degraded"

    def __init__(self, k8s_client: client.CoreV1Api):
        self.v1 = k8s_client

    def add_taint(self, node_name: str, key: str, value: str, effect: str):
        """给节点添加 Taint"""
        try:
            node = self.v1.read_node(node_name)

            # 检查 Taint 是否已存在
            existing_taints = node.spec.taints or []
            for taint in existing_taints:
                if taint.key == key:
                    logger.info(f"节点 {node_name} 已有 Taint {key}")
                    return

            # 添加新 Taint
            new_taint = client.V1Taint(
                key=key,
                value=value,
                effect=effect,
                time_added=datetime.now(),
            )
            existing_taints.append(new_taint)

            body = {"spec": {"taints": [
                {"key": t.key, "value": t.value, "effect": t.effect}
                for t in existing_taints
            ]}}

            self.v1.patch_node(node_name, body)
            logger.warning(f"已给节点 {node_name} 添加 Taint: {key}={value}:{effect}")

        except ApiException as e:
            logger.error(f"添加 Taint 失败: {e}")

    def remove_taint(self, node_name: str, key: str):
        """移除节点的 Taint"""
        try:
            node = self.v1.read_node(node_name)
            existing_taints = node.spec.taints or []

            new_taints = [t for t in existing_taints if t.key != key]

            if len(new_taints) == len(existing_taints):
                logger.info(f"节点 {node_name} 没有 Taint {key}")
                return

            body = {"spec": {"taints": [
                {"key": t.key, "value": t.value, "effect": t.effect}
                for t in new_taints
            ] if new_taints else None}}

            self.v1.patch_node(node_name, body)
            logger.info(f"已移除节点 {node_name} 的 Taint: {key}")

        except ApiException as e:
            logger.error(f"移除 Taint 失败: {e}")


class GPUFaultController:
    """GPU 故障控制器主循环"""

    def __init__(
        self,
        check_interval: int = 30,
        consecutive_failures: int = 3,
    ):
        """
        Args:
            check_interval: 检查间隔（秒）
            consecutive_failures: 连续多少次检查失败才触发隔离
        """
        self.check_interval = check_interval
        self.consecutive_failures = consecutive_failures
        self.checker = GPUHealthChecker()
        self.failure_counts: dict[str, int] = {}  # node_name → 连续失败次数

        # 初始化 K8s 客户端
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        self.v1 = client.CoreV1Api()
        self.taint_manager = NodeTaintManager(self.v1)

    def get_gpu_nodes(self) -> list[str]:
        """获取所有 GPU 节点"""
        nodes = self.v1.list_node(
            label_selector="nvidia.com/gpu.present=true"
        )
        return [node.metadata.name for node in nodes.items]

    def collect_gpu_metrics(self, node_name: str) -> list[GPUHealthMetrics]:
        """
        收集节点的 GPU 指标。

        实际实现应该从以下来源获取：
          1. DCGM Exporter (Prometheus metrics)
          2. Node 上的 DaemonSet 直接执行 nvidia-smi
          3. 自定义 metrics API
        """
        # 模拟实现 — 实际应查询 Prometheus/DCGM
        # 这里返回模拟数据
        metrics = []
        for i in range(8):
            metrics.append(GPUHealthMetrics(
                gpu_index=i,
                temperature=55.0 + i * 2,  # 模拟温度
                ecc_double_bit_errors=0,
                nvidia_smi_responsive=True,
                xid_errors=[],
            ))
        return metrics

    def handle_health_result(self, health: NodeGPUHealth):
        """处理健康检查结果"""
        node_name = health.node_name

        if health.overall_status == GPUHealthStatus.HEALTHY:
            # 健康：重置计数，移除可能存在的 Taint
            if node_name in self.failure_counts:
                logger.info(f"节点 {node_name} 恢复健康")
                del self.failure_counts[node_name]
                self.taint_manager.remove_taint(
                    node_name, NodeTaintManager.TAINT_KEY_UNHEALTHY
                )
                self.taint_manager.remove_taint(
                    node_name, NodeTaintManager.TAINT_KEY_DEGRADED
                )

        elif health.overall_status == GPUHealthStatus.DEGRADED:
            # 降级：添加 NoSchedule Taint（不驱逐现有 Pod）
            logger.warning(f"节点 {node_name} GPU 降级: {health.failure_reason}")
            self.taint_manager.add_taint(
                node_name,
                NodeTaintManager.TAINT_KEY_DEGRADED,
                "true",
                "NoSchedule",
            )

        elif health.overall_status == GPUHealthStatus.UNHEALTHY:
            # 不健康：累计失败次数
            self.failure_counts[node_name] = self.failure_counts.get(node_name, 0) + 1
            count = self.failure_counts[node_name]

            logger.error(
                f"节点 {node_name} GPU 不健康 ({count}/{self.consecutive_failures}): "
                f"{health.failure_reason}"
            )

            if count >= self.consecutive_failures:
                # 达到阈值：添加 NoExecute Taint（驱逐 Pod）
                logger.critical(f"节点 {node_name} 连续 {count} 次检测不健康，执行隔离！")
                self.taint_manager.add_taint(
                    node_name,
                    NodeTaintManager.TAINT_KEY_UNHEALTHY,
                    health.failure_reason[:63],  # Taint value 有长度限制
                    "NoExecute",
                )
                self._send_alert(health)

    def _send_alert(self, health: NodeGPUHealth):
        """发送告警通知"""
        logger.critical(
            f"[ALERT] GPU 故障隔离: node={health.node_name}, "
            f"reason={health.failure_reason}"
        )
        # 实际实现：发送到 Slack/PagerDuty/AlertManager
        # 可以通过 K8s Event 记录
        try:
            event = client.CoreV1Api().create_namespaced_event(
                namespace="default",
                body=client.CoreV1Event(
                    metadata=client.V1ObjectMeta(
                        generate_name="gpu-fault-",
                    ),
                    reason="GPUUnhealthy",
                    message=f"Node {health.node_name}: {health.failure_reason}",
                    type="Warning",
                    involved_object=client.V1ObjectReference(
                        kind="Node",
                        name=health.node_name,
                    ),
                    first_timestamp=datetime.now(),
                    last_timestamp=datetime.now(),
                ),
            )
        except Exception as e:
            logger.error(f"创建 Event 失败: {e}")

    def run(self):
        """主控制循环"""
        logger.info("GPU 故障控制器启动")
        logger.info(f"检查间隔: {self.check_interval}s, 阈值: {self.consecutive_failures} 次")

        while True:
            try:
                gpu_nodes = self.get_gpu_nodes()
                logger.info(f"检查 {len(gpu_nodes)} 个 GPU 节点...")

                for node_name in gpu_nodes:
                    metrics = self.collect_gpu_metrics(node_name)
                    health = self.checker.check_node_health(node_name, metrics)
                    self.handle_health_result(health)

            except Exception as e:
                logger.error(f"检查循环出错: {e}", exc_info=True)

            time.sleep(self.check_interval)


def main():
    parser = argparse.ArgumentParser(description="GPU 节点故障检测与处理控制器")
    parser.add_argument("--check-interval", type=int, default=30, help="检查间隔（秒）")
    parser.add_argument("--consecutive-failures", type=int, default=3, help="连续失败阈值")
    parser.add_argument("--kubeconfig", default=None, help="Kubeconfig 路径")
    args = parser.parse_args()

    if args.kubeconfig:
        os.environ["KUBECONFIG"] = args.kubeconfig

    controller = GPUFaultController(
        check_interval=args.check_interval,
        consecutive_failures=args.consecutive_failures,
    )
    controller.run()


if __name__ == "__main__":
    main()
