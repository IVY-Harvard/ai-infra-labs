"""
K8s API 客户端封装

提供与 K8s API Server 交互的高层接口，
包括节点管理、Pod 操作、事件创建等。
"""

import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from kubernetes import client, config, watch
    from kubernetes.client.rest import ApiException
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False
    logger.warning("kubernetes 库未安装，使用 mock 模式")


@dataclass
class PodInfo:
    """Pod 信息"""
    name: str
    namespace: str
    node_name: Optional[str] = None
    gpu_request: int = 0
    phase: str = "Pending"
    labels: dict = None
    annotations: dict = None


class K8sClient:
    """K8s API 客户端"""

    def __init__(self, kubeconfig: Optional[str] = None, in_cluster: bool = False):
        self._initialized = False
        if not K8S_AVAILABLE:
            logger.warning("K8s client in mock mode")
            return

        try:
            if in_cluster:
                config.load_incluster_config()
            elif kubeconfig:
                config.load_kube_config(config_file=kubeconfig)
            else:
                config.load_kube_config()
            self._initialized = True
        except Exception as e:
            logger.error(f"K8s 配置加载失败: {e}")

        if self._initialized:
            self.v1 = client.CoreV1Api()
            self.apps_v1 = client.AppsV1Api()
            self.scheduling_v1 = client.SchedulingV1Api()

    @property
    def connected(self) -> bool:
        return self._initialized

    def list_gpu_nodes(self) -> list[dict]:
        """获取所有 GPU 节点信息"""
        if not self._initialized:
            return []

        try:
            nodes = self.v1.list_node(
                label_selector="nvidia.com/gpu.present=true"
            )
            result = []
            for node in nodes.items:
                allocatable = node.status.allocatable or {}
                capacity = node.status.capacity or {}

                result.append({
                    "name": node.metadata.name,
                    "labels": node.metadata.labels or {},
                    "annotations": node.metadata.annotations or {},
                    "gpu_capacity": int(capacity.get("nvidia.com/gpu", "0")),
                    "gpu_allocatable": int(allocatable.get("nvidia.com/gpu", "0")),
                    "cpu_allocatable": float(allocatable.get("cpu", "0").rstrip("m")) / 1000
                        if "m" in allocatable.get("cpu", "0")
                        else float(allocatable.get("cpu", "0")),
                    "memory_allocatable_gb": int(allocatable.get("memory", "0Gi").rstrip("KiMiGi")) / (1024**2)
                        if "Ki" in allocatable.get("memory", "0")
                        else 0,
                    "ready": self._is_node_ready(node),
                    "taints": [
                        f"{t.key}={t.value}:{t.effect}"
                        for t in (node.spec.taints or [])
                    ],
                })
            return result
        except ApiException as e:
            logger.error(f"获取节点列表失败: {e}")
            return []

    def list_gpu_pods(self, namespace: str = "") -> list[PodInfo]:
        """获取所有使用 GPU 的 Pod"""
        if not self._initialized:
            return []

        try:
            if namespace:
                pods = self.v1.list_namespaced_pod(namespace)
            else:
                pods = self.v1.list_pod_for_all_namespaces()

            gpu_pods = []
            for pod in pods.items:
                gpu_request = self._get_pod_gpu_request(pod)
                if gpu_request > 0:
                    gpu_pods.append(PodInfo(
                        name=pod.metadata.name,
                        namespace=pod.metadata.namespace,
                        node_name=pod.spec.node_name,
                        gpu_request=gpu_request,
                        phase=pod.status.phase or "Unknown",
                        labels=pod.metadata.labels or {},
                        annotations=pod.metadata.annotations or {},
                    ))
            return gpu_pods
        except ApiException as e:
            logger.error(f"获取 Pod 列表失败: {e}")
            return []

    def bind_pod(self, pod_name: str, namespace: str, node_name: str) -> bool:
        """将 Pod 绑定到节点"""
        if not self._initialized:
            return False

        try:
            binding = client.V1Binding(
                metadata=client.V1ObjectMeta(name=pod_name),
                target=client.V1ObjectReference(
                    api_version="v1",
                    kind="Node",
                    name=node_name,
                ),
            )
            self.v1.create_namespaced_binding(
                namespace=namespace,
                body=binding,
            )
            logger.info(f"绑定 Pod {namespace}/{pod_name} → {node_name}")
            return True
        except ApiException as e:
            logger.error(f"绑定失败: {e}")
            return False

    def taint_node(self, node_name: str, key: str, value: str, effect: str) -> bool:
        """给节点添加 Taint"""
        if not self._initialized:
            return False

        try:
            node = self.v1.read_node(node_name)
            taints = node.spec.taints or []

            # 避免重复添加
            for t in taints:
                if t.key == key:
                    return True

            taints.append(client.V1Taint(key=key, value=value, effect=effect))
            body = {"spec": {"taints": [
                {"key": t.key, "value": t.value, "effect": t.effect}
                for t in taints
            ]}}
            self.v1.patch_node(node_name, body)
            logger.info(f"添加 Taint: {node_name} {key}={value}:{effect}")
            return True
        except ApiException as e:
            logger.error(f"添加 Taint 失败: {e}")
            return False

    def untaint_node(self, node_name: str, key: str) -> bool:
        """移除节点的 Taint"""
        if not self._initialized:
            return False

        try:
            node = self.v1.read_node(node_name)
            taints = node.spec.taints or []
            new_taints = [t for t in taints if t.key != key]

            body = {"spec": {"taints": [
                {"key": t.key, "value": t.value, "effect": t.effect}
                for t in new_taints
            ] if new_taints else None}}
            self.v1.patch_node(node_name, body)
            logger.info(f"移除 Taint: {node_name} {key}")
            return True
        except ApiException as e:
            logger.error(f"移除 Taint 失败: {e}")
            return False

    def create_event(
        self,
        name: str,
        namespace: str,
        reason: str,
        message: str,
        event_type: str = "Normal",
    ):
        """创建 K8s Event"""
        if not self._initialized:
            return

        try:
            from datetime import datetime
            self.v1.create_namespaced_event(
                namespace=namespace,
                body=client.CoreV1Event(
                    metadata=client.V1ObjectMeta(generate_name=f"{name}-"),
                    reason=reason,
                    message=message,
                    type=event_type,
                    involved_object=client.V1ObjectReference(
                        kind="Pod",
                        name=name,
                        namespace=namespace,
                    ),
                    first_timestamp=datetime.now(),
                    last_timestamp=datetime.now(),
                ),
            )
        except Exception as e:
            logger.error(f"创建 Event 失败: {e}")

    def _get_pod_gpu_request(self, pod) -> int:
        """获取 Pod 的 GPU 请求量"""
        total = 0
        for container in (pod.spec.containers or []):
            limits = container.resources.limits if container.resources else None
            if limits and "nvidia.com/gpu" in limits:
                total += int(limits["nvidia.com/gpu"])
        return total

    def _is_node_ready(self, node) -> bool:
        """检查节点是否 Ready"""
        for condition in (node.status.conditions or []):
            if condition.type == "Ready":
                return condition.status == "True"
        return False
