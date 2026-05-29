"""
Cluster API — 集群状态查询接口
"""

import logging
from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)


def create_cluster_router() -> APIRouter:
    router = APIRouter(tags=["cluster"])

    @router.get("/cluster/status")
    async def cluster_status(request: Request):
        """获取集群整体状态"""
        tracker = request.app.state.tracker
        quota_manager = request.app.state.quota_manager
        k8s_client = request.app.state.k8s_client

        # GPU 节点信息
        nodes = []
        if k8s_client.connected:
            nodes = k8s_client.list_gpu_nodes()

        # 利用率
        utilization = tracker.get_cluster_utilization()

        # 租户配额
        tenants = quota_manager.get_all_tenants_status()

        return {
            "cluster": {
                "total_gpu_nodes": len(nodes),
                "healthy_nodes": sum(1 for n in nodes if n.get("ready")),
                "total_gpus": sum(n.get("gpu_capacity", 0) for n in nodes),
                "allocatable_gpus": sum(n.get("gpu_allocatable", 0) for n in nodes),
            },
            "utilization": utilization,
            "tenants": tenants,
            "nodes": nodes,
        }

    @router.get("/cluster/nodes")
    async def list_nodes(request: Request):
        """获取所有 GPU 节点详情"""
        k8s_client = request.app.state.k8s_client
        if k8s_client.connected:
            return k8s_client.list_gpu_nodes()
        return []

    @router.get("/cluster/utilization")
    async def utilization(request: Request):
        """获取 GPU 利用率"""
        tracker = request.app.state.tracker
        return tracker.get_cluster_utilization()

    @router.get("/cluster/tenants")
    async def tenants(request: Request):
        """获取所有租户配额状态"""
        quota_manager = request.app.state.quota_manager
        return quota_manager.get_all_tenants_status()

    @router.get("/cluster/tenants/{tenant_name}")
    async def tenant_detail(tenant_name: str, request: Request):
        """获取单个租户详情"""
        quota_manager = request.app.state.quota_manager
        status = quota_manager.get_tenant_status(tenant_name)
        if not status:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"租户 {tenant_name} 不存在")
        return status

    return router
