"""
FastAPI 服务器 — GPU 调度器 HTTP API 入口
"""

import logging
import argparse
import threading
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from ..scheduler.gpu_scheduler import GPUScheduler, SchedulingStrategy
from ..tenant.quota_manager import QuotaManager, TenantQuota
from ..resource.node_manager import NodeManager
from ..resource.utilization_tracker import UtilizationTracker
from ..k8s.client import K8sClient
from .job_api import create_job_router
from .cluster_api import create_cluster_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app(
    kubeconfig: Optional[str] = None,
    strategy: str = "topology_aware",
    total_gpus: int = 8,
) -> FastAPI:
    """创建 FastAPI 应用"""
    app = FastAPI(
        title="GPU Cluster Scheduler",
        description="面向多租户 GPU 集群的智能调度系统",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 初始化组件
    k8s_client = K8sClient(kubeconfig=kubeconfig)
    scheduler = GPUScheduler(strategy=SchedulingStrategy(strategy))
    quota_manager = QuotaManager(total_cluster_gpus=total_gpus)
    node_manager = NodeManager()
    tracker = UtilizationTracker()

    # 注册默认租户
    default_tenants = [
        TenantQuota("team-training", gpu_quota=4, gpu_burst_limit=8, priority_default=7),
        TenantQuota("team-inference", gpu_quota=2, gpu_burst_limit=4, priority_default=8),
        TenantQuota("team-experiment", gpu_quota=2, gpu_burst_limit=6, priority_default=3,
                    preemptible=True),
    ]
    for tenant in default_tenants:
        quota_manager.register_tenant(tenant)

    # 注入到路由
    app.state.scheduler = scheduler
    app.state.quota_manager = quota_manager
    app.state.node_manager = node_manager
    app.state.tracker = tracker
    app.state.k8s_client = k8s_client

    # 注册路由
    app.include_router(create_job_router(), prefix="/api/v1")
    app.include_router(create_cluster_router(), prefix="/api/v1")

    @app.get("/healthz")
    def health_check():
        return {"status": "ok"}

    @app.on_event("startup")
    async def startup():
        logger.info("GPU Scheduler API 启动")
        # 启动调度循环（后台线程）
        scheduler_thread = threading.Thread(
            target=scheduler.run, daemon=True
        )
        # scheduler_thread.start()  # 生产环境取消注释

    return app


def main():
    parser = argparse.ArgumentParser(description="GPU Cluster Scheduler API Server")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--kubeconfig", default=None, help="Kubeconfig 路径")
    parser.add_argument("--strategy", default="topology_aware",
                        choices=["topology_aware", "bin_packing", "spread"])
    parser.add_argument("--total-gpus", type=int, default=8, help="集群总 GPU 数")
    args = parser.parse_args()

    app = create_app(
        kubeconfig=args.kubeconfig,
        strategy=args.strategy,
        total_gpus=args.total_gpus,
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
