"""
Job API — GPU 任务管理接口
"""

import uuid
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class JobSubmitRequest(BaseModel):
    """任务提交请求"""
    name: str
    tenant: str
    gpu_count: int = Field(ge=1, le=8)
    gpu_memory_min_gb: int = Field(default=0, ge=0)
    cpu_request: float = Field(default=4.0, ge=0)
    memory_request_gb: float = Field(default=32.0, ge=0)
    priority: int = Field(default=5, ge=1, le=10)
    prefer_nvlink: bool = True
    image: str = ""
    command: list[str] = []


class JobResponse(BaseModel):
    """任务响应"""
    id: str
    name: str
    tenant: str
    gpu_count: int
    state: str
    assigned_node: Optional[str] = None
    assigned_gpus: list[int] = []
    submit_time: str
    start_time: Optional[str] = None


def create_job_router() -> APIRouter:
    router = APIRouter(tags=["jobs"])

    @router.post("/jobs", response_model=JobResponse)
    async def submit_job(req: JobSubmitRequest, request: Request):
        """提交 GPU 任务"""
        scheduler = request.app.state.scheduler
        quota_manager = request.app.state.quota_manager

        # 检查配额
        allowed, reason = quota_manager.check_quota(req.tenant, req.gpu_count)
        if not allowed:
            raise HTTPException(status_code=403, detail=f"配额不足: {reason}")

        # 创建 Job
        from ..scheduler.gpu_scheduler import GPUJob
        job = GPUJob(
            id=str(uuid.uuid4()),
            name=req.name,
            tenant=req.tenant,
            gpu_count=req.gpu_count,
            gpu_memory_min_gb=req.gpu_memory_min_gb,
            cpu_request=req.cpu_request,
            memory_request_gb=req.memory_request_gb,
            priority=req.priority,
            prefer_nvlink=req.prefer_nvlink,
            image=req.image,
            command=req.command,
        )

        # 分配配额
        quota_manager.allocate(req.tenant, req.gpu_count)

        # 提交到调度器
        scheduler.submit_job(job)

        return JobResponse(
            id=job.id,
            name=job.name,
            tenant=job.tenant,
            gpu_count=job.gpu_count,
            state=job.state.value,
            submit_time=job.submit_time.isoformat(),
        )

    @router.get("/jobs/queue")
    async def get_queue(request: Request):
        """获取调度队列状态"""
        scheduler = request.app.state.scheduler
        return scheduler.get_queue_status()

    @router.delete("/jobs/{job_id}")
    async def cancel_job(job_id: str, request: Request):
        """取消任务"""
        scheduler = request.app.state.scheduler
        scheduler.release_job(job_id)
        return {"status": "cancelled", "job_id": job_id}

    return router
