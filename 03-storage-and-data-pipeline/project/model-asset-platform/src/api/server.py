"""模型资产平台 API 服务

基于 FastAPI 构建，提供：
- 模型注册与版本管理
- 模型上传 / 下载
- 分发任务管理
- Checkpoint 管理
- 健康检查与监控
"""

import os
import time
import tempfile
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..storage.backend import create_backend
from ..storage.cache_manager import CacheManager
from ..model.registry import ModelRegistry
from ..model.versioning import VersionManager
from ..model.validator import ModelValidator
from ..distribution.distributor import Distributor
from ..distribution.prewarmer import Prewarmer
from ..checkpoint.ckpt_manager import CheckpointManager


# ──────────── App 初始化 ────────────

app = FastAPI(
    title="Model Asset Platform",
    description="模型资产全生命周期管理平台",
    version="0.1.0",
)

# 可通过环境变量配置
STORAGE_TYPE = os.getenv("STORAGE_TYPE", "local")
STORAGE_ROOT = os.getenv("STORAGE_ROOT", "/tmp/model-assets")
CACHE_DIR = os.getenv("CACHE_DIR", "/tmp/model-cache")
CACHE_MAX_GB = int(os.getenv("CACHE_MAX_GB", "50"))

# 全局组件（启动时初始化）
backend = None
cache = None
registry = None
versioning = None
distributor = None
prewarmer = None


@app.on_event("startup")
def startup():
    global backend, cache, registry, versioning, distributor, prewarmer

    backend = create_backend(STORAGE_TYPE, root=STORAGE_ROOT)
    cache = CacheManager(
        cache_dir=CACHE_DIR,
        max_size_gb=CACHE_MAX_GB,
    )
    registry = ModelRegistry()
    versioning = VersionManager()
    distributor = Distributor(backend=backend, cache=cache)
    prewarmer = Prewarmer(backend=backend, cache=cache)


# ──────────── Schema ────────────

class ModelRegisterRequest(BaseModel):
    name: str
    framework: str = "pytorch"
    description: str = ""
    tags: list = []


class ModelVersionRequest(BaseModel):
    model_name: str
    version: str
    storage_key: str
    size_bytes: int = 0
    format: str = "safetensors"


class PrewarmRequest(BaseModel):
    model_key: str
    deadline_seconds: float = 60
    priority: Optional[int] = None


class DistributeRequest(BaseModel):
    model_key: str
    target_nodes: list = []


# ──────────── 健康检查 ────────────

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "storage_type": STORAGE_TYPE,
    }


@app.get("/metrics")
def metrics():
    """简单的运行指标"""
    return {
        "cache_stats": cache.stats() if cache else {},
        "registry_models": len(registry.models) if registry else 0,
        "distributor_tasks": len(distributor.tasks) if distributor else 0,
    }


# ──────────── 模型注册 ────────────

@app.post("/models/register")
def register_model(req: ModelRegisterRequest):
    """注册新模型"""
    entry = registry.register(
        name=req.name,
        framework=req.framework,
        description=req.description,
        tags=req.tags,
    )
    return {"status": "registered", "model": entry.__dict__}


@app.get("/models")
def list_models():
    """列出所有模型"""
    return {"models": registry.list_models()}


@app.get("/models/{model_name}")
def get_model(model_name: str):
    """获取模型详情"""
    entry = registry.get(model_name)
    if not entry:
        raise HTTPException(status_code=404, detail="Model not found")
    return entry.__dict__


# ──────────── 模型版本 ────────────

@app.post("/models/{model_name}/versions")
def add_version(model_name: str, req: ModelVersionRequest):
    """添加模型版本"""
    version = versioning.create_version(
        model_name=model_name,
        version=req.version,
        storage_key=req.storage_key,
        size_bytes=req.size_bytes,
        format=req.format,
    )
    return {"status": "created", "version": version}


@app.get("/models/{model_name}/versions")
def list_versions(model_name: str):
    """列出模型所有版本"""
    versions = versioning.list_versions(model_name)
    return {"model_name": model_name, "versions": versions}


# ──────────── 模型上传 / 下载 ────────────

@app.post("/upload/{model_key:path}")
async def upload_model(model_key: str, file: UploadFile = File(...)):
    """上传模型文件到存储后端"""
    data = await file.read()

    # 存储
    backend.put(model_key, data)

    # 计算校验和
    import hashlib
    checksum = hashlib.sha256(data).hexdigest()

    return {
        "status": "uploaded",
        "key": model_key,
        "size_bytes": len(data),
        "sha256": checksum,
    }


@app.get("/download/{model_key:path}")
def download_model(model_key: str):
    """下载模型文件"""
    # 先检查缓存
    cached = cache.get(model_key) if cache else None
    if cached:
        def iter_file():
            with open(cached, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    yield chunk
        return StreamingResponse(iter_file(),
                                 media_type="application/octet-stream")

    # 从后端读取
    data = backend.get(model_key)
    if data is None:
        raise HTTPException(status_code=404, detail="Model not found")

    # 缓存
    if cache:
        cache.put(model_key, data)

    def iter_bytes():
        offset = 0
        chunk_size = 1024 * 1024
        while offset < len(data):
            yield data[offset:offset + chunk_size]
            offset += chunk_size

    return StreamingResponse(iter_bytes(),
                             media_type="application/octet-stream")


# ──────────── 模型校验 ────────────

@app.post("/validate/{model_key:path}")
def validate_model(model_key: str):
    """校验模型文件完整性"""
    # 获取本地路径
    local_path = cache.get(model_key) if cache else None
    if not local_path:
        # 下载到临时目录
        data = backend.get(model_key)
        if data is None:
            raise HTTPException(status_code=404, detail="Model not found")
        local_path = cache.put(model_key, data) if cache else None
        if not local_path:
            raise HTTPException(status_code=500,
                                detail="Cannot cache for validation")

    result = ModelValidator.validate(local_path)
    return result


# ──────────── 分发 ────────────

@app.post("/distribute")
def distribute_model(req: DistributeRequest):
    """分发模型到目标节点"""
    result = distributor.distribute(
        model_key=req.model_key,
        target_nodes=req.target_nodes or None,
    )
    return result


@app.get("/distribute/{task_id}")
def get_distribution_status(task_id: str):
    """查询分发任务状态"""
    status = distributor.get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status


# ──────────── 预热 ────────────

@app.post("/prewarm")
def submit_prewarm(req: PrewarmRequest):
    """提交预热任务"""
    prewarmer.submit(
        model_key=req.model_key,
        deadline_seconds=req.deadline_seconds,
        priority=req.priority,
    )
    return {"status": "submitted", "model_key": req.model_key}


@app.post("/prewarm/process")
def process_prewarm_queue():
    """处理预热队列"""
    prewarmer.process_queue()
    return prewarmer.status()


# ──────────── Checkpoint ────────────

@app.get("/checkpoints")
def list_checkpoints(local_dir: str = Query(default="/tmp/checkpoints")):
    """列出本地 checkpoint"""
    ckpt_mgr = CheckpointManager(local_dir=local_dir, backend=backend)
    return {"checkpoints": ckpt_mgr.list_checkpoints()}
