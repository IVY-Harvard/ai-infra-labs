"""
FastAPI 管理接口：提供训练、评估、部署的 HTTP API
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

app = FastAPI(title="Finetuning Platform API", version="1.0.0")

# 存储任务状态
jobs: Dict[str, Dict] = {}


# ============================================================
# Request/Response 模型
# ============================================================
class TrainRequest(BaseModel):
    config_path: Optional[str] = None
    model_name: str = "Qwen/Qwen2-7B"
    method: str = "lora"  # lora, qlora, dpo
    train_data: str = ""
    lora_r: int = 64
    lora_alpha: int = 128
    num_epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 4
    num_gpus: int = 1


class EvalRequest(BaseModel):
    model_path: str
    benchmarks: list = ["mmlu", "ceval"]
    custom_eval_file: Optional[str] = None


class DeployRequest(BaseModel):
    model_path: str
    deploy_name: str = "default"
    engine: str = "vllm"
    tensor_parallel: int = 1
    port: int = 8000


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float = 0.0
    result: Optional[Dict] = None
    error: Optional[str] = None
    created_at: str = ""


# ============================================================
# 训练 API
# ============================================================
@app.post("/api/train", response_model=JobStatus)
async def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """提交训练任务"""
    job_id = str(uuid.uuid4())[:8]

    job = {
        "job_id": job_id,
        "type": "train",
        "status": "queued",
        "progress": 0.0,
        "config": request.dict(),
        "created_at": datetime.now().isoformat(),
        "result": None,
        "error": None,
    }
    jobs[job_id] = job

    # 后台执行训练
    background_tasks.add_task(_run_training, job_id, request)

    return JobStatus(**job)


async def _run_training(job_id: str, request: TrainRequest):
    """后台训练任务"""
    job = jobs[job_id]
    job["status"] = "running"

    try:
        # 根据方法选择训练器
        if request.method == "lora" or request.method == "qlora":
            from src.trainer.lora_trainer import LoRATrainer, LoRATrainConfig

            config = LoRATrainConfig(
                model_name=request.model_name,
                train_data=request.train_data,
                lora_r=request.lora_r,
                lora_alpha=request.lora_alpha,
                num_epochs=request.num_epochs,
                learning_rate=request.learning_rate,
                batch_size=request.batch_size,
                use_qlora=(request.method == "qlora"),
                output_dir=f"./output/{job_id}",
            )
            trainer = LoRATrainer(config)

        elif request.method == "dpo":
            from src.trainer.dpo_trainer import DPOTrainerWrapper, DPOTrainConfig

            config = DPOTrainConfig(
                model_name=request.model_name,
                train_data=request.train_data,
                num_epochs=request.num_epochs,
                learning_rate=5e-7,
                output_dir=f"./output/{job_id}",
            )
            trainer = DPOTrainerWrapper(config)
        else:
            raise ValueError(f"不支持的方法: {request.method}")

        # 加载数据
        from src.data.data_loader import DataLoader
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(request.model_name, trust_remote_code=True)
        loader = DataLoader(tokenizer)
        dataset = loader.load(request.train_data)

        # 训练
        job["progress"] = 0.1
        trainer.setup()
        job["progress"] = 0.2

        result = trainer.train(dataset)

        job["status"] = "completed"
        job["progress"] = 1.0
        job["result"] = {
            "model_path": result.model_path,
            "train_loss": result.train_loss,
            "training_time": result.training_time_sec,
        }

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)


# ============================================================
# 评估 API
# ============================================================
@app.post("/api/eval", response_model=JobStatus)
async def start_evaluation(request: EvalRequest, background_tasks: BackgroundTasks):
    """提交评估任务"""
    job_id = str(uuid.uuid4())[:8]

    job = {
        "job_id": job_id,
        "type": "eval",
        "status": "queued",
        "progress": 0.0,
        "config": request.dict(),
        "created_at": datetime.now().isoformat(),
        "result": None,
        "error": None,
    }
    jobs[job_id] = job

    background_tasks.add_task(_run_evaluation, job_id, request)
    return JobStatus(**job)


async def _run_evaluation(job_id: str, request: EvalRequest):
    """后台评估任务"""
    job = jobs[job_id]
    job["status"] = "running"

    try:
        from src.evaluation.evaluator import Evaluator
        evaluator = Evaluator(request.model_path, request.benchmarks)
        result = evaluator.evaluate()

        job["status"] = "completed"
        job["progress"] = 1.0
        job["result"] = {
            "benchmarks": result.benchmarks,
            "custom_metrics": result.custom_metrics,
            "safety": result.safety_scores,
            "passed": result.passed,
        }
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)


# ============================================================
# 部署 API
# ============================================================
@app.post("/api/deploy")
async def deploy_model(request: DeployRequest):
    """部署模型"""
    from src.serving.deploy_manager import DeployManager, DeployConfig

    manager = DeployManager()
    config = DeployConfig(
        model_path=request.model_path,
        deploy_name=request.deploy_name,
        engine=request.engine,
        tensor_parallel=request.tensor_parallel,
        port=request.port,
    )

    result = manager.deploy(config)
    return result


@app.get("/api/deploy/list")
async def list_deployments():
    """列出部署"""
    from src.serving.deploy_manager import DeployManager
    manager = DeployManager()
    return manager.list_deployments()


# ============================================================
# 任务管理 API
# ============================================================
@app.get("/api/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """查询任务状态"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobStatus(**jobs[job_id])


@app.get("/api/jobs")
async def list_jobs():
    """列出所有任务"""
    return list(jobs.values())


# ============================================================
# 健康检查
# ============================================================
@app.get("/health")
async def health_check():
    """健康检查"""
    import torch
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "active_jobs": sum(1 for j in jobs.values() if j["status"] == "running"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
