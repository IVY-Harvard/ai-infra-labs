"""REST API 服务 — 基于 FastAPI 的监控仪表盘后端

提供统一的 HTTP API 供前端仪表盘和外部系统调用:

端点:
- GET  /api/v1/health      健康检查
- GET  /api/v1/metrics      GPU 及推理指标
- GET  /api/v1/alerts       告警列表与状态
- POST /api/v1/alerts/{id}/ack  确认告警
- GET  /api/v1/capacity     容量规划数据
- GET  /api/v1/cost         成本分析数据
- GET  /api/v1/anomalies    异常检测结果
- GET  /api/v1/performance  性能建议

架构:
┌───────────┐     ┌──────────┐     ┌────────────┐
│  Grafana  │────▶│ FastAPI  │────▶│ Collectors │
│  前端      │     │  API     │     │ Analytics  │
└───────────┘     └──────────┘     │ Alerting   │
                                   └────────────┘
"""

import time
import logging
from typing import Dict, List, Optional
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.collectors.gpu_collector import GPUCollector
from src.collectors.inference_collector import InferenceCollector
from src.collectors.training_collector import TrainingCollector
from src.collectors.network_collector import NetworkCollector
from src.analytics.anomaly_engine import AnomalyEngine
from src.analytics.capacity_planner import CapacityPlanner
from src.analytics.cost_analyzer import CostAnalyzer
from src.analytics.performance_advisor import PerformanceAdvisor
from src.alerting.alert_manager import AlertManager

logger = logging.getLogger(__name__)

# ──────────────────────────── FastAPI 应用 ────────────────────────────

app = FastAPI(
    title="AI Observability Platform",
    description="GPU 集群与推理服务的统一监控平台 API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────── 全局组件初始化 ────────────────────────────

gpu_collector = GPUCollector(use_nvml=False)  # 默认使用 mock 数据
inference_collector = InferenceCollector()
training_collector = TrainingCollector()
network_collector = NetworkCollector()

anomaly_engine = AnomalyEngine()
capacity_planner = CapacityPlanner()
cost_analyzer = CostAnalyzer()
performance_advisor = PerformanceAdvisor()

alert_manager = AlertManager()

_start_time = time.time()

# ──────────────────────────── 请求/响应模型 ────────────────────────────


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    uptime_seconds: float
    version: str
    components: Dict[str, str]


class AlertAckRequest(BaseModel):
    """确认告警请求"""
    user: str


class APIResponse(BaseModel):
    """通用 API 响应"""
    success: bool
    data: Optional[Dict] = None
    error: Optional[str] = None
    timestamp: float = 0

    def __init__(self, **kwargs):
        if "timestamp" not in kwargs or kwargs["timestamp"] == 0:
            kwargs["timestamp"] = time.time()
        super().__init__(**kwargs)


# ──────────────────────────── 健康检查 ────────────────────────────

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点

    返回服务运行状态、在线时长、各组件状态
    """
    return HealthResponse(
        status="healthy",
        uptime_seconds=time.time() - _start_time,
        version="1.0.0",
        components={
            "gpu_collector": "ok",
            "inference_collector": "ok",
            "anomaly_engine": "ok",
            "alert_manager": "ok",
            "capacity_planner": "ok",
            "cost_analyzer": "ok",
        },
    )


# ──────────────────────────── 指标端点 ────────────────────────────

@app.get("/api/v1/metrics")
async def get_metrics():
    """获取当前 GPU 和推理服务指标

    聚合所有采集器的最新数据
    """
    try:
        # GPU 指标
        gpu_metrics = gpu_collector.collect()
        gpu_data = [asdict(m) for m in gpu_metrics]

        # 推理指标 (使用 mock)
        inference_data = asdict(inference_collector.collect_mock())

        # 训练指标
        training_data = asdict(training_collector.collect_mock())

        # 网络指标
        network_metrics = network_collector.collect_mock()
        network_data = [asdict(m) for m in network_metrics]

        return APIResponse(
            success=True,
            data={
                "gpu": gpu_data,
                "inference": inference_data,
                "training": training_data,
                "network": network_data,
                "summary": {
                    "gpu_count": len(gpu_data),
                    "avg_gpu_utilization": (
                        sum(g["utilization_pct"] for g in gpu_data) / len(gpu_data)
                        if gpu_data else 0
                    ),
                    "avg_gpu_temperature": (
                        sum(g["temperature_c"] for g in gpu_data) / len(gpu_data)
                        if gpu_data else 0
                    ),
                    "inference_throughput_tps": inference_data.get("throughput_tps", 0),
                    "network_nodes": len(network_data),
                },
            },
        )
    except Exception as e:
        logger.error(f"获取指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"指标采集失败: {str(e)}")


# ──────────────────────────── 告警端点 ────────────────────────────

@app.get("/api/v1/alerts")
async def get_alerts(
    state: Optional[str] = Query(None, description="过滤状态: firing, acknowledged, resolved"),
    severity: Optional[str] = Query(None, description="过滤严重程度: info, warning, critical, page"),
    limit: int = Query(100, ge=1, le=500, description="返回数量上限"),
):
    """获取告警列表

    支持按状态和严重程度过滤
    """
    try:
        all_alerts = alert_manager.get_all_alerts(limit=limit)

        # 过滤
        if state:
            all_alerts = [a for a in all_alerts if a.state.value == state]
        if severity:
            all_alerts = [a for a in all_alerts if a.severity.value == severity]

        alerts_data = [asdict(a) for a in all_alerts]
        # Enum 值序列化
        for ad in alerts_data:
            ad["severity"] = ad["severity"].value if hasattr(ad["severity"], "value") else ad["severity"]
            ad["state"] = ad["state"].value if hasattr(ad["state"], "value") else ad["state"]

        stats = alert_manager.get_stats()

        return APIResponse(
            success=True,
            data={
                "alerts": alerts_data,
                "count": len(alerts_data),
                "stats": stats,
            },
        )
    except Exception as e:
        logger.error(f"获取告警失败: {e}")
        raise HTTPException(status_code=500, detail=f"告警查询失败: {str(e)}")


@app.post("/api/v1/alerts/{fingerprint}/ack")
async def acknowledge_alert(fingerprint: str, request: AlertAckRequest):
    """确认告警

    Args:
        fingerprint: 告警指纹 (用于唯一标识)
        request: 包含确认用户信息
    """
    success = alert_manager.acknowledge(fingerprint, request.user)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"告警不存在或已 resolved: {fingerprint}",
        )

    return APIResponse(
        success=True,
        data={"fingerprint": fingerprint, "acknowledged_by": request.user},
    )


@app.post("/api/v1/alerts/{fingerprint}/resolve")
async def resolve_alert(fingerprint: str):
    """手动解除告警"""
    success = alert_manager.resolve(fingerprint)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"告警不存在: {fingerprint}",
        )

    return APIResponse(
        success=True,
        data={"fingerprint": fingerprint, "state": "resolved"},
    )


# ──────────────────────────── 容量规划 ────────────────────────────

@app.get("/api/v1/capacity")
async def get_capacity():
    """获取容量规划数据

    包含当前利用率、预测和扩容建议
    """
    try:
        utilization = capacity_planner.current_utilization()
        forecast = capacity_planner.forecast_capacity_breach("kv_cache_usage", 0.9)
        scaling = capacity_planner.scaling_recommendation(target_qps=50)

        return APIResponse(
            success=True,
            data={
                "current_utilization": utilization,
                "forecast": forecast,
                "scaling_recommendation": scaling,
            },
        )
    except Exception as e:
        logger.error(f"获取容量数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"容量分析失败: {str(e)}")


# ──────────────────────────── 成本分析 ────────────────────────────

@app.get("/api/v1/cost")
async def get_cost():
    """获取成本分析数据

    包含日成本、租户分摊、优化建议
    """
    try:
        daily = cost_analyzer.daily_cost_summary()
        tenants = cost_analyzer.tenant_breakdown()
        suggestions = cost_analyzer.optimization_suggestions()

        tenant_data = [asdict(t) for t in tenants]

        return APIResponse(
            success=True,
            data={
                "daily_summary": daily,
                "tenant_breakdown": tenant_data,
                "optimization_suggestions": suggestions,
            },
        )
    except Exception as e:
        logger.error(f"获取成本数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"成本分析失败: {str(e)}")


# ──────────────────────────── 异常检测 ────────────────────────────

@app.get("/api/v1/anomalies")
async def get_anomalies(
    limit: int = Query(50, ge=1, le=500, description="返回数量上限"),
):
    """获取异常检测结果"""
    try:
        anomalies = anomaly_engine.get_recent_anomalies(limit=limit)
        anomaly_data = [asdict(a) for a in anomalies]
        for ad in anomaly_data:
            ad["severity"] = ad["severity"].value if hasattr(ad["severity"], "value") else ad["severity"]

        return APIResponse(
            success=True,
            data={
                "anomalies": anomaly_data,
                "count": len(anomaly_data),
                "anomaly_rate_per_min": anomaly_engine.get_anomaly_rate(),
            },
        )
    except Exception as e:
        logger.error(f"获取异常数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"异常查询失败: {str(e)}")


# ──────────────────────────── 性能建议 ────────────────────────────

@app.get("/api/v1/performance")
async def get_performance():
    """获取性能优化建议

    基于当前指标生成优化建议
    """
    try:
        # 构建指标快照供 advisor 分析
        inference_metrics = inference_collector.collect_mock()
        metrics_snapshot = {
            "kv_cache_usage": inference_metrics.kv_cache_usage,
            "tpot_p99_ms": inference_metrics.tpot_p99_ms,
            "prefix_cache_hit_rate": inference_metrics.prefix_cache_hit_rate,
        }

        suggestions = performance_advisor.analyze(metrics_snapshot)

        return APIResponse(
            success=True,
            data={
                "suggestions": suggestions,
                "metrics_snapshot": metrics_snapshot,
            },
        )
    except Exception as e:
        logger.error(f"获取性能建议失败: {e}")
        raise HTTPException(status_code=500, detail=f"性能分析失败: {str(e)}")


# ──────────────────────────── 启动事件 ────────────────────────────

@app.on_event("startup")
async def startup_event():
    """服务启动时初始化"""
    logger.info("AI Observability Platform API 启动")
    logger.info(f"GPU Collector: mock 模式")
    logger.info(f"告警管理器: suppression={alert_manager.suppression_window}s")


@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时清理"""
    logger.info("AI Observability Platform API 关闭")
