"""
Spot 中断处理器 — GPU 推理服务优雅中断
=======================================

处理 Spot Instance 回收事件:
1. 监听中断通知 (AWS: 2分钟, GCP: 30秒)
2. 停止接收新请求
3. 迁移正在处理的请求到健康实例
4. 优雅关闭

依赖: asyncio, aiohttp
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class InterruptionSource(Enum):
    AWS_SPOT = "aws_spot"          # AWS: 2 分钟通知
    GCP_PREEMPTIBLE = "gcp_preemptible"  # GCP: 30 秒通知
    AZURE_SPOT = "azure_spot"      # Azure: 30 秒通知
    MANUAL = "manual"              # 手动触发 (维护)


@dataclass
class ActiveRequest:
    """正在处理的请求"""
    request_id: str
    client_id: str
    prompt_tokens: int
    generated_tokens: int         # 已生成的 token 数
    generated_text: str = ""      # 已生成的文本
    start_time: float = 0
    estimated_remaining_s: float = 0
    can_migrate: bool = True      # 是否可以迁移


@dataclass
class MigrationResult:
    """迁移结果"""
    request_id: str
    success: bool
    target_instance: str = ""
    migration_time_ms: float = 0
    tokens_regenerated: int = 0
    error: str = ""


class SpotInterruptionHandler:
    """Spot 中断处理器

    处理流程 (AWS 2 分钟窗口):
    ┌─────────────────────────────────────────────────────┐
    │ T=0s:    收到中断通知                                │
    │ T=0-5s:  从 LB 摘除, 停止接收新请求                  │
    │ T=5-30s: 等待短请求完成 (< 20s 的)                   │
    │ T=30-90s: 迁移长请求到其他实例                        │
    │ T=90-110s: 清理资源, 保存状态                        │
    │ T=110-120s: 确认 shutdown                            │
    └─────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        instance_id: str,
        healthy_instances: List[str] = None,
        short_request_threshold_s: float = 20.0,
        migration_timeout_s: float = 60.0,
    ):
        self.instance_id = instance_id
        self.healthy_instances = healthy_instances or []
        self.short_threshold = short_request_threshold_s
        self.migration_timeout = migration_timeout_s

        self._active_requests: Dict[str, ActiveRequest] = {}
        self._is_draining = False
        self._interruption_time: Optional[float] = None
        self._on_drain_callbacks: List[Callable] = []
        self._migration_results: List[MigrationResult] = []

    def register_request(self, request: ActiveRequest):
        """注册活跃请求"""
        self._active_requests[request.request_id] = request

    def unregister_request(self, request_id: str):
        """请求完成, 取消注册"""
        self._active_requests.pop(request_id, None)

    def on_drain(self, callback: Callable):
        """注册 drain 回调"""
        self._on_drain_callbacks.append(callback)

    @property
    def is_draining(self) -> bool:
        return self._is_draining

    async def handle_interruption(self, source: InterruptionSource = InterruptionSource.AWS_SPOT):
        """处理中断事件 (主入口)"""
        self._interruption_time = time.time()
        self._is_draining = True

        # 根据来源确定可用时间
        available_time_s = {
            InterruptionSource.AWS_SPOT: 120,
            InterruptionSource.GCP_PREEMPTIBLE: 30,
            InterruptionSource.AZURE_SPOT: 30,
            InterruptionSource.MANUAL: 300,
        }.get(source, 120)

        logger.warning(
            f"SPOT INTERRUPTION received! Instance={self.instance_id}, "
            f"source={source.value}, available_time={available_time_s}s, "
            f"active_requests={len(self._active_requests)}"
        )

        # Phase 1: 停止接收新请求
        await self._phase_drain_start()

        # Phase 2: 等待短请求完成
        wait_budget = min(available_time_s * 0.25, 30)
        await self._phase_wait_short_requests(timeout_s=wait_budget)

        # Phase 3: 迁移长请求
        migrate_budget = available_time_s * 0.5
        await self._phase_migrate_requests(timeout_s=migrate_budget)

        # Phase 4: 清理
        await self._phase_cleanup()

        elapsed = time.time() - self._interruption_time
        logger.info(
            f"Interruption handling complete in {elapsed:.1f}s. "
            f"Remaining requests: {len(self._active_requests)}, "
            f"Migrations: {len(self._migration_results)}"
        )

        return self._generate_report()

    async def _phase_drain_start(self):
        """Phase 1: 开始 Drain"""
        logger.info("Phase 1: Removing from load balancer...")
        # 通知 LB 摘除 (实际实现: 调用 K8s API 设置 Pod 为 NotReady)
        for callback in self._on_drain_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Drain callback failed: {e}")

        await asyncio.sleep(2)  # 等待 LB 生效
        logger.info("Phase 1 complete: No new requests will arrive")

    async def _phase_wait_short_requests(self, timeout_s: float):
        """Phase 2: 等待短请求自然完成"""
        logger.info(f"Phase 2: Waiting for short requests (timeout={timeout_s}s)...")
        start = time.time()

        while time.time() - start < timeout_s and self._active_requests:
            # 检查是否所有剩余请求都是长请求
            short_requests = [
                r for r in self._active_requests.values()
                if r.estimated_remaining_s < self.short_threshold
            ]
            if not short_requests:
                break
            await asyncio.sleep(1)

        completed_in_wait = len([
            r for r in self._active_requests.values()
            if r.estimated_remaining_s < 0
        ])
        logger.info(f"Phase 2 complete: {completed_in_wait} requests completed naturally")

    async def _phase_migrate_requests(self, timeout_s: float):
        """Phase 3: 迁移剩余请求"""
        remaining = list(self._active_requests.values())
        if not remaining:
            logger.info("Phase 3: No requests to migrate")
            return

        logger.info(f"Phase 3: Migrating {len(remaining)} requests...")

        # 按已生成 token 数排序 (优先迁移进度最多的)
        remaining.sort(key=lambda r: r.generated_tokens, reverse=True)

        for request in remaining:
            if not request.can_migrate:
                logger.warning(f"Request {request.request_id} cannot be migrated")
                continue

            if not self.healthy_instances:
                logger.error("No healthy instances available for migration!")
                break

            # 选择目标实例 (round-robin 简化)
            target = self.healthy_instances[
                hash(request.request_id) % len(self.healthy_instances)
            ]

            result = await self._migrate_single_request(request, target)
            self._migration_results.append(result)

            if result.success:
                self._active_requests.pop(request.request_id, None)

    async def _migrate_single_request(
        self, request: ActiveRequest, target_instance: str
    ) -> MigrationResult:
        """迁移单个请求

        迁移策略:
        - 发送 prompt + 已生成 text 到目标实例
        - 目标实例从已生成部分继续 (避免重复 prefill)
        - 客户端 streaming 短暂中断后恢复
        """
        start = time.time()

        try:
            # 构造迁移 payload
            migration_payload = {
                "request_id": request.request_id,
                "client_id": request.client_id,
                "prompt_tokens": request.prompt_tokens,
                "generated_text": request.generated_text,
                "generated_tokens": request.generated_tokens,
                "continue_from_token": request.generated_tokens,
            }

            # 模拟发送到目标实例 (实际: HTTP/gRPC call)
            # response = await self._send_migration(target_instance, migration_payload)
            await asyncio.sleep(0.1)  # 模拟网络延迟

            migration_time = (time.time() - start) * 1000

            logger.info(
                f"Migrated {request.request_id} to {target_instance}: "
                f"{request.generated_tokens} tokens preserved, "
                f"migration_time={migration_time:.0f}ms"
            )

            return MigrationResult(
                request_id=request.request_id,
                success=True,
                target_instance=target_instance,
                migration_time_ms=migration_time,
                tokens_regenerated=0,  # 理想情况不需要重新生成
            )

        except Exception as e:
            return MigrationResult(
                request_id=request.request_id,
                success=False,
                error=str(e),
            )

    async def _phase_cleanup(self):
        """Phase 4: 清理资源"""
        logger.info("Phase 4: Cleanup...")
        # 释放 GPU 显存
        # 关闭网络连接
        # 保存日志
        await asyncio.sleep(1)
        logger.info("Phase 4 complete: Resources released")

    def _generate_report(self) -> Dict:
        """生成中断处理报告"""
        total_time = time.time() - self._interruption_time if self._interruption_time else 0
        successful = sum(1 for r in self._migration_results if r.success)
        failed = sum(1 for r in self._migration_results if not r.success)

        return {
            "instance_id": self.instance_id,
            "total_handling_time_s": round(total_time, 1),
            "requests_at_interruption": len(self._migration_results) + len(self._active_requests),
            "migrations_attempted": len(self._migration_results),
            "migrations_successful": successful,
            "migrations_failed": failed,
            "requests_abandoned": len(self._active_requests),
            "avg_migration_time_ms": round(
                np.mean([r.migration_time_ms for r in self._migration_results if r.success]) if successful else 0, 1
            ) if self._migration_results else 0,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    async def demo():
        handler = SpotInterruptionHandler(
            instance_id="vllm-spot-0",
            healthy_instances=["vllm-ondemand-0", "vllm-ondemand-1"],
        )

        # 注册一些活跃请求
        for i in range(5):
            handler.register_request(ActiveRequest(
                request_id=f"req-{i:03d}",
                client_id=f"client-{i % 3}",
                prompt_tokens=1024 + i * 500,
                generated_tokens=50 + i * 30,
                generated_text=f"Generated text for request {i}...",
                estimated_remaining_s=5 + i * 10,
            ))

        # 模拟中断
        report = await handler.handle_interruption(InterruptionSource.AWS_SPOT)
        print("\n=== Interruption Report ===")
        for k, v in report.items():
            print(f"  {k}: {v}")

    asyncio.run(demo())
