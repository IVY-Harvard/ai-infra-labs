"""
Lab 10: 灰度发布系统
实现基于流量比例的灰度发布 + 自动决策
"""
import time
import random
import hashlib
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import deque


class DeployStage(Enum):
    SHADOW = "shadow"
    CANARY_1 = "canary_1%"
    CANARY_5 = "canary_5%"
    CANARY_25 = "canary_25%"
    CANARY_50 = "canary_50%"
    FULL = "full_100%"
    ROLLED_BACK = "rolled_back"


STAGE_TRAFFIC = {
    DeployStage.SHADOW: 0.0,
    DeployStage.CANARY_1: 0.01,
    DeployStage.CANARY_5: 0.05,
    DeployStage.CANARY_25: 0.25,
    DeployStage.CANARY_50: 0.50,
    DeployStage.FULL: 1.0,
}

STAGE_ORDER = [
    DeployStage.SHADOW, DeployStage.CANARY_1, DeployStage.CANARY_5,
    DeployStage.CANARY_25, DeployStage.CANARY_50, DeployStage.FULL,
]


@dataclass
class DeploymentMetrics:
    quality_scores: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    errors: int = 0
    total_requests: int = 0

    @property
    def avg_quality(self):
        return np.mean(self.quality_scores) if self.quality_scores else 0

    @property
    def p95_latency(self):
        return np.percentile(self.latencies, 95) if self.latencies else 0

    @property
    def error_rate(self):
        return self.errors / self.total_requests if self.total_requests else 0


class CanaryDeployment:
    """灰度发布系统"""

    def __init__(self, new_version: str, old_version: str = "stable"):
        self.new_version = new_version
        self.old_version = old_version
        self.current_stage = DeployStage.SHADOW
        self.metrics = {
            "new": DeploymentMetrics(),
            "old": DeploymentMetrics(),
        }
        self.thresholds = {
            "min_quality": 0.80,
            "max_p95_latency": 3000,
            "max_error_rate": 0.05,
            "min_requests": 50,
        }
        self.history = []

    def route_request(self, user_id: str) -> str:
        """路由请求到新/旧版本"""
        traffic_ratio = STAGE_TRAFFIC[self.current_stage]
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return self.new_version if hash_val < traffic_ratio else self.old_version

    def record_metric(self, version: str, quality: float,
                      latency: float, is_error: bool = False):
        """记录请求指标"""
        key = "new" if version == self.new_version else "old"
        m = self.metrics[key]
        m.total_requests += 1
        m.quality_scores.append(quality)
        m.latencies.append(latency)
        if is_error:
            m.errors += 1

    def evaluate_stage(self) -> str:
        """评估当前阶段是否可以推进"""
        new_m = self.metrics["new"]
        old_m = self.metrics["old"]

        if new_m.total_requests < self.thresholds["min_requests"]:
            return "hold"  # 样本不足

        checks = {
            "quality": new_m.avg_quality >= self.thresholds["min_quality"],
            "latency": new_m.p95_latency <= self.thresholds["max_p95_latency"],
            "error_rate": new_m.error_rate <= self.thresholds["max_error_rate"],
        }

        # 与旧版本对比
        if old_m.total_requests > 0:
            checks["quality_vs_old"] = new_m.avg_quality >= old_m.avg_quality * 0.95

        if all(checks.values()):
            return "promote"
        elif any(not v for k, v in checks.items() if k in ["error_rate", "quality"]):
            # 关键指标不达标
            if new_m.error_rate > self.thresholds["max_error_rate"] * 2:
                return "rollback"  # 严重问题立即回滚
        return "hold"

    def advance(self) -> DeployStage:
        """推进到下一阶段"""
        current_idx = STAGE_ORDER.index(self.current_stage)
        if current_idx < len(STAGE_ORDER) - 1:
            self.current_stage = STAGE_ORDER[current_idx + 1]
            self._reset_metrics()
            self.history.append({
                "action": "advance",
                "stage": self.current_stage.value,
                "timestamp": datetime.now().isoformat(),
            })
        return self.current_stage

    def rollback(self, reason: str):
        """回滚"""
        self.current_stage = DeployStage.ROLLED_BACK
        self.history.append({
            "action": "rollback",
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        })
        print(f"!!! 回滚: {reason}")

    def _reset_metrics(self):
        self.metrics = {
            "new": DeploymentMetrics(),
            "old": DeploymentMetrics(),
        }


def simulate_deployment():
    """模拟灰度发布流程"""
    print("=" * 60)
    print("灰度发布模拟")
    print("=" * 60)

    deploy = CanaryDeployment("rag-v2.0", "rag-v1.0")

    for stage_idx in range(len(STAGE_ORDER)):
        print(f"\n--- 阶段: {deploy.current_stage.value} ---")
        traffic = STAGE_TRAFFIC[deploy.current_stage]
        print(f"  新版本流量: {traffic*100:.0f}%")

        # 模拟请求
        for i in range(100):
            user_id = f"user_{random.randint(0, 10000)}"
            version = deploy.route_request(user_id)

            # 模拟指标（新版本略优于旧版本）
            if version == "rag-v2.0":
                quality = 0.88 + random.gauss(0, 0.05)
                latency = 1200 + random.gauss(0, 200)
                is_error = random.random() < 0.02
            else:
                quality = 0.82 + random.gauss(0, 0.05)
                latency = 1000 + random.gauss(0, 150)
                is_error = random.random() < 0.03

            deploy.record_metric(version, quality, latency, is_error)

        # 评估
        decision = deploy.evaluate_stage()
        new_m = deploy.metrics["new"]
        old_m = deploy.metrics["old"]

        print(f"  新版本: quality={new_m.avg_quality:.3f}, "
              f"p95={new_m.p95_latency:.0f}ms, errors={new_m.error_rate:.2%}")
        print(f"  旧版本: quality={old_m.avg_quality:.3f}, "
              f"p95={old_m.p95_latency:.0f}ms, errors={old_m.error_rate:.2%}")
        print(f"  决策: {decision}")

        if decision == "promote":
            if deploy.current_stage == DeployStage.FULL:
                print("\n  发布完成！")
                break
            deploy.advance()
        elif decision == "rollback":
            deploy.rollback("指标不达标")
            break

    # 发布历史
    print(f"\n发布历史:")
    for event in deploy.history:
        print(f"  [{event['action']}] {event.get('stage', event.get('reason', ''))}")


if __name__ == "__main__":
    simulate_deployment()
