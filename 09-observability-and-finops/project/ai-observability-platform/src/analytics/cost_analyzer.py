"""成本分析器 — GPU 推理成本归因与优化"""

import logging
from typing import Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CostBreakdown:
    tenant_id: str
    total_tokens: int
    gpu_hours: float
    cost_usd: float
    cost_per_1m_tokens: float


class CostAnalyzer:
    """GPU 推理成本分析"""

    def __init__(self, gpu_cost_per_hour: float = 4.0, gpus_per_instance: int = 8):
        self.gpu_cost_per_hour = gpu_cost_per_hour
        self.gpus_per_instance = gpus_per_instance

    def daily_cost_summary(self) -> Dict:
        return {
            "total_cost_usd": 768.0,
            "total_gpu_hours": 192,
            "total_tokens_generated": 130_000_000,
            "cost_per_1m_tokens": 5.9,
            "utilization_pct": 72,
            "waste_usd": 215.0,
            "waste_reason": "凌晨 0-8 点低利用率",
        }

    def tenant_breakdown(self) -> List[CostBreakdown]:
        return [
            CostBreakdown("team-search", 50_000_000, 80, 320, 6.4),
            CostBreakdown("team-chat", 45_000_000, 70, 280, 6.2),
            CostBreakdown("team-code", 25_000_000, 30, 120, 4.8),
            CostBreakdown("team-analytics", 10_000_000, 12, 48, 4.8),
        ]

    def optimization_suggestions(self) -> List[Dict]:
        return [
            {
                "type": "scheduled_scaling",
                "description": "凌晨 0-8 点缩容到 1 个实例",
                "potential_saving_monthly": 3840,
                "risk": "low",
            },
            {
                "type": "spot_instances",
                "description": "30% 负载使用 Spot Instance",
                "potential_saving_monthly": 4608,
                "risk": "medium",
            },
            {
                "type": "prefix_caching",
                "description": "开启 Prefix Caching (team-chat 有大量重复 system prompt)",
                "potential_saving_monthly": 1500,
                "risk": "low",
            },
        ]
