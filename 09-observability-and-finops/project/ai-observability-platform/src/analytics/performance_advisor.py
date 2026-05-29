"""性能优化建议器"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class PerformanceAdvisor:
    """基于指标数据给出性能优化建议"""

    def analyze(self, metrics: Dict) -> List[Dict]:
        suggestions = []

        kv_usage = metrics.get("kv_cache_usage", 0)
        if kv_usage > 0.85:
            suggestions.append({
                "priority": "high",
                "category": "kv_cache",
                "title": "KV Cache 使用率偏高",
                "current": f"{kv_usage*100:.0f}%",
                "target": "< 80%",
                "actions": [
                    "增加 gpu_memory_utilization (当前可能偏低)",
                    "减小 max_model_len (如果大多数请求不需要 32K)",
                    "开启 prefix caching",
                    "考虑水平扩容",
                ],
            })

        tpot = metrics.get("tpot_p99_ms", 0)
        if tpot > 60:
            suggestions.append({
                "priority": "medium",
                "category": "latency",
                "title": "TPOT P99 偏高",
                "current": f"{tpot:.0f}ms",
                "target": "< 50ms",
                "actions": [
                    "降低 max_num_seqs 减小 batch size",
                    "检查 NVLink 带宽是否正常",
                    "确认 GPU 未被 throttle",
                ],
            })

        prefix_hit = metrics.get("prefix_cache_hit_rate", -1)
        if 0 <= prefix_hit < 0.5 and metrics.get("prefix_caching_enabled", False):
            suggestions.append({
                "priority": "medium",
                "category": "efficiency",
                "title": "Prefix Cache 命中率低",
                "current": f"{prefix_hit*100:.0f}%",
                "target": "> 70%",
                "actions": [
                    "检查 system prompt 是否统一",
                    "增大 prefix cache 预留空间",
                    "确认 KV Cache 不会频繁淘汰 prefix blocks",
                ],
            })

        if not suggestions:
            suggestions.append({
                "priority": "info",
                "category": "general",
                "title": "系统运行正常",
                "actions": ["继续监控, 无需优化"],
            })

        return suggestions
