"""
ChatOps 机器人 — 自然语言运维交互
==================================

通过 Slack/企业微信与 GPU 推理监控系统交互:
- "集群状态如何?" → 查询 Prometheus 返回摘要
- "为什么 TTFT 升高了?" → 运行根因分析
- "扩容到 4 个实例" → 执行扩容 (需确认)
- "最近有什么告警?" → 查询 Alertmanager

依赖: asyncio, aiohttp (实际场景)
"""

import re
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """意图类型"""
    QUERY_STATUS = "query_status"           # 查询状态
    QUERY_METRICS = "query_metrics"         # 查询指标
    QUERY_ALERTS = "query_alerts"           # 查询告警
    DIAGNOSE = "diagnose"                   # 诊断问题
    ACTION_SCALE = "action_scale"           # 扩缩容
    ACTION_RESTART = "action_restart"       # 重启服务
    ACTION_RATE_LIMIT = "action_rate_limit" # 限流
    HELP = "help"                           # 帮助
    UNKNOWN = "unknown"


@dataclass
class ChatMessage:
    """聊天消息"""
    user_id: str
    text: str
    channel: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class ChatResponse:
    """回复消息"""
    text: str
    blocks: List[Dict] = field(default_factory=list)  # Slack Block Kit
    needs_confirmation: bool = False
    action_payload: Optional[Dict] = None


class IntentClassifier:
    """意图分类器 (基于关键词规则, 生产环境可用 LLM)"""

    PATTERNS = {
        IntentType.QUERY_STATUS: [
            r"(状态|status|怎么样|情况|概览|overview|健康|health)",
        ],
        IntentType.QUERY_METRICS: [
            r"(ttft|tpot|吞吐|throughput|延迟|latency|kv.?cache|缓存|利用率|utilization)",
        ],
        IntentType.QUERY_ALERTS: [
            r"(告警|alert|报警|异常|问题|故障|incident)",
        ],
        IntentType.DIAGNOSE: [
            r"(为什么|why|原因|根因|root.?cause|诊断|分析|怎么回事)",
        ],
        IntentType.ACTION_SCALE: [
            r"(扩容|缩容|scale|扩展|增加.*实例|减少.*实例|replica)",
        ],
        IntentType.ACTION_RESTART: [
            r"(重启|restart|重新启动|reboot|kill)",
        ],
        IntentType.ACTION_RATE_LIMIT: [
            r"(限流|rate.?limit|限速|throttle|流控)",
        ],
        IntentType.HELP: [
            r"(帮助|help|命令|command|怎么用|usage)",
        ],
    }

    def classify(self, text: str) -> Tuple[IntentType, float]:
        """分类用户意图

        Returns:
            (意图类型, 置信度)
        """
        text_lower = text.lower()

        for intent, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return intent, 0.9

        return IntentType.UNKNOWN, 0.1

    def extract_parameters(self, text: str, intent: IntentType) -> Dict:
        """从文本中提取参数"""
        params = {}

        # 提取数字
        numbers = re.findall(r'\d+', text)
        if numbers:
            params["numbers"] = [int(n) for n in numbers]

        # 提取实例名
        instance_match = re.search(r'(vllm-\w+|instance-\w+)', text)
        if instance_match:
            params["instance"] = instance_match.group(1)

        # 提取时间范围
        time_match = re.search(r'(\d+)\s*(分钟|min|小时|hour|h)', text)
        if time_match:
            value = int(time_match.group(1))
            unit = time_match.group(2)
            if unit in ("小时", "hour", "h"):
                value *= 60
            params["time_range_min"] = value

        return params


class GPUInferenceChatBot:
    """GPU 推理 ChatOps 机器人

    命令示例:
    - "集群状态" → 返回总体健康状况
    - "TTFT 是多少" → 查询当前 TTFT P50/P99
    - "为什么延迟升高了" → 运行诊断
    - "扩容到 4 个实例" → 执行扩容 (需确认)
    - "最近 30 分钟的告警" → 查询告警历史
    """

    def __init__(self):
        self.classifier = IntentClassifier()
        self._confirmation_pending: Dict[str, Dict] = {}  # user_id → pending action

        # 模拟的系统状态
        self._mock_metrics = {
            "ttft_p50_ms": 250,
            "ttft_p99_ms": 1200,
            "tpot_p50_ms": 22,
            "tpot_p99_ms": 45,
            "throughput_tps": 1500,
            "kv_cache_usage": 0.72,
            "gpu_temp_avg": 68,
            "replicas": 2,
            "queue_length": 3,
            "error_rate": 0.001,
            "active_requests": 45,
        }

    async def handle_message(self, message: ChatMessage) -> ChatResponse:
        """处理用户消息"""
        text = message.text.strip()

        # 检查是否在确认动作
        if message.user_id in self._confirmation_pending:
            return self._handle_confirmation(message)

        # 意图分类
        intent, confidence = self.classifier.classify(text)
        params = self.classifier.extract_parameters(text, intent)

        logger.info(f"Intent: {intent.value} (conf={confidence:.2f}), params={params}")

        # 路由到处理器
        handlers = {
            IntentType.QUERY_STATUS: self._handle_status,
            IntentType.QUERY_METRICS: self._handle_metrics,
            IntentType.QUERY_ALERTS: self._handle_alerts,
            IntentType.DIAGNOSE: self._handle_diagnose,
            IntentType.ACTION_SCALE: self._handle_scale,
            IntentType.ACTION_RESTART: self._handle_restart,
            IntentType.ACTION_RATE_LIMIT: self._handle_rate_limit,
            IntentType.HELP: self._handle_help,
        }

        handler = handlers.get(intent, self._handle_unknown)
        return await handler(message, params)

    async def _handle_status(self, msg: ChatMessage, params: Dict) -> ChatResponse:
        m = self._mock_metrics
        health = "HEALTHY" if m["kv_cache_usage"] < 0.9 and m["error_rate"] < 0.01 else "DEGRADED"

        text = f"""*GPU 推理集群状态* ({health})
:gpu: 实例: {m['replicas']} 个运行中
:chart_with_upwards_trend: 吞吐: {m['throughput_tps']} tokens/s
:hourglass: TTFT P99: {m['ttft_p99_ms']}ms | TPOT P99: {m['tpot_p99_ms']}ms
:package: KV Cache: {m['kv_cache_usage']*100:.0f}%
:thermometer: GPU 温度: {m['gpu_temp_avg']}°C (avg)
:inbox_tray: 排队: {m['queue_length']} | 活跃: {m['active_requests']}
:x: 错误率: {m['error_rate']*100:.2f}%"""

        return ChatResponse(text=text)

    async def _handle_metrics(self, msg: ChatMessage, params: Dict) -> ChatResponse:
        m = self._mock_metrics
        text_lower = msg.text.lower()

        if "ttft" in text_lower:
            return ChatResponse(
                text=f"*TTFT*: P50={m['ttft_p50_ms']}ms, P99={m['ttft_p99_ms']}ms\n"
                     f"SLO: P99 < 5000ms {'OK' if m['ttft_p99_ms'] < 5000 else 'BREACH'}"
            )
        elif "tpot" in text_lower:
            return ChatResponse(
                text=f"*TPOT*: P50={m['tpot_p50_ms']}ms, P99={m['tpot_p99_ms']}ms\n"
                     f"SLO: P99 < 80ms {'OK' if m['tpot_p99_ms'] < 80 else 'BREACH'}"
            )
        elif "kv" in text_lower or "cache" in text_lower or "缓存" in text_lower:
            return ChatResponse(
                text=f"*KV Cache*: {m['kv_cache_usage']*100:.1f}%\n"
                     f"状态: {'正常' if m['kv_cache_usage'] < 0.8 else '注意' if m['kv_cache_usage'] < 0.9 else '告警!'}"
            )
        else:
            return ChatResponse(
                text=f"*当前指标*:\n"
                     f"吞吐: {m['throughput_tps']} tps\n"
                     f"TTFT: P50={m['ttft_p50_ms']}ms P99={m['ttft_p99_ms']}ms\n"
                     f"TPOT: P50={m['tpot_p50_ms']}ms P99={m['tpot_p99_ms']}ms\n"
                     f"KV Cache: {m['kv_cache_usage']*100:.0f}%"
            )

    async def _handle_alerts(self, msg: ChatMessage, params: Dict) -> ChatResponse:
        return ChatResponse(
            text="*最近告警*:\n"
                 "- [已恢复] KVCacheUsageHigh @ vllm-0 (30分钟前)\n"
                 "- [活跃] GPUTemperatureHigh @ gpu-node-2 (温度 81°C)\n"
                 "无 Critical 级别告警"
        )

    async def _handle_diagnose(self, msg: ChatMessage, params: Dict) -> ChatResponse:
        m = self._mock_metrics
        analysis = []

        if m["ttft_p99_ms"] > 2000:
            if m["kv_cache_usage"] > 0.9:
                analysis.append("TTFT 升高原因: KV Cache 使用率 > 90%, Preemption 频繁")
                analysis.append("建议: 扩容或限制最大 prompt 长度")
            elif m["queue_length"] > 10:
                analysis.append("TTFT 升高原因: 排队过长")
                analysis.append("建议: 增加实例数")
            else:
                analysis.append("TTFT 在正常范围内 (P99 < 2s)")

        if m["tpot_p99_ms"] > 60:
            analysis.append("TPOT 偏高: 可能是 batch size 过大或 GPU 限频")

        if not analysis:
            analysis.append("系统各项指标正常, 未发现异常")

        return ChatResponse(text="*诊断分析*:\n" + "\n".join(f"- {a}" for a in analysis))

    async def _handle_scale(self, msg: ChatMessage, params: Dict) -> ChatResponse:
        target = params.get("numbers", [None])[0] if params.get("numbers") else None
        current = self._mock_metrics["replicas"]

        if target is None:
            return ChatResponse(text=f"当前 {current} 个实例. 请指定目标数量, 如 '扩容到 4 个实例'")

        self._confirmation_pending[msg.user_id] = {
            "action": "scale",
            "target_replicas": target,
            "current_replicas": current,
        }

        return ChatResponse(
            text=f"确认: 将实例从 {current} 个调整到 {target} 个?\n"
                 f"影响: {'新增' if target > current else '减少'} "
                 f"{abs(target - current)} 个实例 "
                 f"({abs(target - current) * 8} GPUs)\n"
                 f"回复 `确认` 执行或 `取消` 放弃",
            needs_confirmation=True,
        )

    async def _handle_restart(self, msg: ChatMessage, params: Dict) -> ChatResponse:
        instance = params.get("instance", "")
        if not instance:
            return ChatResponse(text="请指定要重启的实例, 如 '重启 vllm-0'")

        self._confirmation_pending[msg.user_id] = {
            "action": "restart",
            "instance": instance,
        }

        return ChatResponse(
            text=f"确认: 重启实例 {instance}?\n"
                 f"影响: 该实例上的请求将被迁移, 预计 3-5 分钟恢复\n"
                 f"回复 `确认` 执行或 `取消` 放弃",
            needs_confirmation=True,
        )

    async def _handle_rate_limit(self, msg: ChatMessage, params: Dict) -> ChatResponse:
        return ChatResponse(
            text="*限流操作*:\n"
                 "当前限流状态: 未启用\n"
                 "可用选项:\n"
                 "- `限流 50%` — 拒绝 50% 新请求\n"
                 "- `限流 长请求` — 拒绝 prompt > 4K tokens 的请求\n"
                 "- `取消限流` — 恢复正常"
        )

    async def _handle_help(self, msg: ChatMessage, params: Dict) -> ChatResponse:
        return ChatResponse(
            text="*GPU 推理 ChatOps 命令*:\n"
                 ":mag: *查询*\n"
                 "  `集群状态` — 总体概览\n"
                 "  `TTFT 是多少` — 查询延迟\n"
                 "  `KV Cache 使用率` — 查询缓存\n"
                 "  `最近告警` — 查看告警\n"
                 ":wrench: *诊断*\n"
                 "  `为什么延迟升高了` — 自动诊断\n"
                 ":rocket: *操作*\n"
                 "  `扩容到 N 个实例` — 扩缩容\n"
                 "  `重启 vllm-0` — 重启实例\n"
                 "  `限流 50%` — 启用限流"
        )

    async def _handle_unknown(self, msg: ChatMessage, params: Dict) -> ChatResponse:
        return ChatResponse(
            text=f"不太理解 '{msg.text}'. 输入 `帮助` 查看支持的命令."
        )

    def _handle_confirmation(self, msg: ChatMessage) -> ChatResponse:
        pending = self._confirmation_pending.pop(msg.user_id, None)
        if not pending:
            return ChatResponse(text="没有待确认的操作")

        if msg.text.strip() in ("确认", "confirm", "yes", "y"):
            action = pending["action"]
            if action == "scale":
                return ChatResponse(
                    text=f"已提交扩缩容请求: {pending['current_replicas']} → {pending['target_replicas']} 实例\n"
                         f"预计 3-5 分钟完成 (模型加载)"
                )
            elif action == "restart":
                return ChatResponse(
                    text=f"已提交重启: {pending['instance']}\n正在执行滚动重启..."
                )
        else:
            return ChatResponse(text="操作已取消")


if __name__ == "__main__":
    import asyncio

    async def demo():
        bot = GPUInferenceChatBot()

        messages = [
            "集群状态怎么样",
            "TTFT 是多少",
            "KV Cache 使用率",
            "最近有什么告警",
            "为什么延迟升高了",
            "扩容到 4 个实例",
            "确认",
            "帮助",
        ]

        print("=== ChatOps Bot Demo ===\n")
        for text in messages:
            msg = ChatMessage(user_id="user-001", text=text)
            response = await bot.handle_message(msg)
            print(f"User: {text}")
            print(f"Bot: {response.text}\n")

    asyncio.run(demo())
