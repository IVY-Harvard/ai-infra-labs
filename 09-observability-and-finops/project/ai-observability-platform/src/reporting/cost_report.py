"""GPU 成本报告生成器 — 成本归因、分摊与浪费识别

职责:
1. 按团队/模型的 GPU 成本分摊 (Chargeback)
2. Spot vs On-Demand 成本对比
3. 闲置资源浪费识别
4. 生成成本明细表

成本模型:
┌────────────────┬──────────┬───────────┐
│ 实例类型        │ 单价/h   │ 折扣率     │
├────────────────┼──────────┼───────────┤
│ On-Demand A100 │ $4.00    │ -         │
│ Spot A100      │ $1.60    │ 60%       │
│ Reserved A100  │ $2.40    │ 40%       │
└────────────────┴──────────┴───────────┘
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GPUInstanceCost:
    """GPU 实例成本配置"""
    instance_type: str  # on_demand, spot, reserved
    gpu_model: str  # A100, H100 等
    cost_per_hour: float
    discount_pct: float = 0  # 相对 on-demand 的折扣


@dataclass
class TeamCostRecord:
    """团队成本记录"""
    team_id: str
    team_name: str
    gpu_hours: float = 0
    tokens_generated: int = 0
    cost_usd: float = 0
    cost_per_1m_tokens: float = 0
    models_used: List[str] = field(default_factory=list)
    instance_types: Dict[str, float] = field(default_factory=dict)  # type → hours


@dataclass
class WasteRecord:
    """资源浪费记录"""
    category: str  # idle_gpu, over_provisioned, unused_reservation
    description: str
    gpu_hours_wasted: float = 0
    cost_wasted_usd: float = 0
    time_range: str = ""
    recommendation: str = ""


@dataclass
class CostReportData:
    """成本报告数据结构"""
    report_date: str = ""
    generated_at: float = 0
    period: str = "daily"  # daily, weekly, monthly

    # 总成本
    total_cost_usd: float = 0
    total_gpu_hours: float = 0
    total_tokens: int = 0

    # 实例类型成本分布
    cost_by_instance_type: Dict[str, float] = field(default_factory=dict)
    hours_by_instance_type: Dict[str, float] = field(default_factory=dict)

    # 团队分摊
    team_costs: List[TeamCostRecord] = field(default_factory=list)

    # Spot vs On-Demand 对比
    spot_savings_usd: float = 0
    spot_ratio: float = 0  # Spot 占总量的比例

    # 浪费识别
    waste_records: List[WasteRecord] = field(default_factory=list)
    total_waste_usd: float = 0

    # 优化潜力
    potential_monthly_savings: float = 0


class CostReportGenerator:
    """GPU 成本报告生成器"""

    # 默认实例成本配置
    DEFAULT_INSTANCE_COSTS = [
        GPUInstanceCost("on_demand", "A100-80GB", 4.00, 0),
        GPUInstanceCost("spot", "A100-80GB", 1.60, 60),
        GPUInstanceCost("reserved_1yr", "A100-80GB", 2.40, 40),
        GPUInstanceCost("on_demand", "H100-80GB", 5.50, 0),
        GPUInstanceCost("spot", "H100-80GB", 2.20, 60),
        GPUInstanceCost("reserved_1yr", "H100-80GB", 3.30, 40),
    ]

    def __init__(self, instance_costs: Optional[List[GPUInstanceCost]] = None):
        self.instance_costs = instance_costs or list(self.DEFAULT_INSTANCE_COSTS)
        self._cost_lookup: Dict[str, float] = {}
        for ic in self.instance_costs:
            key = f"{ic.instance_type}:{ic.gpu_model}"
            self._cost_lookup[key] = ic.cost_per_hour

    def get_cost_per_hour(self, instance_type: str, gpu_model: str = "A100-80GB") -> float:
        """获取指定实例类型的单价"""
        key = f"{instance_type}:{gpu_model}"
        return self._cost_lookup.get(key, 4.0)  # 默认 on-demand 价格

    def generate(self) -> CostReportData:
        """生成成本报告

        使用模拟数据演示成本归因和浪费识别逻辑
        """
        report = CostReportData(
            report_date=time.strftime("%Y-%m-%d"),
            generated_at=time.time(),
        )

        # 模拟团队使用数据
        self._simulate_team_costs(report)

        # 按实例类型汇总
        self._aggregate_by_instance_type(report)

        # Spot 节省计算
        self._calculate_spot_savings(report)

        # 浪费识别
        self._identify_waste(report)

        # 汇总
        self._calculate_totals(report)

        logger.info(
            f"成本报告已生成: date={report.report_date}, "
            f"total=${report.total_cost_usd:.2f}, "
            f"waste=${report.total_waste_usd:.2f}"
        )

        return report

    def _simulate_team_costs(self, report: CostReportData):
        """模拟各团队的 GPU 使用和成本"""
        team_data = [
            {
                "team_id": "team-search",
                "team_name": "搜索团队",
                "gpu_hours": 80,
                "tokens": 50_000_000,
                "models": ["qwen-72b-chat"],
                "instances": {"on_demand": 48, "spot": 32},
            },
            {
                "team_id": "team-chat",
                "team_name": "对话团队",
                "gpu_hours": 70,
                "tokens": 45_000_000,
                "models": ["qwen-72b-chat", "qwen-14b-chat"],
                "instances": {"on_demand": 40, "spot": 30},
            },
            {
                "team_id": "team-code",
                "team_name": "代码助手团队",
                "gpu_hours": 30,
                "tokens": 25_000_000,
                "models": ["deepseek-coder-33b"],
                "instances": {"on_demand": 20, "spot": 10},
            },
            {
                "team_id": "team-analytics",
                "team_name": "数据分析团队",
                "gpu_hours": 12,
                "tokens": 10_000_000,
                "models": ["qwen-14b-chat"],
                "instances": {"on_demand": 8, "spot": 4},
            },
        ]

        for td in team_data:
            # 计算成本
            cost = 0
            for inst_type, hours in td["instances"].items():
                rate = self.get_cost_per_hour(inst_type)
                cost += hours * rate

            cost_per_1m = (cost / td["tokens"] * 1_000_000) if td["tokens"] > 0 else 0

            record = TeamCostRecord(
                team_id=td["team_id"],
                team_name=td["team_name"],
                gpu_hours=td["gpu_hours"],
                tokens_generated=td["tokens"],
                cost_usd=round(cost, 2),
                cost_per_1m_tokens=round(cost_per_1m, 2),
                models_used=td["models"],
                instance_types=td["instances"],
            )
            report.team_costs.append(record)

    def _aggregate_by_instance_type(self, report: CostReportData):
        """按实例类型汇总成本"""
        cost_by_type: Dict[str, float] = {}
        hours_by_type: Dict[str, float] = {}

        for team in report.team_costs:
            for inst_type, hours in team.instance_types.items():
                rate = self.get_cost_per_hour(inst_type)
                cost_by_type[inst_type] = cost_by_type.get(inst_type, 0) + hours * rate
                hours_by_type[inst_type] = hours_by_type.get(inst_type, 0) + hours

        report.cost_by_instance_type = {k: round(v, 2) for k, v in cost_by_type.items()}
        report.hours_by_instance_type = hours_by_type

    def _calculate_spot_savings(self, report: CostReportData):
        """计算 Spot 实例带来的成本节省"""
        spot_hours = report.hours_by_instance_type.get("spot", 0)
        total_hours = sum(report.hours_by_instance_type.values())

        if total_hours > 0:
            report.spot_ratio = spot_hours / total_hours

        # 如果全部使用 on-demand 的成本
        on_demand_rate = self.get_cost_per_hour("on_demand")
        spot_rate = self.get_cost_per_hour("spot")

        hypothetical_od_cost = spot_hours * on_demand_rate
        actual_spot_cost = spot_hours * spot_rate
        report.spot_savings_usd = round(hypothetical_od_cost - actual_spot_cost, 2)

    def _identify_waste(self, report: CostReportData):
        """识别资源浪费"""
        on_demand_rate = self.get_cost_per_hour("on_demand")

        # 1. 低利用率时段 (模拟凌晨 0-8 点低负载)
        idle_hours = 8 * 8  # 8 小时 * 8 GPU
        idle_cost = idle_hours * on_demand_rate * 0.5  # 假设 50% 空闲
        report.waste_records.append(WasteRecord(
            category="idle_gpu",
            description="凌晨 0:00-8:00 GPU 利用率低于 20%, 存在大量空闲",
            gpu_hours_wasted=idle_hours * 0.5,
            cost_wasted_usd=round(idle_cost, 2),
            time_range="00:00 - 08:00",
            recommendation="启用定时缩容, 凌晨时段缩减到 1 个实例",
        ))

        # 2. 过量配置
        over_provision_hours = 16
        over_provision_cost = over_provision_hours * on_demand_rate
        report.waste_records.append(WasteRecord(
            category="over_provisioned",
            description="team-analytics 分配 4 GPU 但平均利用率仅 35%",
            gpu_hours_wasted=over_provision_hours,
            cost_wasted_usd=round(over_provision_cost, 2),
            time_range="全天",
            recommendation="将 team-analytics 从独占实例改为共享队列",
        ))

        # 3. Spot 转换机会
        convertible_hours = 30
        savings = convertible_hours * (on_demand_rate - self.get_cost_per_hour("spot"))
        report.waste_records.append(WasteRecord(
            category="spot_opportunity",
            description="team-code 的批处理任务可改用 Spot 实例",
            gpu_hours_wasted=0,
            cost_wasted_usd=round(savings, 2),
            time_range="工作日白天",
            recommendation="批处理负载迁移到 Spot 实例, 预计节省 60%",
        ))

        report.total_waste_usd = round(
            sum(w.cost_wasted_usd for w in report.waste_records), 2
        )

        # 月度优化潜力
        report.potential_monthly_savings = round(report.total_waste_usd * 30, 2)

    def _calculate_totals(self, report: CostReportData):
        """汇总计算"""
        report.total_cost_usd = round(
            sum(t.cost_usd for t in report.team_costs), 2
        )
        report.total_gpu_hours = sum(t.gpu_hours for t in report.team_costs)
        report.total_tokens = sum(t.tokens_generated for t in report.team_costs)

    def to_json(self, report: CostReportData) -> Dict:
        """转为 JSON 格式"""
        return {
            "report_date": report.report_date,
            "generated_at": report.generated_at,
            "period": report.period,
            "summary": {
                "total_cost_usd": report.total_cost_usd,
                "total_gpu_hours": report.total_gpu_hours,
                "total_tokens": report.total_tokens,
                "spot_savings_usd": report.spot_savings_usd,
                "spot_ratio": round(report.spot_ratio, 2),
                "total_waste_usd": report.total_waste_usd,
                "potential_monthly_savings": report.potential_monthly_savings,
            },
            "cost_by_instance_type": report.cost_by_instance_type,
            "hours_by_instance_type": report.hours_by_instance_type,
            "team_breakdown": [
                {
                    "team_id": t.team_id,
                    "team_name": t.team_name,
                    "gpu_hours": t.gpu_hours,
                    "tokens_generated": t.tokens_generated,
                    "cost_usd": t.cost_usd,
                    "cost_per_1m_tokens": t.cost_per_1m_tokens,
                    "models_used": t.models_used,
                    "instance_types": t.instance_types,
                }
                for t in report.team_costs
            ],
            "waste_analysis": [
                {
                    "category": w.category,
                    "description": w.description,
                    "gpu_hours_wasted": w.gpu_hours_wasted,
                    "cost_wasted_usd": w.cost_wasted_usd,
                    "time_range": w.time_range,
                    "recommendation": w.recommendation,
                }
                for w in report.waste_records
            ],
        }

    def to_markdown(self, report: CostReportData) -> str:
        """转为 Markdown 文本"""
        lines = [
            f"# GPU 成本报告",
            f"**日期:** {report.report_date} | **周期:** {report.period}",
            "",
            "---",
            "",
            "## 1. 成本概览",
            "",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| 总成本 | ${report.total_cost_usd:.2f} |",
            f"| 总 GPU 时 | {report.total_gpu_hours:.1f} h |",
            f"| 总 Token 数 | {report.total_tokens:,} |",
            f"| Spot 节省 | ${report.spot_savings_usd:.2f} |",
            f"| Spot 占比 | {report.spot_ratio:.0%} |",
            f"| 识别浪费 | ${report.total_waste_usd:.2f} |",
            "",
            "## 2. 实例类型成本分布",
            "",
            f"| 实例类型 | GPU 时 | 成本 |",
            f"|----------|--------|------|",
        ]

        for inst_type in report.cost_by_instance_type:
            hours = report.hours_by_instance_type.get(inst_type, 0)
            cost = report.cost_by_instance_type[inst_type]
            lines.append(f"| {inst_type} | {hours:.1f} h | ${cost:.2f} |")

        lines.extend([
            "",
            "## 3. 团队成本分摊 (Chargeback)",
            "",
            f"| 团队 | GPU 时 | Token 数 | 成本 | 单价($/1M tokens) |",
            f"|------|--------|----------|------|-------------------|",
        ])

        for t in sorted(report.team_costs, key=lambda x: x.cost_usd, reverse=True):
            lines.append(
                f"| {t.team_name} | {t.gpu_hours:.1f} h | {t.tokens_generated:,} "
                f"| ${t.cost_usd:.2f} | ${t.cost_per_1m_tokens:.2f} |"
            )

        lines.extend([
            "",
            "## 4. 浪费识别",
            "",
        ])

        for i, w in enumerate(report.waste_records, 1):
            lines.extend([
                f"### 4.{i} {w.category}",
                f"- **描述:** {w.description}",
                f"- **浪费成本:** ${w.cost_wasted_usd:.2f}/天",
                f"- **时间范围:** {w.time_range}",
                f"- **建议:** {w.recommendation}",
                "",
            ])

        lines.extend([
            "## 5. 优化潜力",
            "",
            f"- 月度潜在节省: **${report.potential_monthly_savings:.2f}**",
            "",
            "---",
            f"*报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.generated_at))}*",
        ])

        return "\n".join(lines)
