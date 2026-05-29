"""
诊断报告生成器模块

聚合设备发现、健康诊断、性能基准测试的结果，
生成HTML和JSON格式的综合诊断报告。包含摘要表格、
节点/链路健康状态、性能对比图表等。
"""

import logging
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import datetime

logger = logging.getLogger(__name__)


@dataclass
class ReportSection:
    """报告章节"""
    title: str
    section_id: str
    content_html: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # info, warning, error, critical


@dataclass
class DiagnosticReport:
    """完整的诊断报告"""
    report_id: str
    generated_at: str
    cluster_name: str
    sections: List[ReportSection] = field(default_factory=list)
    overall_health: str = "healthy"  # healthy, warning, degraded, critical
    summary_text: str = ""


class ReportGenerator:
    """
    诊断报告生成器

    将各个模块的诊断结果汇总成统一格式的报告，
    支持HTML和JSON两种输出格式。
    """

    # HTML报告模板
    HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: #f5f6fa; color: #2c3e50; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #2c3e50, #3498db);
                   color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
        .header h1 {{ font-size: 24px; margin-bottom: 10px; }}
        .header .meta {{ opacity: 0.8; font-size: 14px; }}
        .health-badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px;
                         font-size: 14px; font-weight: bold; color: white; }}
        .health-healthy {{ background: #27ae60; }}
        .health-warning {{ background: #f39c12; }}
        .health-degraded {{ background: #e67e22; }}
        .health-critical {{ background: #e74c3c; }}
        .section {{ background: white; border-radius: 10px; padding: 20px;
                    margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
        .section h2 {{ font-size: 18px; margin-bottom: 15px; padding-bottom: 10px;
                       border-bottom: 2px solid #ecf0f1; }}
        .section h2 .badge {{ float: right; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #ecf0f1; }}
        th {{ background: #f8f9fa; font-weight: 600; color: #7f8c8d; text-transform: uppercase;
              font-size: 12px; }}
        tr:hover {{ background: #f8f9fa; }}
        .status-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%;
                       margin-right: 6px; }}
        .dot-green {{ background: #27ae60; }}
        .dot-yellow {{ background: #f39c12; }}
        .dot-orange {{ background: #e67e22; }}
        .dot-red {{ background: #e74c3c; }}
        .dot-gray {{ background: #95a5a6; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                         gap: 15px; margin: 15px 0; }}
        .summary-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
        .summary-card .value {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
        .summary-card .label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; }}
        .alert {{ padding: 12px 15px; border-radius: 6px; margin: 8px 0; }}
        .alert-warning {{ background: #fef9e7; border-left: 4px solid #f39c12; }}
        .alert-critical {{ background: #fdedec; border-left: 4px solid #e74c3c; }}
        .alert-info {{ background: #eaf2f8; border-left: 4px solid #3498db; }}
        .progress-bar {{ height: 8px; background: #ecf0f1; border-radius: 4px; overflow: hidden; }}
        .progress-fill {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
        .footer {{ text-align: center; padding: 20px; color: #95a5a6; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        {header}
        {summary}
        {sections}
        <div class="footer">
            集群网络诊断平台 | 报告生成时间: {timestamp}
        </div>
    </div>
</body>
</html>"""

    def __init__(self, config: dict):
        """
        初始化报告生成器

        Args:
            config: 配置字典
        """
        self.cluster_name = config.get("cluster_name", "AI集群")
        self.output_dir = config.get("report_output_dir", "/tmp/cluster-diag-reports")
        self.sections: List[ReportSection] = []
        self.overall_health = "healthy"
        self.discovery_data: Optional[dict] = None
        self.diagnosis_data: Optional[dict] = None
        self.benchmark_data: Optional[dict] = None

    def set_discovery_results(self, data: dict) -> None:
        """设置设备发现结果数据"""
        self.discovery_data = data
        logger.info("已加载设备发现结果")

    def set_diagnosis_results(self, data: dict) -> None:
        """设置健康诊断结果数据"""
        self.diagnosis_data = data
        logger.info("已加载健康诊断结果")

    def set_benchmark_results(self, data: dict) -> None:
        """设置性能基准测试结果数据"""
        self.benchmark_data = data
        logger.info("已加载性能基准测试结果")

    def _build_discovery_section(self) -> Optional[ReportSection]:
        """构建设备发现章节"""
        if not self.discovery_data:
            return None

        summary = self.discovery_data.get("summary", {})
        nodes = self.discovery_data.get("nodes", {})

        # 构建HTML表格
        rows_html = ""
        for hostname, node_info in nodes.items():
            status_dot = "dot-green" if node_info.get("scan_success") else "dot-red"
            devices = node_info.get("devices", [])
            dev_names = ", ".join(d.get("device_name", "") for d in devices)
            fw_versions = ", ".join(
                d.get("firmware_version", "") for d in devices
            )
            active = node_info.get("active_ports", 0)
            total = node_info.get("total_ports", 0)

            rows_html += f"""
            <tr>
                <td><span class="status-dot {status_dot}"></span>{hostname}</td>
                <td>{node_info.get('ip_address', '')}</td>
                <td>{dev_names}</td>
                <td>{fw_versions}</td>
                <td>{active}/{total}</td>
            </tr>"""

        content_html = f"""
        <div class="summary-grid">
            <div class="summary-card">
                <div class="value">{summary.get('total_nodes', 0)}</div>
                <div class="label">总节点数</div>
            </div>
            <div class="summary-card">
                <div class="value">{summary.get('total_devices', 0)}</div>
                <div class="label">RDMA设备数</div>
            </div>
            <div class="summary-card">
                <div class="value">{summary.get('active_ports', 0)}/{summary.get('total_ports', 0)}</div>
                <div class="label">活跃端口</div>
            </div>
            <div class="summary-card">
                <div class="value">{summary.get('successful_scans', 0)}</div>
                <div class="label">扫描成功</div>
            </div>
        </div>
        <table>
            <thead>
                <tr>
                    <th>主机名</th><th>IP地址</th><th>RDMA设备</th>
                    <th>固件版本</th><th>活跃端口</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>"""

        severity = "info"
        if summary.get("failed_scans", 0) > 0:
            severity = "warning"

        return ReportSection(
            title="RDMA设备清单",
            section_id="discovery",
            content_html=content_html,
            data=summary,
            severity=severity,
        )

    def _build_diagnosis_section(self) -> Optional[ReportSection]:
        """构建健康诊断章节"""
        if not self.diagnosis_data:
            return None

        health_summary = self.diagnosis_data.get("summary", {})
        results = self.diagnosis_data.get("results", [])

        # 状态颜色映射
        status_colors = {
            "healthy": "dot-green",
            "warning": "dot-yellow",
            "degraded": "dot-orange",
            "critical": "dot-red",
            "down": "dot-gray",
        }

        rows_html = ""
        for r in results:
            status = r.get("status", "unknown")
            dot_class = status_colors.get(status, "dot-gray")
            issues = r.get("issues", [])
            issue_text = "; ".join(issues[:3]) if issues else "无"

            rows_html += f"""
            <tr>
                <td><span class="status-dot {dot_class}"></span>{r.get('node_hostname', '')}</td>
                <td>{r.get('device_name', '')}/{r.get('port_number', '')}</td>
                <td>{status}</td>
                <td>{r.get('counters', {}).get('total_errors', 0)}</td>
                <td>{issue_text}</td>
            </tr>"""

        # 告警信息
        alerts_html = ""
        critical_results = [r for r in results if r.get("status") == "critical"]
        for cr in critical_results:
            recs = cr.get("recommendations", [])
            rec_text = "<br>".join(recs)
            alerts_html += f"""
            <div class="alert alert-critical">
                <strong>{cr.get('node_hostname', '')} {cr.get('device_name', '')}/{cr.get('port_number', '')}</strong>:
                {rec_text}
            </div>"""

        content_html = f"""
        <div class="summary-grid">
            <div class="summary-card">
                <div class="value" style="color: #27ae60;">{health_summary.get('healthy', 0)}</div>
                <div class="label">正常</div>
            </div>
            <div class="summary-card">
                <div class="value" style="color: #f39c12;">{health_summary.get('warning', 0)}</div>
                <div class="label">告警</div>
            </div>
            <div class="summary-card">
                <div class="value" style="color: #e67e22;">{health_summary.get('degraded', 0)}</div>
                <div class="label">退化</div>
            </div>
            <div class="summary-card">
                <div class="value" style="color: #e74c3c;">{health_summary.get('critical', 0)}</div>
                <div class="label">严重</div>
            </div>
        </div>
        {alerts_html}
        <table>
            <thead>
                <tr>
                    <th>节点</th><th>设备/端口</th><th>状态</th>
                    <th>错误数</th><th>问题</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>"""

        severity = "info"
        if health_summary.get("critical", 0) > 0:
            severity = "critical"
            self.overall_health = "critical"
        elif health_summary.get("degraded", 0) > 0:
            severity = "error"
            if self.overall_health not in ("critical",):
                self.overall_health = "degraded"
        elif health_summary.get("warning", 0) > 0:
            severity = "warning"
            if self.overall_health == "healthy":
                self.overall_health = "warning"

        return ReportSection(
            title="链路健康诊断",
            section_id="diagnosis",
            content_html=content_html,
            data=health_summary,
            severity=severity,
        )

    def _build_benchmark_section(self) -> Optional[ReportSection]:
        """构建性能基准测试章节"""
        if not self.benchmark_data:
            return None

        bw_data = self.benchmark_data.get("bandwidth", {})
        lat_data = self.benchmark_data.get("latency", {})
        nccl_data = self.benchmark_data.get("nccl", {})

        content_parts = []

        # 带宽测试摘要
        if bw_data:
            bw_stats = bw_data.get("statistics", {})
            underperformers = bw_data.get("underperforming_pairs", [])

            content_parts.append(f"""
            <h3>RDMA带宽测试</h3>
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="value">{bw_stats.get('avg_bandwidth_gbps', 0):.1f}</div>
                    <div class="label">平均带宽 (Gb/s)</div>
                </div>
                <div class="summary-card">
                    <div class="value">{bw_stats.get('min_bandwidth_gbps', 0):.1f}</div>
                    <div class="label">最小带宽 (Gb/s)</div>
                </div>
                <div class="summary-card">
                    <div class="value">{bw_stats.get('max_bandwidth_gbps', 0):.1f}</div>
                    <div class="label">最大带宽 (Gb/s)</div>
                </div>
                <div class="summary-card">
                    <div class="value">{bw_stats.get('avg_efficiency', 0):.0%}</div>
                    <div class="label">平均效率</div>
                </div>
            </div>""")

            if underperformers:
                alerts = "\n".join(
                    f'<div class="alert alert-warning">{p}</div>'
                    for p in underperformers[:5]
                )
                content_parts.append(f"<h4>性能不达标的链路</h4>\n{alerts}")

        # 延迟测试摘要
        if lat_data:
            lat_stats = lat_data.get("statistics", {})
            content_parts.append(f"""
            <h3>RDMA延迟测试</h3>
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="value">{lat_stats.get('cluster_avg_latency_us', 0):.2f}</div>
                    <div class="label">平均延迟 (us)</div>
                </div>
                <div class="summary-card">
                    <div class="value">{lat_stats.get('cluster_p50_us', 0):.2f}</div>
                    <div class="label">P50延迟 (us)</div>
                </div>
                <div class="summary-card">
                    <div class="value">{lat_stats.get('cluster_p99_us', 0):.2f}</div>
                    <div class="label">P99延迟 (us)</div>
                </div>
                <div class="summary-card">
                    <div class="value">{lat_stats.get('cluster_p999_us', 0):.2f}</div>
                    <div class="label">P999延迟 (us)</div>
                </div>
            </div>""")

        # NCCL测试摘要
        if nccl_data:
            nccl_summary = nccl_data.get("summary", {})
            rows_html = ""
            for op_name, op_data in nccl_summary.items():
                rows_html += f"""
                <tr>
                    <td>{op_name}</td>
                    <td>{op_data.get('peak_bus_bw_gbps', 0):.2f} GB/s</td>
                    <td>{op_data.get('peak_algo_bw_gbps', 0):.2f} GB/s</td>
                    <td>{op_data.get('peak_message_size', '-')}</td>
                    <td>{'通过' if op_data.get('success') else '失败'}</td>
                </tr>"""

            content_parts.append(f"""
            <h3>NCCL集合通信测试</h3>
            <table>
                <thead>
                    <tr><th>操作</th><th>峰值总线带宽</th><th>峰值算法带宽</th>
                    <th>峰值消息大小</th><th>状态</th></tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>""")

        content_html = "\n".join(content_parts) if content_parts else "<p>无基准测试数据</p>"

        return ReportSection(
            title="性能基准测试",
            section_id="benchmark",
            content_html=content_html,
            data=self.benchmark_data,
            severity="info",
        )

    def build_report(self) -> DiagnosticReport:
        """
        构建完整的诊断报告

        Returns:
            DiagnosticReport对象
        """
        report = DiagnosticReport(
            report_id=f"diag-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
            generated_at=datetime.datetime.now().isoformat(),
            cluster_name=self.cluster_name,
        )

        # 构建各章节
        section_builders = [
            self._build_discovery_section,
            self._build_diagnosis_section,
            self._build_benchmark_section,
        ]

        for builder in section_builders:
            section = builder()
            if section:
                report.sections.append(section)

        report.overall_health = self.overall_health

        # 生成摘要文本
        report.summary_text = self._generate_summary_text(report)

        logger.info(
            f"诊断报告构建完成: {report.report_id}, "
            f"整体状态: {report.overall_health}"
        )
        return report

    def _generate_summary_text(self, report: DiagnosticReport) -> str:
        """生成报告摘要文本"""
        parts = [f"集群 {self.cluster_name} 诊断报告"]

        if self.discovery_data:
            summary = self.discovery_data.get("summary", {})
            parts.append(
                f"设备发现: {summary.get('total_nodes', 0)} 节点, "
                f"{summary.get('total_devices', 0)} 设备, "
                f"{summary.get('active_ports', 0)}/{summary.get('total_ports', 0)} 端口活跃"
            )

        if self.diagnosis_data:
            hs = self.diagnosis_data.get("summary", {})
            parts.append(
                f"健康状态: 正常 {hs.get('healthy', 0)}, "
                f"告警 {hs.get('warning', 0)}, "
                f"严重 {hs.get('critical', 0)}"
            )

        if self.benchmark_data:
            bw = self.benchmark_data.get("bandwidth", {}).get("statistics", {})
            if bw:
                parts.append(
                    f"带宽: 平均 {bw.get('avg_bandwidth_gbps', 0):.1f} Gb/s, "
                    f"效率 {bw.get('avg_efficiency', 0):.0%}"
                )

        return " | ".join(parts)

    def render_html(self, report: DiagnosticReport) -> str:
        """
        渲染HTML报告

        Args:
            report: 诊断报告对象

        Returns:
            HTML字符串
        """
        # 头部
        health_class = f"health-{report.overall_health}"
        header_html = f"""
        <div class="header">
            <h1>{report.cluster_name} - 网络诊断报告
                <span class="health-badge {health_class}">{report.overall_health.upper()}</span>
            </h1>
            <div class="meta">
                报告ID: {report.report_id} | 生成时间: {report.generated_at}
            </div>
        </div>"""

        # 摘要
        summary_html = f"""
        <div class="section">
            <h2>摘要</h2>
            <p>{report.summary_text}</p>
        </div>"""

        # 各章节
        sections_html = ""
        for section in report.sections:
            badge_class = {
                "info": "",
                "warning": "health-warning",
                "error": "health-degraded",
                "critical": "health-critical",
            }.get(section.severity, "")

            badge_html = ""
            if section.severity != "info":
                badge_html = f'<span class="health-badge {badge_class}">{section.severity.upper()}</span>'

            sections_html += f"""
            <div class="section">
                <h2>{section.title} {badge_html}</h2>
                {section.content_html}
            </div>"""

        html = self.HTML_TEMPLATE.format(
            title=f"{report.cluster_name} - 网络诊断报告",
            header=header_html,
            summary=summary_html,
            sections=sections_html,
            timestamp=report.generated_at,
        )

        return html

    def save_report(self, report: DiagnosticReport,
                    output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        保存报告到文件

        Args:
            report: 诊断报告
            output_dir: 输出目录

        Returns:
            文件路径字典 {"html": "...", "json": "..."}
        """
        out_dir = output_dir or self.output_dir
        os.makedirs(out_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = {}

        # 保存HTML报告
        html_path = os.path.join(out_dir, f"report_{timestamp}.html")
        html_content = self.render_html(report)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        paths["html"] = html_path
        logger.info(f"HTML报告已保存: {html_path}")

        # 保存JSON报告
        json_path = os.path.join(out_dir, f"report_{timestamp}.json")
        json_data = {
            "report_id": report.report_id,
            "generated_at": report.generated_at,
            "cluster_name": report.cluster_name,
            "overall_health": report.overall_health,
            "summary": report.summary_text,
            "discovery": self.discovery_data,
            "diagnosis": self.diagnosis_data,
            "benchmark": self.benchmark_data,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
        paths["json"] = json_path
        logger.info(f"JSON报告已保存: {json_path}")

        return paths
