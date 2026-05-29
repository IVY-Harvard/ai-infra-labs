"""
HTML 报告生成器

将基准测试结果生成可视化的 HTML 报告。
"""

import json
from datetime import datetime
from typing import Dict, List, Any


class HTMLReporter:
    """HTML 报告生成器"""

    def __init__(self):
        self.sections = []

    def add_header(self, title: str, subtitle: str = ""):
        """添加报告头"""
        self.sections.append(f"""
        <div class="header">
            <h1>{title}</h1>
            <p>{subtitle}</p>
            <p class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """)

    def add_table(self, title: str, headers: List[str], rows: List[List[str]],
                  highlight_col: int = -1):
        """添加数据表格"""
        header_html = "".join(f"<th>{h}</th>" for h in headers)
        rows_html = ""
        for row in rows:
            cells = ""
            for i, cell in enumerate(row):
                cls = ' class="highlight"' if i == highlight_col else ""
                cells += f"<td{cls}>{cell}</td>"
            rows_html += f"<tr>{cells}</tr>"

        self.sections.append(f"""
        <div class="section">
            <h2>{title}</h2>
            <table>
                <thead><tr>{header_html}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """)

    def add_metric_card(self, title: str, value: str, subtitle: str = "", color: str = "green"):
        """添加指标卡片"""
        self.sections.append(f"""
        <div class="metric-card {color}">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtitle">{subtitle}</div>
        </div>
        """)

    def add_section_title(self, title: str):
        self.sections.append(f'<h2 class="section-title">{title}</h2>')

    def add_p2p_matrix(self, title: str, matrix: List[List[float]], num_gpus: int):
        """添加 P2P 带宽矩阵"""
        header_html = "<th></th>" + "".join(f"<th>GPU {i}</th>" for i in range(num_gpus))
        rows_html = ""
        for i in range(num_gpus):
            cells = f"<td><strong>GPU {i}</strong></td>"
            for j in range(num_gpus):
                val = matrix[i][j] if i != j else "-"
                if isinstance(val, float):
                    cells += f"<td>{val:.0f}</td>"
                else:
                    cells += f"<td>{val}</td>"
            rows_html += f"<tr>{cells}</tr>"

        self.sections.append(f"""
        <div class="section">
            <h2>{title}</h2>
            <table class="matrix">
                <thead><tr>{header_html}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """)

    def generate(self, output_path: str):
        """生成 HTML 文件"""
        content = "\n".join(self.sections)
        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>GPU Benchmark Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           margin: 40px; background: #f5f5f5; color: #333; }}
    .header {{ text-align: center; margin-bottom: 30px; }}
    .header h1 {{ color: #1a1a1a; margin-bottom: 5px; }}
    .timestamp {{ color: #888; font-size: 14px; }}
    .section {{ background: white; border-radius: 8px; padding: 20px;
               margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .section h2 {{ color: #2d2d2d; border-bottom: 2px solid #eee; padding-bottom: 8px; }}
    .section-title {{ color: #1a1a1a; margin-top: 30px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
    th, td {{ padding: 10px 15px; text-align: right; border-bottom: 1px solid #eee; }}
    th {{ background: #f8f8f8; font-weight: 600; color: #555; }}
    td:first-child {{ text-align: left; font-weight: 500; }}
    .highlight {{ color: #2196F3; font-weight: 600; }}
    .metric-card {{ display: inline-block; width: 200px; margin: 10px;
                   padding: 20px; border-radius: 8px; text-align: center;
                   background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .metric-title {{ font-size: 14px; color: #888; margin-bottom: 5px; }}
    .metric-value {{ font-size: 28px; font-weight: 700; }}
    .metric-subtitle {{ font-size: 12px; color: #aaa; margin-top: 5px; }}
    .green .metric-value {{ color: #4CAF50; }}
    .yellow .metric-value {{ color: #FF9800; }}
    .red .metric-value {{ color: #F44336; }}
    .matrix td {{ text-align: center; }}
</style>
</head>
<body>
{content}
</body>
</html>"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"报告已生成: {output_path}")


def generate_benchmark_report(results: Dict[str, Any], output_path: str):
    """从基准测试结果生成报告"""
    reporter = HTMLReporter()
    reporter.add_header(
        "GPU Kernel Benchmark Report",
        f"Tested {results.get('num_gpus', 'N/A')} GPU(s)"
    )

    # 计算性能
    if 'compute' in results:
        reporter.add_section_title("Compute Performance")
        for dtype, res_list in results['compute'].items():
            rows = []
            for r in res_list:
                color = "green" if r['utilization'] > 80 else ("yellow" if r['utilization'] > 50 else "red")
                rows.append([
                    r['dtype'], str(r['M']),
                    f"{r['time_ms']:.3f}",
                    f"{r['tflops']:.2f}",
                    f"{r['peak_tflops']:.1f}",
                    f"{r['utilization']:.1f}%",
                ])
            reporter.add_table(
                f"Compute ({dtype})",
                ['Dtype', 'Size', 'Time (ms)', 'TFLOPS', 'Peak', 'Utilization'],
                rows,
                highlight_col=3,
            )

    # 内存带宽
    if 'memory' in results:
        reporter.add_section_title("Memory Bandwidth")
        rows = []
        for category, res_list in results['memory'].items():
            for r in res_list:
                rows.append([
                    r['test_name'],
                    f"{r['data_size_mb']:.0f}",
                    f"{r['time_ms']:.3f}",
                    f"{r['bandwidth_gb_s']:.1f}",
                    f"{r['utilization']:.1f}%",
                ])
        reporter.add_table(
            "HBM Bandwidth",
            ['Test', 'Size (MB)', 'Time (ms)', 'BW (GB/s)', 'Utilization'],
            rows,
            highlight_col=3,
        )

    reporter.generate(output_path)


if __name__ == "__main__":
    # Demo: 生成示例报告
    reporter = HTMLReporter()
    reporter.add_header("GPU Benchmark Report (Demo)", "8x NVIDIA H20")

    reporter.add_metric_card("FP16 TFLOPS", "139.2", "Peak: 148.0", "green")
    reporter.add_metric_card("HBM BW", "3812 GB/s", "Peak: 4000 GB/s", "green")
    reporter.add_metric_card("Temperature", "72°C", "Max: 83°C", "yellow")

    reporter.add_table(
        "Compute Performance",
        ['GPU', 'FP32 TFLOPS', 'FP16 TFLOPS', 'INT8 TOPS'],
        [['GPU 0', '41.2', '139.2', '271.5'],
         ['GPU 1', '41.5', '140.1', '273.2']],
    )

    reporter.generate("demo_report.html")
