"""
数据质量检查工具：自动检查微调数据集的质量问题

检查项:
- 格式合法性
- 内容长度
- 重复数据
- 特殊字符和编码
- 角色合法性
- Token 长度分布

用法:
    python data_quality_checker.py --input data.jsonl
    python data_quality_checker.py --input data.jsonl --report report.json --fix output_cleaned.jsonl
"""

import argparse
import json
import re
import hashlib
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional


class DataQualityChecker:
    """数据质量检查器"""

    def __init__(self, max_length=8000, min_length=10, min_assistant_length=5):
        self.max_length = max_length
        self.min_length = min_length
        self.min_assistant_length = min_assistant_length
        self.issues = defaultdict(list)
        self.stats = Counter()

    def check_all(self, data: List[Dict]) -> Dict:
        """运行所有检查"""
        print(f"开始检查 {len(data)} 条数据...")
        self.issues.clear()
        self.stats.clear()

        clean_data = []
        for idx, item in enumerate(data):
            problems = self._check_single(item, idx)
            if problems:
                for p in problems:
                    self.issues[p["type"]].append(p)
                    self.stats[p["type"]] += 1
            else:
                self.stats["clean"] += 1
                clean_data.append(item)

        # 全局检查
        dup_count = self._check_duplicates(data)
        self.stats["duplicates"] = dup_count

        # 统计信息
        length_stats = self._compute_length_stats(data)

        report = {
            "total": len(data),
            "clean": self.stats["clean"],
            "issues_summary": dict(self.stats),
            "length_stats": length_stats,
            "issue_details": {k: v[:10] for k, v in self.issues.items()},  # 每类最多10条
        }

        return report, clean_data

    def _check_single(self, item: Dict, idx: int) -> List[Dict]:
        """检查单条数据"""
        problems = []

        # 检测格式
        if "messages" in item:
            problems.extend(self._check_messages_format(item, idx))
        elif "conversations" in item:
            problems.extend(self._check_sharegpt_format(item, idx))
        elif "instruction" in item:
            problems.extend(self._check_alpaca_format(item, idx))
        else:
            problems.append({"type": "unknown_format", "idx": idx,
                           "detail": f"无法识别的格式，key: {list(item.keys())}"})
            return problems

        return problems

    def _check_messages_format(self, item: Dict, idx: int) -> List[Dict]:
        """检查 messages 格式"""
        problems = []
        messages = item.get("messages", [])

        if not messages:
            problems.append({"type": "empty_messages", "idx": idx, "detail": "messages 为空"})
            return problems

        valid_roles = {"system", "user", "assistant"}
        has_user = False
        has_assistant = False

        for i, msg in enumerate(messages):
            # 角色检查
            role = msg.get("role", "")
            if role not in valid_roles:
                problems.append({"type": "invalid_role", "idx": idx,
                               "detail": f"无效角色: {role}"})

            if role == "user":
                has_user = True
            if role == "assistant":
                has_assistant = True

            # 内容检查
            content = msg.get("content", "")
            if not content or not content.strip():
                problems.append({"type": "empty_content", "idx": idx,
                               "detail": f"第{i}条消息内容为空 (role={role})"})

            # 长度检查
            if role == "assistant" and len(content.strip()) < self.min_assistant_length:
                problems.append({"type": "too_short", "idx": idx,
                               "detail": f"assistant 回答过短: {len(content)} 字符"})

            if len(content) > self.max_length:
                problems.append({"type": "too_long", "idx": idx,
                               "detail": f"内容过长: {len(content)} 字符"})

            # 特殊字符检查
            if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', content):
                problems.append({"type": "special_chars", "idx": idx,
                               "detail": "包含不可见控制字符"})

        if not has_user:
            problems.append({"type": "missing_user", "idx": idx,
                           "detail": "缺少 user 消息"})
        if not has_assistant:
            problems.append({"type": "missing_assistant", "idx": idx,
                           "detail": "缺少 assistant 消息"})

        return problems

    def _check_sharegpt_format(self, item: Dict, idx: int) -> List[Dict]:
        """检查 ShareGPT 格式"""
        problems = []
        conversations = item.get("conversations", [])

        if len(conversations) < 2:
            problems.append({"type": "too_few_turns", "idx": idx,
                           "detail": f"对话轮数不足: {len(conversations)}"})

        for i, conv in enumerate(conversations):
            if "from" not in conv or "value" not in conv:
                problems.append({"type": "missing_fields", "idx": idx,
                               "detail": f"第{i}轮缺少 from/value 字段"})

        return problems

    def _check_alpaca_format(self, item: Dict, idx: int) -> List[Dict]:
        """检查 Alpaca 格式"""
        problems = []

        if not item.get("instruction", "").strip():
            problems.append({"type": "empty_instruction", "idx": idx,
                           "detail": "instruction 为空"})

        if not item.get("output", "").strip():
            problems.append({"type": "empty_output", "idx": idx,
                           "detail": "output 为空"})

        return problems

    def _check_duplicates(self, data: List[Dict]) -> int:
        """检查重复数据"""
        seen = set()
        dup_count = 0

        for item in data:
            content_str = json.dumps(item, ensure_ascii=False, sort_keys=True)
            content_hash = hashlib.md5(content_str.encode()).hexdigest()

            if content_hash in seen:
                dup_count += 1
            else:
                seen.add(content_hash)

        return dup_count

    def _compute_length_stats(self, data: List[Dict]) -> Dict:
        """计算长度统计"""
        lengths = []
        for item in data:
            total_len = len(json.dumps(item, ensure_ascii=False))
            lengths.append(total_len)

        if not lengths:
            return {}

        lengths.sort()
        return {
            "count": len(lengths),
            "min": lengths[0],
            "max": lengths[-1],
            "mean": sum(lengths) / len(lengths),
            "median": lengths[len(lengths) // 2],
            "p90": lengths[int(len(lengths) * 0.9)],
            "p99": lengths[int(len(lengths) * 0.99)],
        }


def print_report(report: Dict):
    """打印质量报告"""
    print("\n" + "=" * 60)
    print("数据质量报告")
    print("=" * 60)

    print(f"\n总数据量: {report['total']}")
    print(f"合格数据: {report['clean']} ({100*report['clean']/max(report['total'],1):.1f}%)")

    print("\n问题分布:")
    for issue_type, count in sorted(report["issues_summary"].items()):
        if issue_type != "clean":
            print(f"  {issue_type}: {count}")

    if report.get("length_stats"):
        stats = report["length_stats"]
        print(f"\n长度统计:")
        print(f"  最短: {stats['min']}, 最长: {stats['max']}")
        print(f"  平均: {stats['mean']:.0f}, 中位数: {stats['median']}")
        print(f"  P90: {stats['p90']}, P99: {stats['p99']}")

    if report.get("issue_details"):
        print("\n问题样本（每类最多 3 条）:")
        for issue_type, details in report["issue_details"].items():
            print(f"\n  [{issue_type}]")
            for d in details[:3]:
                print(f"    行 {d['idx']}: {d['detail']}")


def main():
    parser = argparse.ArgumentParser(description="数据质量检查工具")
    parser.add_argument("--input", required=True, help="输入数据文件")
    parser.add_argument("--report", default=None, help="报告输出文件 (JSON)")
    parser.add_argument("--fix", default=None, help="输出清洗后的数据")
    parser.add_argument("--max_length", type=int, default=8000)
    parser.add_argument("--min_length", type=int, default=10)
    args = parser.parse_args()

    # 加载数据
    data = []
    if args.input.endswith(".jsonl"):
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]

    # 检查
    checker = DataQualityChecker(
        max_length=args.max_length,
        min_length=args.min_length,
    )
    report, clean_data = checker.check_all(data)

    # 输出报告
    print_report(report)

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n报告已保存到: {args.report}")

    if args.fix:
        if args.fix.endswith(".jsonl"):
            with open(args.fix, "w", encoding="utf-8") as f:
                for item in clean_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            with open(args.fix, "w", encoding="utf-8") as f:
                json.dump(clean_data, f, ensure_ascii=False, indent=2)
        print(f"清洗后数据已保存到: {args.fix} ({len(clean_data)} 条)")


if __name__ == "__main__":
    main()
