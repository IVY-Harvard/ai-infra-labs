"""模型版本管理"""

import time
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class VersionDiff:
    """版本差异"""
    version_a: str
    version_b: str
    size_diff_bytes: int
    format_changed: bool
    metadata_changes: Dict


class VersionManager:
    """版本管理器 — 提供语义化版本和版本比较"""

    @staticmethod
    def parse_version(version: str) -> tuple:
        """解析语义化版本号"""
        parts = version.lstrip("v").split(".")
        return tuple(int(p) for p in parts)

    @staticmethod
    def next_version(current: str, bump: str = "patch") -> str:
        """生成下一个版本号"""
        parts = list(VersionManager.parse_version(current))
        while len(parts) < 3:
            parts.append(0)

        if bump == "major":
            parts[0] += 1
            parts[1] = 0
            parts[2] = 0
        elif bump == "minor":
            parts[1] += 1
            parts[2] = 0
        else:
            parts[2] += 1

        return f"v{parts[0]}.{parts[1]}.{parts[2]}"

    @staticmethod
    def compare_versions(versions: List[str]) -> List[str]:
        """按版本号排序"""
        return sorted(versions,
                      key=lambda v: VersionManager.parse_version(v))

    @staticmethod
    def diff(version_a: dict, version_b: dict) -> VersionDiff:
        """比较两个版本的差异"""
        return VersionDiff(
            version_a=version_a.get("version", ""),
            version_b=version_b.get("version", ""),
            size_diff_bytes=(version_b.get("size_bytes", 0) -
                           version_a.get("size_bytes", 0)),
            format_changed=(version_a.get("format") !=
                          version_b.get("format")),
            metadata_changes={
                k: (version_a.get("metadata", {}).get(k),
                    version_b.get("metadata", {}).get(k))
                for k in set(list(version_a.get("metadata", {}).keys()) +
                           list(version_b.get("metadata", {}).keys()))
                if version_a.get("metadata", {}).get(k) !=
                   version_b.get("metadata", {}).get(k)
            },
        )
