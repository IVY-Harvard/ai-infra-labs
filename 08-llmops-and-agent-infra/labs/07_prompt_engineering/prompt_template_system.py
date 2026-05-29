"""
Lab 07: Prompt 模板系统
生产级 Prompt 版本管理和模板渲染
"""
import os
import json
import yaml
import hashlib
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from jinja2 import Template, Environment, BaseLoader


@dataclass
class PromptVersion:
    """Prompt 版本"""
    name: str
    version: str
    template: str
    description: str = ""
    author: str = ""
    variables: list[dict] = field(default_factory=list)
    model_config: dict = field(default_factory=dict)
    eval_score: Optional[float] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def template_hash(self) -> str:
        return hashlib.md5(self.template.encode()).hexdigest()[:8]


class PromptTemplateSystem:
    """
    Prompt 模板系统 - 生产级实现
    功能：版本管理、模板渲染、变量验证、审计日志
    """

    def __init__(self, storage_dir: str = "./prompts"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.registry: dict[str, dict[str, PromptVersion]] = {}
        self.production_versions: dict[str, str] = {}
        self.render_log: list[dict] = []
        self.jinja_env = Environment(loader=BaseLoader())

    def register(self, prompt: PromptVersion) -> str:
        """注册新 Prompt 版本"""
        if prompt.name not in self.registry:
            self.registry[prompt.name] = {}

        key = prompt.version
        self.registry[prompt.name][key] = prompt

        # 保存到文件
        self._save_to_file(prompt)

        print(f"✓ 注册 Prompt: {prompt.name} v{prompt.version} "
              f"(hash: {prompt.template_hash})")
        return key

    def get(self, name: str, version: str = None) -> Optional[PromptVersion]:
        """获取 Prompt（默认返回生产版本）"""
        if name not in self.registry:
            return None

        if version is None:
            version = self.production_versions.get(name)
            if version is None:
                # 返回最新版本
                versions = sorted(self.registry[name].keys())
                version = versions[-1] if versions else None

        if version is None:
            return None
        return self.registry[name].get(version)

    def render(self, name: str, version: str = None, **variables) -> str:
        """渲染 Prompt 模板"""
        prompt = self.get(name, version)
        if not prompt:
            raise ValueError(f"Prompt not found: {name} v{version}")

        # 验证必需变量
        self._validate_variables(prompt, variables)

        # Jinja2 渲染
        template = self.jinja_env.from_string(prompt.template)
        rendered = template.render(**variables)

        # 记录渲染日志
        self.render_log.append({
            "name": name,
            "version": prompt.version,
            "hash": prompt.template_hash,
            "variables": {k: str(v)[:50] for k, v in variables.items()},
            "output_length": len(rendered),
            "timestamp": datetime.now().isoformat(),
        })

        return rendered

    def promote_to_production(self, name: str, version: str,
                               min_eval_score: float = 0.8):
        """将版本提升为生产版本（需通过评估门禁）"""
        prompt = self.get(name, version)
        if not prompt:
            raise ValueError(f"Prompt not found: {name} v{version}")

        if prompt.eval_score is not None and prompt.eval_score < min_eval_score:
            raise ValueError(
                f"评估分数 {prompt.eval_score} 低于阈值 {min_eval_score}"
            )

        old_version = self.production_versions.get(name, "none")
        self.production_versions[name] = version

        print(f"✓ 提升 {name} 生产版本: {old_version} → {version}")

    def diff(self, name: str, version_a: str, version_b: str) -> dict:
        """对比两个版本的差异"""
        a = self.get(name, version_a)
        b = self.get(name, version_b)

        if not a or not b:
            raise ValueError("版本不存在")

        return {
            "name": name,
            "version_a": version_a,
            "version_b": version_b,
            "template_changed": a.template != b.template,
            "hash_a": a.template_hash,
            "hash_b": b.template_hash,
            "model_config_changed": a.model_config != b.model_config,
            "eval_score_a": a.eval_score,
            "eval_score_b": b.eval_score,
        }

    def list_versions(self, name: str) -> list[dict]:
        """列出所有版本"""
        if name not in self.registry:
            return []

        versions = []
        for ver, prompt in sorted(self.registry[name].items()):
            is_production = self.production_versions.get(name) == ver
            versions.append({
                "version": ver,
                "hash": prompt.template_hash,
                "eval_score": prompt.eval_score,
                "created_at": prompt.created_at,
                "is_production": is_production,
            })
        return versions

    def _validate_variables(self, prompt: PromptVersion, variables: dict):
        """验证变量"""
        required = [v["name"] for v in prompt.variables if v.get("required", False)]
        missing = [v for v in required if v not in variables]
        if missing:
            raise ValueError(f"缺少必需变量: {missing}")

    def _save_to_file(self, prompt: PromptVersion):
        """保存到文件"""
        dir_path = self.storage_dir / prompt.name
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / f"v{prompt.version}.yaml"

        data = asdict(prompt)
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


# =============================================================================
# 演示
# =============================================================================

def main():
    print("=" * 60)
    print("Prompt 模板系统演示")
    print("=" * 60)

    system = PromptTemplateSystem("./demo_prompts")

    # 注册多个版本
    v1 = PromptVersion(
        name="qa",
        version="1.0.0",
        template="请回答以下问题：{{ question }}",
        description="基础 QA Prompt",
        author="team",
        variables=[{"name": "question", "type": "string", "required": True}],
        model_config={"temperature": 0.7, "max_tokens": 500},
        eval_score=0.75,
    )

    v2 = PromptVersion(
        name="qa",
        version="2.0.0",
        template="""你是一个专业的技术顾问。

## 上下文
{{ context }}

## 问题
{{ question }}

## 要求
1. 仅基于上下文回答
2. 如果不确定，请说明
3. 提供相关引用

## 回答""",
        description="带上下文和结构化输出的 QA Prompt",
        author="team",
        variables=[
            {"name": "question", "type": "string", "required": True},
            {"name": "context", "type": "string", "required": True},
        ],
        model_config={"temperature": 0.3, "max_tokens": 1000},
        eval_score=0.89,
    )

    system.register(v1)
    system.register(v2)

    # 提升为生产版本
    system.promote_to_production("qa", "2.0.0", min_eval_score=0.8)

    # 渲染
    print("\n--- 渲染 Prompt ---")
    rendered = system.render(
        "qa",
        question="H20 GPU 能跑多大的模型？",
        context="H20 GPU 有 96GB HBM3 显存，支持 FP8 推理。",
    )
    print(rendered)

    # 版本列表
    print("\n--- 版本列表 ---")
    for v in system.list_versions("qa"):
        prod_marker = " [PRODUCTION]" if v["is_production"] else ""
        print(f"  v{v['version']} (hash: {v['hash']}, "
              f"score: {v['eval_score']}){prod_marker}")

    # Diff
    print("\n--- 版本对比 ---")
    diff = system.diff("qa", "1.0.0", "2.0.0")
    print(f"  模板变更: {diff['template_changed']}")
    print(f"  评估分数: {diff['eval_score_a']} → {diff['eval_score_b']}")

    # 渲染日志
    print(f"\n--- 渲染日志 ({len(system.render_log)} 条) ---")
    for log in system.render_log:
        print(f"  {log['name']} v{log['version']} | "
              f"output_len={log['output_length']}")


if __name__ == "__main__":
    main()
