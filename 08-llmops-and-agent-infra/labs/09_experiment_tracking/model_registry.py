"""
Lab 09: 模型/配置注册中心
管理 LLM 应用各组件的版本和生命周期
"""
import json
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


class Stage(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    CANARY = "canary"
    PRODUCTION = "production"
    ARCHIVED = "archived"


VALID_TRANSITIONS = {
    Stage.DEVELOPMENT: [Stage.STAGING],
    Stage.STAGING: [Stage.CANARY, Stage.ARCHIVED],
    Stage.CANARY: [Stage.PRODUCTION, Stage.STAGING],
    Stage.PRODUCTION: [Stage.ARCHIVED],
    Stage.ARCHIVED: [Stage.STAGING],  # 可以重新启用
}


@dataclass
class VersionedConfig:
    """版本化的配置"""
    name: str
    version: str
    config_type: str  # "prompt" / "rag" / "agent" / "model"
    config: dict
    stage: Stage = Stage.DEVELOPMENT
    eval_scores: dict = field(default_factory=dict)
    created_by: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    promoted_at: Optional[str] = None
    notes: str = ""


class ModelRegistry:
    """
    模型/配置注册中心
    管理 LLM 应用各组件版本的完整生命周期
    """

    def __init__(self, storage_dir: str = "./registry"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.configs: dict[str, dict[str, VersionedConfig]] = {}
        self.history: list[dict] = []

    def register(self, config: VersionedConfig) -> str:
        """注册新版本"""
        if config.name not in self.configs:
            self.configs[config.name] = {}

        self.configs[config.name][config.version] = config
        self._save(config)
        self._log_event("register", config)

        print(f"✓ 注册: {config.name} v{config.version} ({config.config_type})")
        return f"{config.name}:{config.version}"

    def promote(self, name: str, version: str, target_stage: Stage,
                min_eval_score: float = 0.8) -> bool:
        """提升版本阶段"""
        config = self.get(name, version)
        if not config:
            raise ValueError(f"未找到: {name} v{version}")

        # 检查状态转换合法性
        if target_stage not in VALID_TRANSITIONS.get(config.stage, []):
            raise ValueError(
                f"非法状态转换: {config.stage.value} → {target_stage.value}"
            )

        # 进入 Canary/Production 需要评估门禁
        if target_stage in (Stage.CANARY, Stage.PRODUCTION):
            avg_score = (
                sum(config.eval_scores.values()) / len(config.eval_scores)
                if config.eval_scores else 0
            )
            if avg_score < min_eval_score:
                print(f"✗ 评估未通过: {avg_score:.2f} < {min_eval_score}")
                return False

        old_stage = config.stage
        config.stage = target_stage
        config.promoted_at = datetime.now().isoformat()

        # 如果提升到 Production，归档旧的 Production 版本
        if target_stage == Stage.PRODUCTION:
            self._archive_current_production(name, version)

        self._save(config)
        self._log_event("promote", config,
                        extra={"from": old_stage.value, "to": target_stage.value})

        print(f"✓ 提升: {name} v{version}: {old_stage.value} → {target_stage.value}")
        return True

    def get(self, name: str, version: str) -> Optional[VersionedConfig]:
        """获取指定版本"""
        return self.configs.get(name, {}).get(version)

    def get_production(self, name: str) -> Optional[VersionedConfig]:
        """获取生产版本"""
        for ver, config in self.configs.get(name, {}).items():
            if config.stage == Stage.PRODUCTION:
                return config
        return None

    def list_versions(self, name: str) -> list[dict]:
        """列出所有版本"""
        versions = []
        for ver, config in sorted(self.configs.get(name, {}).items()):
            versions.append({
                "version": ver,
                "stage": config.stage.value,
                "eval_scores": config.eval_scores,
                "created_at": config.created_at,
            })
        return versions

    def rollback(self, name: str) -> Optional[str]:
        """回滚到上一个 Production 版本"""
        archived = [
            (v, c) for v, c in self.configs.get(name, {}).items()
            if c.stage == Stage.ARCHIVED and c.promoted_at
        ]
        if not archived:
            print("✗ 无可回滚版本")
            return None

        # 找最近被归档的
        archived.sort(key=lambda x: x[1].promoted_at or "", reverse=True)
        rollback_version = archived[0][0]

        self.promote(name, rollback_version, Stage.PRODUCTION, min_eval_score=0)
        print(f"✓ 回滚到: {name} v{rollback_version}")
        return rollback_version

    def _archive_current_production(self, name: str, exclude_version: str):
        """归档当前的 Production 版本"""
        for ver, config in self.configs.get(name, {}).items():
            if config.stage == Stage.PRODUCTION and ver != exclude_version:
                config.stage = Stage.ARCHIVED
                self._log_event("archive", config)

    def _save(self, config: VersionedConfig):
        """保存到文件"""
        dir_path = self.storage_dir / config.name
        dir_path.mkdir(exist_ok=True)
        file_path = dir_path / f"v{config.version}.json"
        data = asdict(config)
        data["stage"] = config.stage.value
        file_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def _log_event(self, event_type: str, config: VersionedConfig, extra: dict = None):
        """记录事件"""
        self.history.append({
            "event": event_type,
            "name": config.name,
            "version": config.version,
            "stage": config.stage.value,
            "timestamp": datetime.now().isoformat(),
            **(extra or {}),
        })


def main():
    print("=" * 60)
    print("模型注册中心演示")
    print("=" * 60)

    registry = ModelRegistry()

    # 注册 RAG 配置版本
    v1 = VersionedConfig(
        name="rag-pipeline", version="1.0.0", config_type="rag",
        config={"chunk_size": 500, "top_k": 5, "reranker": "none"},
        eval_scores={"faithfulness": 0.78, "relevancy": 0.72},
        created_by="team",
    )
    v2 = VersionedConfig(
        name="rag-pipeline", version="2.0.0", config_type="rag",
        config={"chunk_size": 1000, "top_k": 10, "reranker": "bge-reranker-v2-m3", "hyde": True},
        eval_scores={"faithfulness": 0.92, "relevancy": 0.88},
        created_by="team",
    )

    registry.register(v1)
    registry.register(v2)

    # 生命周期管理
    registry.promote("rag-pipeline", "1.0.0", Stage.STAGING)
    registry.promote("rag-pipeline", "1.0.0", Stage.CANARY, min_eval_score=0.7)
    registry.promote("rag-pipeline", "1.0.0", Stage.PRODUCTION, min_eval_score=0.7)

    registry.promote("rag-pipeline", "2.0.0", Stage.STAGING)
    registry.promote("rag-pipeline", "2.0.0", Stage.CANARY)
    registry.promote("rag-pipeline", "2.0.0", Stage.PRODUCTION)

    # 查看状态
    print(f"\n版本列表:")
    for v in registry.list_versions("rag-pipeline"):
        print(f"  v{v['version']}: {v['stage']} | scores={v['eval_scores']}")

    prod = registry.get_production("rag-pipeline")
    if prod:
        print(f"\n生产版本: v{prod.version}")

    # 事件历史
    print(f"\n事件历史:")
    for event in registry.history[-5:]:
        print(f"  [{event['event']}] {event['name']} v{event['version']} → {event['stage']}")


if __name__ == "__main__":
    main()
