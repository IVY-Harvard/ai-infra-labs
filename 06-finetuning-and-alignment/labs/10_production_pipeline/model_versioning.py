"""
模型版本管理工具

用法:
    python model_versioning.py register --model_path ./output/lora --metadata '{"task": "cs"}'
    python model_versioning.py list
    python model_versioning.py promote --version v1.0 --status validated
    python model_versioning.py rollback --to_version v0.9
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Optional


class ModelVersionManager:
    """模型版本管理器"""

    def __init__(self, registry_path: str = "./model_registry"):
        self.registry_path = registry_path
        self.registry_file = os.path.join(registry_path, "registry.json")
        os.makedirs(registry_path, exist_ok=True)

        if os.path.exists(self.registry_file):
            with open(self.registry_file, "r") as f:
                self.registry = json.load(f)
        else:
            self.registry = {"models": [], "current_deployed": None}

    def _save(self):
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)

    def register(self, model_path: str, metadata: Dict = None, version: str = None):
        """注册新模型"""
        if version is None:
            version = f"v{len(self.registry['models']) + 1}.0"

        entry = {
            "version": version,
            "model_path": os.path.abspath(model_path),
            "status": "registered",  # registered → validated → approved → deployed
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "history": [{"status": "registered", "at": datetime.now().isoformat()}],
        }

        self.registry["models"].append(entry)
        self._save()
        print(f"已注册模型: {version} ({model_path})")
        return version

    def list_models(self, status: Optional[str] = None):
        """列出所有模型"""
        models = self.registry["models"]
        if status:
            models = [m for m in models if m["status"] == status]

        print(f"\n{'Version':<10} {'Status':<12} {'Path':<40} {'Created':<20}")
        print("-" * 85)
        for m in models:
            print(f"{m['version']:<10} {m['status']:<12} "
                  f"{m['model_path'][:38]:<40} {m['created_at'][:19]}")

        print(f"\n当前部署: {self.registry.get('current_deployed', 'None')}")

    def promote(self, version: str, new_status: str):
        """提升模型状态"""
        valid_transitions = {
            "registered": ["validated"],
            "validated": ["approved"],
            "approved": ["deployed"],
        }

        entry = self._find_version(version)
        if not entry:
            print(f"版本 {version} 不存在")
            return

        current = entry["status"]
        if new_status not in valid_transitions.get(current, []):
            print(f"无效状态转换: {current} → {new_status}")
            print(f"允许的转换: {current} → {valid_transitions.get(current, [])}")
            return

        # 如果是部署，先取消之前的部署
        if new_status == "deployed":
            for m in self.registry["models"]:
                if m["status"] == "deployed":
                    m["status"] = "superseded"
                    m["history"].append({"status": "superseded", "at": datetime.now().isoformat()})
            self.registry["current_deployed"] = version

        entry["status"] = new_status
        entry["history"].append({"status": new_status, "at": datetime.now().isoformat()})
        self._save()
        print(f"模型 {version}: {current} → {new_status}")

    def rollback(self, to_version: str):
        """回滚到指定版本"""
        target = self._find_version(to_version)
        if not target:
            print(f"版本 {to_version} 不存在")
            return

        # 取消当前部署
        for m in self.registry["models"]:
            if m["status"] == "deployed":
                m["status"] = "rolled_back"
                m["history"].append({"status": "rolled_back", "at": datetime.now().isoformat()})

        # 部署目标版本
        target["status"] = "deployed"
        target["history"].append({"status": "deployed (rollback)", "at": datetime.now().isoformat()})
        self.registry["current_deployed"] = to_version
        self._save()
        print(f"已回滚到: {to_version}")

    def get_deployed(self) -> Optional[Dict]:
        """获取当前部署的模型"""
        for m in self.registry["models"]:
            if m["status"] == "deployed":
                return m
        return None

    def _find_version(self, version: str) -> Optional[Dict]:
        for m in self.registry["models"]:
            if m["version"] == version:
                return m
        return None


def main():
    parser = argparse.ArgumentParser(description="模型版本管理")
    parser.add_argument("command", choices=["register", "list", "promote", "rollback", "deployed"])
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--status", default=None)
    parser.add_argument("--to_version", default=None)
    parser.add_argument("--metadata", default="{}")
    parser.add_argument("--registry", default="./model_registry")
    args = parser.parse_args()

    mgr = ModelVersionManager(args.registry)

    if args.command == "register":
        if not args.model_path:
            parser.error("register 需要 --model_path")
        metadata = json.loads(args.metadata)
        mgr.register(args.model_path, metadata, args.version)

    elif args.command == "list":
        mgr.list_models(args.status)

    elif args.command == "promote":
        if not args.version or not args.status:
            parser.error("promote 需要 --version 和 --status")
        mgr.promote(args.version, args.status)

    elif args.command == "rollback":
        if not args.to_version:
            parser.error("rollback 需要 --to_version")
        mgr.rollback(args.to_version)

    elif args.command == "deployed":
        deployed = mgr.get_deployed()
        if deployed:
            print(f"当前部署: {deployed['version']} ({deployed['model_path']})")
        else:
            print("当前无部署模型")


if __name__ == "__main__":
    main()
