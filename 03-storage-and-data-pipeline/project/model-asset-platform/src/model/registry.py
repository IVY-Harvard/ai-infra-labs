"""模型注册中心"""

import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field

from ..storage.backend import StorageBackend


@dataclass
class ModelVersion:
    version: str
    format: str           # safetensors / bin / gguf
    size_bytes: int
    storage_key: str
    created_at: float
    checksum: str = ""
    metadata: Dict = field(default_factory=dict)
    status: str = "active"  # active / deprecated / deleted


@dataclass
class ModelEntry:
    name: str
    description: str
    owner: str
    created_at: float
    versions: Dict[str, ModelVersion] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    latest_version: str = ""


class ModelRegistry:
    """模型注册中心"""

    def __init__(self, backend: StorageBackend, registry_prefix: str = "_registry"):
        self.backend = backend
        self.prefix = registry_prefix
        self.models: Dict[str, ModelEntry] = {}
        self._load_registry()

    def _registry_key(self) -> str:
        return f"{self.prefix}/models.json"

    def _load_registry(self):
        data = self.backend.get(self._registry_key())
        if data:
            registry = json.loads(data.decode())
            for name, entry_data in registry.items():
                versions = {}
                for v, vdata in entry_data.get("versions", {}).items():
                    versions[v] = ModelVersion(**vdata)
                entry_data["versions"] = versions
                self.models[name] = ModelEntry(**entry_data)

    def _save_registry(self):
        data = {}
        for name, entry in self.models.items():
            d = asdict(entry)
            data[name] = d
        self.backend.put(self._registry_key(), json.dumps(data).encode())

    def register(self, name: str, description: str = "",
                 owner: str = "", tags: List[str] = None) -> ModelEntry:
        """注册新模型"""
        if name in self.models:
            raise ValueError(f"Model '{name}' already exists")

        entry = ModelEntry(
            name=name,
            description=description,
            owner=owner,
            created_at=time.time(),
            tags=tags or [],
        )
        self.models[name] = entry
        self._save_registry()
        return entry

    def add_version(self, name: str, version: str, format: str,
                    data: bytes, metadata: Dict = None) -> ModelVersion:
        """添加模型版本"""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found")

        storage_key = f"models/{name}/{version}/{name}.{format}"
        self.backend.put(storage_key, data)

        import hashlib
        checksum = hashlib.sha256(data).hexdigest()

        ver = ModelVersion(
            version=version,
            format=format,
            size_bytes=len(data),
            storage_key=storage_key,
            created_at=time.time(),
            checksum=checksum,
            metadata=metadata or {},
        )

        self.models[name].versions[version] = ver
        self.models[name].latest_version = version
        self._save_registry()
        return ver

    def get_model(self, name: str) -> Optional[ModelEntry]:
        return self.models.get(name)

    def get_version(self, name: str, version: str) -> Optional[ModelVersion]:
        entry = self.models.get(name)
        if entry:
            return entry.versions.get(version)
        return None

    def list_models(self) -> List[Dict]:
        return [
            {"name": e.name, "latest": e.latest_version,
             "versions": len(e.versions), "owner": e.owner}
            for e in self.models.values()
        ]

    def delete_version(self, name: str, version: str) -> bool:
        entry = self.models.get(name)
        if not entry or version not in entry.versions:
            return False
        ver = entry.versions[version]
        self.backend.delete(ver.storage_key)
        ver.status = "deleted"
        self._save_registry()
        return True
