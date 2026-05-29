"""Policy Loader - Load policies from YAML files."""
import yaml
from pathlib import Path
from .policy_engine import Policy, PolicyEngine


class PolicyLoader:
    def __init__(self, policy_dir: str = "./policies"):
        self.policy_dir = Path(policy_dir)

    def load_all(self, engine: PolicyEngine):
        if not self.policy_dir.exists():
            return
        for f in self.policy_dir.glob("*.yaml"):
            policy = self._load_file(f)
            if policy:
                engine.add_policy(policy)

    def _load_file(self, path: Path) -> Policy | None:
        with open(path) as f:
            data = yaml.safe_load(f)
        if not data:
            return None
        return Policy(
            name=data.get("name", path.stem),
            description=data.get("description", ""),
            rules=data.get("rules", []),
            enabled=data.get("enabled", True),
            severity=data.get("severity", "high"),
        )
