"""Policy Engine - Evaluate and enforce policies."""
from dataclasses import dataclass
from typing import Any


@dataclass
class Policy:
    name: str
    description: str
    rules: list[dict]
    enabled: bool = True
    severity: str = "high"  # high, medium, low


class PolicyEngine:
    def __init__(self):
        self.policies: list[Policy] = []

    def add_policy(self, policy: Policy):
        self.policies.append(policy)

    def evaluate(self, context: dict[str, Any]) -> list[dict]:
        violations = []
        for policy in self.policies:
            if not policy.enabled:
                continue
            for rule in policy.rules:
                if self._check_rule(rule, context):
                    violations.append({
                        "policy": policy.name,
                        "severity": policy.severity,
                        "rule": rule.get("name", "unnamed"),
                    })
        return violations

    def _check_rule(self, rule: dict, context: dict) -> bool:
        rule_type = rule.get("type")
        if rule_type == "max_tokens":
            return context.get("tokens", 0) > rule.get("value", float("inf"))
        elif rule_type == "blocked_model":
            return context.get("model") in rule.get("models", [])
        elif rule_type == "time_restriction":
            import time
            hour = time.localtime().tm_hour
            return not (rule.get("start", 0) <= hour < rule.get("end", 24))
        return False
