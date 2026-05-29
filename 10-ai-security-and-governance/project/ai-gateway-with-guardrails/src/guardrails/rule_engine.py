"""Rule Engine - Load and execute guardrail rules from YAML config."""
import yaml
import re
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Rule:
    name: str
    type: str  # "regex", "keyword", "length"
    target: str  # "input" or "output"
    pattern: str = ""
    keywords: list = None
    max_length: int = 0
    action: str = "block"  # "block" or "warn"


class RuleEngine:
    def __init__(self):
        self.rules: list[Rule] = []

    def load_from_yaml(self, path: str):
        with open(path) as f:
            config = yaml.safe_load(f)
        for r in config.get("rules", []):
            self.rules.append(Rule(
                name=r["name"], type=r["type"], target=r["target"],
                pattern=r.get("pattern", ""), keywords=r.get("keywords", []),
                max_length=r.get("max_length", 0), action=r.get("action", "block"),
            ))

    def evaluate(self, text: str, target: str) -> list[dict]:
        violations = []
        for rule in self.rules:
            if rule.target != target:
                continue
            if rule.type == "regex" and re.search(rule.pattern, text, re.IGNORECASE):
                violations.append({"rule": rule.name, "action": rule.action})
            elif rule.type == "keyword":
                for kw in (rule.keywords or []):
                    if kw in text:
                        violations.append({"rule": rule.name, "action": rule.action})
                        break
            elif rule.type == "length" and len(text) > rule.max_length:
                violations.append({"rule": rule.name, "action": rule.action})
        return violations
