"""Compliance checker for audit logs."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ComplianceRule:
    """A single compliance rule."""
    rule_id: str
    description: str
    field: str
    check: str  # "required", "max_value", "allowed_values"
    value: Optional[any] = None


DEFAULT_RULES = [
    ComplianceRule("R001", "Event ID must be present", "event_id", "required"),
    ComplianceRule("R002", "Timestamp must be present", "timestamp", "required"),
    ComplianceRule("R003", "User ID must be present", "user_id", "required"),
    ComplianceRule("R004", "Model must be specified", "model", "required"),
    ComplianceRule("R005", "Action must be valid", "action", "allowed_values",
                   ["request", "response", "error"]),
    ComplianceRule("R006", "Latency must be under 30s", "latency_ms", "max_value", 30000),
]


@dataclass
class Violation:
    """A compliance violation."""
    rule_id: str
    description: str
    entry_id: str
    details: str


class ComplianceChecker:
    """Checks audit logs against compliance rules."""

    def __init__(self, rules: list[ComplianceRule] = None):
        self.rules = rules or DEFAULT_RULES

    def check_entry(self, entry: dict) -> list[Violation]:
        """Check a single log entry against all rules."""
        violations = []
        for rule in self.rules:
            violation = self._apply_rule(rule, entry)
            if violation:
                violations.append(violation)
        return violations

    def _apply_rule(self, rule: ComplianceRule, entry: dict) -> Optional[Violation]:
        """Apply a single rule to an entry."""
        value = entry.get(rule.field)
        entry_id = entry.get("event_id", "unknown")

        if rule.check == "required" and not value:
            return Violation(rule.rule_id, rule.description, entry_id,
                             f"Field '{rule.field}' is missing or empty")
        elif rule.check == "max_value" and value is not None:
            if isinstance(value, (int, float)) and value > rule.value:
                return Violation(rule.rule_id, rule.description, entry_id,
                                 f"Field '{rule.field}' value {value} exceeds max {rule.value}")
        elif rule.check == "allowed_values" and value is not None:
            if value not in rule.value:
                return Violation(rule.rule_id, rule.description, entry_id,
                                 f"Field '{rule.field}' value '{value}' not in {rule.value}")
        return None

    def check_log_file(self, filepath: str) -> list[Violation]:
        """Check all entries in a log file."""
        violations = []
        path = Path(filepath)
        if not path.exists():
            print(f"[ComplianceChecker] File not found: {filepath}")
            return violations
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                violations.extend(self.check_entry(entry))
        return violations


def demo():
    """Demonstrate compliance checking."""
    checker = ComplianceChecker()

    # Test with valid entry
    valid_entry = {
        "event_id": "abc-123", "timestamp": "2024-01-01T00:00:00Z",
        "user_id": "user_1", "model": "gpt-4", "action": "request",
        "latency_ms": 150.0,
    }
    violations = checker.check_entry(valid_entry)
    print(f"Valid entry violations: {len(violations)}")

    # Test with invalid entry
    invalid_entry = {
        "event_id": "", "timestamp": "2024-01-01T00:00:00Z",
        "user_id": "", "model": "", "action": "invalid_action",
        "latency_ms": 50000,
    }
    violations = checker.check_entry(invalid_entry)
    print(f"Invalid entry violations: {len(violations)}")
    for v in violations:
        print(f"  [{v.rule_id}] {v.details}")


if __name__ == "__main__":
    demo()
