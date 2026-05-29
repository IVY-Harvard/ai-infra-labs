"""Rule-Based Policy Engine for AI Governance Compliance."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class Action(Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    REVIEW = "REVIEW"


@dataclass
class PolicyRule:
    name: str
    description: str
    condition: Callable[[dict], bool]
    action: Action
    priority: int = 0


@dataclass
class PolicyResult:
    rule_name: str
    action: Action
    reason: str


class PolicyEngine:
    """Evaluates requests against a set of ordered policy rules."""

    def __init__(self):
        self.rules: list[PolicyRule] = []

    def add_rule(self, rule: PolicyRule):
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def evaluate(self, context: dict) -> list[PolicyResult]:
        results = []
        for rule in self.rules:
            if rule.condition(context):
                results.append(PolicyResult(rule.name, rule.action, rule.description))
        return results

    def decide(self, context: dict) -> Action:
        """Return final action: DENY > REVIEW > ALLOW."""
        results = self.evaluate(context)
        if not results:
            return Action.ALLOW
        if any(r.action == Action.DENY for r in results):
            return Action.DENY
        if any(r.action == Action.REVIEW for r in results):
            return Action.REVIEW
        return Action.ALLOW


def build_default_engine() -> PolicyEngine:
    engine = PolicyEngine()
    engine.add_rule(PolicyRule(
        "no-pii-without-consent", "PII requires explicit user consent",
        lambda ctx: ctx.get("has_pii") and not ctx.get("consent_granted"),
        Action.DENY, priority=10
    ))
    engine.add_rule(PolicyRule(
        "high-risk-needs-review", "High-risk models need human review",
        lambda ctx: ctx.get("risk_level") in ("HIGH", "CRITICAL"),
        Action.REVIEW, priority=5
    ))
    engine.add_rule(PolicyRule(
        "rate-limit-exceeded", "Deny if rate limit exceeded",
        lambda ctx: ctx.get("requests_per_minute", 0) > 100,
        Action.DENY, priority=8
    ))
    engine.add_rule(PolicyRule(
        "no-autonomous-safety", "Autonomous safety models need override",
        lambda ctx: ctx.get("autonomy") == "autonomous" and ctx.get("domain") == "safety",
        Action.REVIEW, priority=7
    ))
    return engine


if __name__ == "__main__":
    engine = build_default_engine()
    test_cases = [
        {"has_pii": True, "consent_granted": False, "risk_level": "LOW"},
        {"has_pii": False, "risk_level": "HIGH", "requests_per_minute": 50},
        {"has_pii": False, "risk_level": "LOW", "requests_per_minute": 200},
        {"autonomy": "autonomous", "domain": "safety", "risk_level": "CRITICAL"},
    ]
    for i, ctx in enumerate(test_cases, 1):
        decision = engine.decide(ctx)
        triggered = engine.evaluate(ctx)
        print(f"Case {i}: {decision.value}")
        for r in triggered:
            print(f"  - [{r.action.value}] {r.rule_name}: {r.reason}")
