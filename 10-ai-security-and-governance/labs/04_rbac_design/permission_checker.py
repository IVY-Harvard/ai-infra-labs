"""
Permission Checker — Policy Evaluation Engine
===============================================
Evaluates access requests against the RBAC model.
Implements default-deny, audit logging, and context-aware checks.

Usage:
    python permission_checker.py
"""

import time
from dataclasses import dataclass, field
from enum import Enum

from rbac_model import (
    Action,
    ResourceType,
    Role,
    User,
    ROLE_ML_ENGINEER,
    ROLE_VIEWER,
    ROLE_ADMIN,
    ROLE_AUDITOR,
    Permission,
)


# ---------------------------------------------------------------------------
# Access decision
# ---------------------------------------------------------------------------
class Decision(Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"


@dataclass
class AccessRequest:
    user: User
    resource: ResourceType
    action: Action
    resource_id: str = ""        # specific resource instance
    context: dict = field(default_factory=dict)  # extra context (IP, time, etc.)


@dataclass
class AccessDecision:
    decision: Decision
    request: AccessRequest
    reason: str
    timestamp: float = field(default_factory=time.time)

    def __str__(self):
        icon = "ALLOW" if self.decision == Decision.ALLOW else " DENY"
        return (
            f"[{icon}] {self.request.user.name} -> "
            f"{self.request.resource.value}:{self.request.action.value} "
            f"| {self.reason}"
        )


# ---------------------------------------------------------------------------
# Audit log (in-memory; replace with persistent store in production)
# ---------------------------------------------------------------------------
audit_log: list[AccessDecision] = []


# ---------------------------------------------------------------------------
# Policy checker
# ---------------------------------------------------------------------------
class PermissionChecker:
    """Evaluates access requests using default-deny semantics."""

    def __init__(self, enable_audit: bool = True):
        self.enable_audit = enable_audit

    def check(self, request: AccessRequest) -> AccessDecision:
        """Main evaluation — returns ALLOW or DENY with reason."""
        # 1. Check if user has any roles at all
        if not request.user.roles:
            return self._deny(request, "user has no roles assigned")

        # 2. Aggregate permissions from all user roles
        all_perms = request.user.all_permissions()
        required = Permission(request.resource, request.action)

        # 3. Check if the required permission exists
        if required not in all_perms:
            return self._deny(
                request,
                f"no role grants {required}",
            )

        # 4. Context-based restrictions (example: time-based)
        if self._is_restricted_by_context(request):
            return self._deny(request, "context restriction (maintenance window)")

        # 5. Explicit allow
        return self._allow(request, f"granted via role permissions")

    def check_bulk(self, requests: list[AccessRequest]) -> list[AccessDecision]:
        """Check multiple requests at once."""
        return [self.check(r) for r in requests]

    # --- internal ---
    def _allow(self, req: AccessRequest, reason: str) -> AccessDecision:
        decision = AccessDecision(Decision.ALLOW, req, reason)
        if self.enable_audit:
            audit_log.append(decision)
        return decision

    def _deny(self, req: AccessRequest, reason: str) -> AccessDecision:
        decision = AccessDecision(Decision.DENY, req, reason)
        if self.enable_audit:
            audit_log.append(decision)
        return decision

    def _is_restricted_by_context(self, req: AccessRequest) -> bool:
        """Example context check: block destructive ops during maintenance."""
        if req.context.get("maintenance_mode") and req.action in (
            Action.DELETE, Action.DEPLOY, Action.UPDATE
        ):
            return True
        return False


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    checker = PermissionChecker()

    # Sample users
    alice = User(user_id="u-001", name="Alice", roles=[ROLE_ML_ENGINEER])
    bob = User(user_id="u-002", name="Bob", roles=[ROLE_VIEWER])
    carol = User(user_id="u-003", name="Carol", roles=[ROLE_ADMIN])
    dave = User(user_id="u-004", name="Dave", roles=[ROLE_AUDITOR])
    eve = User(user_id="u-005", name="Eve", roles=[])  # no roles

    test_requests = [
        AccessRequest(alice, ResourceType.MODEL, Action.DEPLOY, "model-gpt4"),
        AccessRequest(bob, ResourceType.MODEL, Action.DEPLOY, "model-gpt4"),
        AccessRequest(bob, ResourceType.MODEL, Action.READ, "model-gpt4"),
        AccessRequest(carol, ResourceType.DATASET, Action.DELETE, "ds-pii"),
        AccessRequest(dave, ResourceType.AUDIT_LOG, Action.READ),
        AccessRequest(dave, ResourceType.MODEL, Action.DELETE, "model-old"),
        AccessRequest(eve, ResourceType.MODEL, Action.READ, "model-gpt4"),
        # Context: maintenance mode blocks deploy
        AccessRequest(
            alice, ResourceType.MODEL, Action.DEPLOY, "model-v2",
            context={"maintenance_mode": True},
        ),
    ]

    print(f"\n{'='*65}")
    print(f"  Permission Checker — Access Decision Demo")
    print(f"{'='*65}\n")

    for req in test_requests:
        result = checker.check(req)
        icon = "+" if result.decision == Decision.ALLOW else "x"
        print(f"  [{icon}] {result}")

    print(f"\n  Audit log entries: {len(audit_log)}")
    print()
