"""
RBAC Model for AI Platforms
============================
Defines users, roles, and permissions for controlling access to ML resources
(models, datasets, endpoints, experiments).

Usage:
    python rbac_model.py
"""

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Resource types in an AI platform
# ---------------------------------------------------------------------------
class ResourceType(Enum):
    MODEL = "model"
    DATASET = "dataset"
    ENDPOINT = "endpoint"
    EXPERIMENT = "experiment"
    AUDIT_LOG = "audit_log"
    API_KEY = "api_key"


# ---------------------------------------------------------------------------
# Actions that can be performed
# ---------------------------------------------------------------------------
class Action(Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    DEPLOY = "deploy"
    INVOKE = "invoke"      # call an inference endpoint
    EXPORT = "export"      # export data outside the platform


# ---------------------------------------------------------------------------
# Core data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Permission:
    resource: ResourceType
    action: Action

    def __str__(self):
        return f"{self.resource.value}:{self.action.value}"


@dataclass
class Role:
    name: str
    description: str
    permissions: set[Permission] = field(default_factory=set)

    def has_permission(self, resource: ResourceType, action: Action) -> bool:
        return Permission(resource, action) in self.permissions


@dataclass
class User:
    user_id: str
    name: str
    roles: list[Role] = field(default_factory=list)

    def all_permissions(self) -> set[Permission]:
        perms: set[Permission] = set()
        for role in self.roles:
            perms.update(role.permissions)
        return perms

    def has_role(self, role_name: str) -> bool:
        return any(r.name == role_name for r in self.roles)


# ---------------------------------------------------------------------------
# Pre-built roles for an AI platform
# ---------------------------------------------------------------------------
ROLE_VIEWER = Role(
    name="viewer",
    description="Read-only access to models and experiments",
    permissions={
        Permission(ResourceType.MODEL, Action.READ),
        Permission(ResourceType.EXPERIMENT, Action.READ),
        Permission(ResourceType.DATASET, Action.READ),
    },
)

ROLE_ML_ENGINEER = Role(
    name="ml_engineer",
    description="Train and deploy models",
    permissions={
        Permission(ResourceType.MODEL, Action.CREATE),
        Permission(ResourceType.MODEL, Action.READ),
        Permission(ResourceType.MODEL, Action.UPDATE),
        Permission(ResourceType.MODEL, Action.DEPLOY),
        Permission(ResourceType.EXPERIMENT, Action.CREATE),
        Permission(ResourceType.EXPERIMENT, Action.READ),
        Permission(ResourceType.DATASET, Action.READ),
        Permission(ResourceType.ENDPOINT, Action.CREATE),
        Permission(ResourceType.ENDPOINT, Action.INVOKE),
    },
)

ROLE_DATA_STEWARD = Role(
    name="data_steward",
    description="Manage datasets and privacy controls",
    permissions={
        Permission(ResourceType.DATASET, Action.CREATE),
        Permission(ResourceType.DATASET, Action.READ),
        Permission(ResourceType.DATASET, Action.UPDATE),
        Permission(ResourceType.DATASET, Action.DELETE),
        Permission(ResourceType.DATASET, Action.EXPORT),
        Permission(ResourceType.AUDIT_LOG, Action.READ),
    },
)

ROLE_ADMIN = Role(
    name="admin",
    description="Full platform access",
    permissions={Permission(rt, act) for rt in ResourceType for act in Action},
)

ROLE_AUDITOR = Role(
    name="auditor",
    description="Read-only access to logs and configurations",
    permissions={
        Permission(ResourceType.AUDIT_LOG, Action.READ),
        Permission(ResourceType.MODEL, Action.READ),
        Permission(ResourceType.ENDPOINT, Action.READ),
        Permission(ResourceType.API_KEY, Action.READ),
    },
)

ALL_ROLES = [ROLE_VIEWER, ROLE_ML_ENGINEER, ROLE_DATA_STEWARD, ROLE_ADMIN, ROLE_AUDITOR]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"  RBAC Model — AI Platform Roles & Permissions")
    print(f"{'='*55}\n")

    for role in ALL_ROLES:
        perms = sorted(str(p) for p in role.permissions)
        print(f"  Role: {role.name}")
        print(f"    {role.description}")
        print(f"    Permissions ({len(perms)}):")
        for p in perms:
            print(f"      - {p}")
        print()

    # Create a sample user
    alice = User(user_id="u-001", name="Alice", roles=[ROLE_ML_ENGINEER])
    print(f"  User: {alice.name} (roles: {[r.name for r in alice.roles]})")
    print(f"    Can deploy models? {alice.roles[0].has_permission(ResourceType.MODEL, Action.DEPLOY)}")
    print(f"    Can delete datasets? {alice.roles[0].has_permission(ResourceType.DATASET, Action.DELETE)}")
    print()
