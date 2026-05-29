"""RBAC - Role-Based Access Control."""
from dataclasses import dataclass, field


@dataclass
class Role:
    name: str
    permissions: set = field(default_factory=set)


class RBACManager:
    def __init__(self):
        self.roles: dict[str, Role] = {
            "admin": Role("admin", {"chat", "manage", "audit", "deploy"}),
            "developer": Role("developer", {"chat", "audit"}),
            "viewer": Role("viewer", {"chat"}),
        }
        self.user_roles: dict[str, str] = {}

    def assign_role(self, user: str, role: str):
        if role not in self.roles:
            raise ValueError(f"Unknown role: {role}")
        self.user_roles[user] = role

    def check_permission(self, user: str, permission: str) -> bool:
        role_name = self.user_roles.get(user, "viewer")
        role = self.roles.get(role_name)
        return role is not None and permission in role.permissions

    def get_user_role(self, user: str) -> str:
        return self.user_roles.get(user, "viewer")
