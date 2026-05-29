"""Tests for auth module."""
import pytest
from src.auth.rbac import RBACManager
from src.auth.api_key import APIKeyManager
from src.auth.quota import QuotaManager


class TestRBAC:
    def test_default_role(self):
        rbac = RBACManager()
        assert rbac.check_permission("unknown_user", "chat") is True
        assert rbac.check_permission("unknown_user", "manage") is False

    def test_admin_permissions(self):
        rbac = RBACManager()
        rbac.assign_role("admin_user", "admin")
        assert rbac.check_permission("admin_user", "chat") is True
        assert rbac.check_permission("admin_user", "manage") is True


class TestAPIKey:
    def test_create_and_validate(self):
        mgr = APIKeyManager()
        key = mgr.create_key("test_user")
        assert key.startswith("sk-")
        assert mgr.validate(key) == "test_user"

    def test_invalid_key(self):
        mgr = APIKeyManager()
        assert mgr.validate("sk-invalid") is None

    def test_revoke(self):
        mgr = APIKeyManager()
        key = mgr.create_key("user1")
        mgr.revoke(key)
        assert mgr.validate(key) is None


class TestQuota:
    def test_consume_within_limit(self):
        qm = QuotaManager(default_daily=1000)
        assert qm.consume("user1", 500) is True
        assert qm.consume("user1", 400) is True

    def test_exceed_limit(self):
        qm = QuotaManager(default_daily=1000)
        assert qm.consume("user1", 1001) is False
