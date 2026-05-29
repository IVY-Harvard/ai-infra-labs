"""
API Key Manager — Secure Key Lifecycle
========================================
Manages creation, rotation, revocation, and validation of API keys
for accessing AI model endpoints.

Usage:
    python api_key_manager.py
"""

import hashlib
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Key metadata
# ---------------------------------------------------------------------------
class KeyStatus(Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    ROTATED = "rotated"  # superseded by a newer key


@dataclass
class APIKey:
    key_id: str
    owner: str
    prefix: str             # first 8 chars shown to user (e.g., "sk-abc123...")
    hashed_key: str         # stored hash — never store plaintext
    scopes: list[str]       # e.g., ["model:invoke", "model:read"]
    status: KeyStatus = KeyStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None  # None = no expiry
    last_used_at: float | None = None
    usage_count: int = 0

    @property
    def is_valid(self) -> bool:
        if self.status != KeyStatus.ACTIVE:
            return False
        if self.expires_at and time.time() > self.expires_at:
            return False
        return True


# ---------------------------------------------------------------------------
# Key store (in-memory; use encrypted DB in production)
# ---------------------------------------------------------------------------
class APIKeyManager:
    def __init__(self):
        self._keys: dict[str, APIKey] = {}  # key_id -> APIKey

    def create_key(
        self,
        owner: str,
        scopes: list[str],
        ttl_seconds: int | None = 86400 * 30,  # 30 days default
    ) -> tuple[str, APIKey]:
        """Create a new API key. Returns (plaintext_key, metadata)."""
        raw_key = f"sk-{secrets.token_urlsafe(32)}"
        key_id = f"kid-{secrets.token_hex(8)}"
        prefix = raw_key[:11] + "..."
        hashed = self._hash(raw_key)

        expires = time.time() + ttl_seconds if ttl_seconds else None

        api_key = APIKey(
            key_id=key_id,
            owner=owner,
            prefix=prefix,
            hashed_key=hashed,
            scopes=scopes,
            expires_at=expires,
        )
        self._keys[key_id] = api_key
        return raw_key, api_key

    def validate_key(self, raw_key: str) -> APIKey | None:
        """Validate a plaintext key. Returns metadata if valid, None otherwise."""
        hashed = self._hash(raw_key)
        for key in self._keys.values():
            if key.hashed_key == hashed:
                if not key.is_valid:
                    return None
                key.last_used_at = time.time()
                key.usage_count += 1
                return key
        return None

    def revoke_key(self, key_id: str) -> bool:
        """Revoke a key immediately."""
        if key_id in self._keys:
            self._keys[key_id].status = KeyStatus.REVOKED
            return True
        return False

    def rotate_key(self, key_id: str, new_ttl: int | None = 86400 * 30) -> tuple[str, APIKey] | None:
        """Rotate: revoke old key and issue a new one with same scopes."""
        old = self._keys.get(key_id)
        if not old:
            return None
        old.status = KeyStatus.ROTATED
        return self.create_key(old.owner, old.scopes, new_ttl)

    def list_keys(self, owner: str | None = None) -> list[APIKey]:
        """List keys, optionally filtered by owner."""
        keys = list(self._keys.values())
        if owner:
            keys = [k for k in keys if k.owner == owner]
        return keys

    def cleanup_expired(self) -> int:
        """Mark expired keys. Returns count of newly expired keys."""
        count = 0
        for key in self._keys.values():
            if key.status == KeyStatus.ACTIVE and key.expires_at:
                if time.time() > key.expires_at:
                    key.status = KeyStatus.EXPIRED
                    count += 1
        return count

    @staticmethod
    def _hash(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mgr = APIKeyManager()

    print(f"\n{'='*60}")
    print(f"  API Key Manager Demo")
    print(f"{'='*60}\n")

    # Create keys for different users
    key1_raw, key1_meta = mgr.create_key("alice", ["model:invoke", "model:read"])
    key2_raw, key2_meta = mgr.create_key("bob", ["model:invoke"])
    key3_raw, key3_meta = mgr.create_key("carol", ["model:invoke", "model:deploy"], ttl_seconds=0)

    print(f"  Created key for alice: {key1_meta.prefix}  scopes={key1_meta.scopes}")
    print(f"  Created key for bob:   {key2_meta.prefix}  scopes={key2_meta.scopes}")
    print(f"  Created key for carol: {key3_meta.prefix}  (already expired)")

    # Validate
    print(f"\n  --- Validation ---")
    result = mgr.validate_key(key1_raw)
    print(f"  alice key valid: {result is not None}")

    result = mgr.validate_key(key3_raw)
    print(f"  carol key valid (expired): {result is not None}")

    result = mgr.validate_key("sk-fake-key-that-doesnt-exist")
    print(f"  fake key valid: {result is not None}")

    # Revoke
    print(f"\n  --- Revocation ---")
    mgr.revoke_key(key2_meta.key_id)
    result = mgr.validate_key(key2_raw)
    print(f"  bob key after revoke: {result is not None}")

    # Rotate
    print(f"\n  --- Rotation ---")
    new_raw, new_meta = mgr.rotate_key(key1_meta.key_id)
    old_valid = mgr.validate_key(key1_raw)
    new_valid = mgr.validate_key(new_raw)
    print(f"  alice old key valid: {old_valid is not None}")
    print(f"  alice new key valid: {new_valid is not None}")
    print(f"  alice new prefix: {new_meta.prefix}")

    # List
    print(f"\n  --- All Keys ---")
    for k in mgr.list_keys():
        print(f"    {k.prefix}  owner={k.owner}  status={k.status.value}")
    print()
