"""API Key Manager."""
import hashlib
import secrets
import time
from dataclasses import dataclass


@dataclass
class APIKey:
    key_hash: str
    user: str
    created_at: float
    expires_at: float = 0
    active: bool = True


class APIKeyManager:
    def __init__(self):
        self.keys: dict[str, APIKey] = {}

    def create_key(self, user: str, ttl_days: int = 90) -> str:
        raw_key = f"sk-{secrets.token_hex(24)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        self.keys[key_hash] = APIKey(
            key_hash=key_hash, user=user,
            created_at=time.time(),
            expires_at=time.time() + ttl_days * 86400,
        )
        return raw_key

    def validate(self, raw_key: str) -> str | None:
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        record = self.keys.get(key_hash)
        if not record or not record.active:
            return None
        if record.expires_at and time.time() > record.expires_at:
            return None
        return record.user

    def revoke(self, raw_key: str):
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        if key_hash in self.keys:
            self.keys[key_hash].active = False
