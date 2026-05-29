"""
Data Anonymizer — Masking and Pseudonymization
================================================
Transforms text containing PII into safe versions using multiple strategies:
redaction, masking, hashing, and consistent pseudonymization.

Usage:
    python data_anonymizer.py
"""

import hashlib
import re
from dataclasses import dataclass
from enum import Enum

from pii_detector import PIIDetector, PIIEntity, PIIType


# ---------------------------------------------------------------------------
# Anonymization strategies
# ---------------------------------------------------------------------------
class Strategy(Enum):
    REDACT = "redact"           # Replace with [REDACTED]
    MASK = "mask"               # Partial masking: jo***@***.com
    HASH = "hash"               # SHA-256 hash (one-way)
    PSEUDONYMIZE = "pseudonym"  # Consistent fake replacement


# ---------------------------------------------------------------------------
# Pseudonym generators (deterministic per input for consistency)
# ---------------------------------------------------------------------------
_FAKE_NAMES = ["Alex Johnson", "Sam Rivera", "Jordan Lee", "Taylor Kim", "Casey Chen"]
_FAKE_EMAILS = ["user1@anon.local", "user2@anon.local", "user3@anon.local"]
_FAKE_PHONES = ["(000) 000-0001", "(000) 000-0002", "(000) 000-0003"]


def _pseudonym_for(value: str, pool: list[str]) -> str:
    """Pick a consistent pseudonym based on hash of original value."""
    idx = int(hashlib.md5(value.encode()).hexdigest(), 16) % len(pool)
    return pool[idx]


# ---------------------------------------------------------------------------
# Anonymizer
# ---------------------------------------------------------------------------
@dataclass
class AnonymizationResult:
    original: str
    anonymized: str
    entities_found: int
    strategy_used: Strategy


class DataAnonymizer:
    def __init__(self, strategy: Strategy = Strategy.REDACT):
        self.strategy = strategy
        self.detector = PIIDetector()

    def anonymize(self, text: str) -> AnonymizationResult:
        """Anonymize all PII in text using the configured strategy."""
        entities = self.detector.scan(text)
        if not entities:
            return AnonymizationResult(text, text, 0, self.strategy)

        # Process replacements from end to start to preserve indices
        result = text
        for entity in reversed(entities):
            replacement = self._get_replacement(entity)
            result = result[:entity.start] + replacement + result[entity.end:]

        return AnonymizationResult(
            original=text,
            anonymized=result,
            entities_found=len(entities),
            strategy_used=self.strategy,
        )

    def anonymize_batch(self, texts: list[str]) -> list[AnonymizationResult]:
        """Anonymize a batch of texts."""
        return [self.anonymize(t) for t in texts]

    def _get_replacement(self, entity: PIIEntity) -> str:
        if self.strategy == Strategy.REDACT:
            return f"[{entity.pii_type.value.upper()}_REDACTED]"

        elif self.strategy == Strategy.MASK:
            return self._mask(entity)

        elif self.strategy == Strategy.HASH:
            h = hashlib.sha256(entity.value.encode()).hexdigest()[:12]
            return f"[hash:{h}]"

        elif self.strategy == Strategy.PSEUDONYMIZE:
            return self._pseudonymize(entity)

        return "[UNKNOWN]"

    def _mask(self, entity: PIIEntity) -> str:
        val = entity.value
        if entity.pii_type == PIIType.EMAIL:
            parts = val.split("@")
            return parts[0][:2] + "***@***." + parts[1].split(".")[-1]
        elif entity.pii_type == PIIType.SSN:
            return "***-**-" + val[-4:]
        elif entity.pii_type == PIIType.CREDIT_CARD:
            return "****-****-****-" + val[-4:]
        elif entity.pii_type == PIIType.PHONE_US:
            return "(***) ***-" + val[-4:]
        else:
            # Generic: show first 2 and last 2 chars
            if len(val) > 4:
                return val[:2] + "*" * (len(val) - 4) + val[-2:]
            return "****"

    def _pseudonymize(self, entity: PIIEntity) -> str:
        if entity.pii_type == PIIType.PERSON_NAME:
            return _pseudonym_for(entity.value, _FAKE_NAMES)
        elif entity.pii_type == PIIType.EMAIL:
            return _pseudonym_for(entity.value, _FAKE_EMAILS)
        elif entity.pii_type == PIIType.PHONE_US:
            return _pseudonym_for(entity.value, _FAKE_PHONES)
        else:
            # Fallback to hash for types without pools
            h = hashlib.sha256(entity.value.encode()).hexdigest()[:8]
            return f"[pseudo:{h}]"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "Dear Mr. John Smith, your account (SSN: 123-45-6789) has been "
        "charged $500. Contact us at support@company.com or 555-867-5309. "
        "Your IP: 192.168.1.42."
    )

    print(f"\n{'='*65}")
    print(f"  Data Anonymizer Demo")
    print(f"{'='*65}\n")
    print(f"  Original:\n    {sample}\n")

    for strategy in Strategy:
        anon = DataAnonymizer(strategy=strategy)
        result = anon.anonymize(sample)
        print(f"  Strategy: {strategy.value}")
        print(f"    {result.anonymized}")
        print(f"    ({result.entities_found} entities processed)\n")
