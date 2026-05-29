"""
PII Detector — Regex + Pattern-Based Detection
================================================
Scans text for personally identifiable information using regular expressions
and keyword heuristics. In production, pair with an NER model for higher recall.

Usage:
    python pii_detector.py
"""

import re
from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# PII entity types
# ---------------------------------------------------------------------------
class PIIType(Enum):
    EMAIL = "email"
    PHONE_US = "phone_us"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    DATE_OF_BIRTH = "date_of_birth"
    PERSON_NAME = "person_name"  # heuristic-based


@dataclass
class PIIEntity:
    pii_type: PIIType
    value: str
    start: int
    end: int
    confidence: float  # 0.0 - 1.0

    def __str__(self):
        masked = self.value[:2] + "*" * (len(self.value) - 4) + self.value[-2:]
        return f"{self.pii_type.value}: {masked} (pos {self.start}-{self.end}, conf={self.confidence:.0%})"


# ---------------------------------------------------------------------------
# Detection patterns
# ---------------------------------------------------------------------------
PII_PATTERNS: list[tuple[PIIType, str, float]] = [
    (PIIType.EMAIL, r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", 0.95),
    (PIIType.PHONE_US, r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", 0.85),
    (PIIType.SSN, r"\b\d{3}-\d{2}-\d{4}\b", 0.95),
    (PIIType.CREDIT_CARD, r"\b(?:\d{4}[-\s]?){3}\d{4}\b", 0.90),
    (PIIType.IP_ADDRESS, r"\b(?:\d{1,3}\.){3}\d{1,3}\b", 0.80),
    (PIIType.DATE_OF_BIRTH, r"\b(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/(?:19|20)\d{2}\b", 0.75),
]

# Simple name heuristic — titles followed by capitalized words
NAME_PATTERN = r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------
class PIIDetector:
    def __init__(self, patterns: list[tuple[PIIType, str, float]] | None = None):
        self.patterns = patterns or PII_PATTERNS

    def scan(self, text: str) -> list[PIIEntity]:
        """Scan text and return all PII entities found."""
        entities: list[PIIEntity] = []

        for pii_type, pattern, confidence in self.patterns:
            for match in re.finditer(pattern, text):
                entities.append(PIIEntity(
                    pii_type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence,
                ))

        # Name heuristic
        for match in re.finditer(NAME_PATTERN, text):
            entities.append(PIIEntity(
                pii_type=PIIType.PERSON_NAME,
                value=match.group(),
                start=match.start(),
                end=match.end(),
                confidence=0.70,
            ))

        # Sort by position
        entities.sort(key=lambda e: e.start)
        return entities

    def has_pii(self, text: str) -> bool:
        """Quick check: does text contain any PII?"""
        return len(self.scan(text)) > 0

    def summary(self, text: str) -> dict[str, int]:
        """Count PII entities by type."""
        entities = self.scan(text)
        counts: dict[str, int] = {}
        for e in entities:
            counts[e.pii_type.value] = counts.get(e.pii_type.value, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    detector = PIIDetector()

    sample_texts = [
        "Please contact Mr. John Smith at john.smith@example.com or 555-123-4567.",
        "My SSN is 123-45-6789 and my credit card is 4111-1111-1111-1111.",
        "Server logs show requests from 192.168.1.100 and 10.0.0.1.",
        "The patient Dr. Jane Doe was born on 03/15/1985.",
        "The weather in Tokyo is sunny today with clear skies.",
    ]

    print(f"\n{'='*60}")
    print(f"  PII Detector Demo")
    print(f"{'='*60}\n")

    for text in sample_texts:
        entities = detector.scan(text)
        has_pii = len(entities) > 0
        icon = "!!" if has_pii else "ok"
        print(f"  [{icon}] \"{text[:65]}\"")
        if entities:
            for e in entities:
                print(f"        -> {e}")
        else:
            print(f"        -> No PII detected")
        print()
