"""
Privacy Audit — Compliance Checking & Reporting
=================================================
Scans datasets and logs for PII exposure, generates compliance reports,
and tracks data handling violations against configurable policies.

Usage:
    python privacy_audit.py
"""

import time
from dataclasses import dataclass, field
from enum import Enum

from pii_detector import PIIDetector, PIIType


# ---------------------------------------------------------------------------
# Audit policy configuration
# ---------------------------------------------------------------------------
class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Map PII types to risk levels
PII_RISK_MAP: dict[PIIType, RiskLevel] = {
    PIIType.EMAIL: RiskLevel.MEDIUM,
    PIIType.PHONE_US: RiskLevel.MEDIUM,
    PIIType.SSN: RiskLevel.CRITICAL,
    PIIType.CREDIT_CARD: RiskLevel.CRITICAL,
    PIIType.IP_ADDRESS: RiskLevel.LOW,
    PIIType.DATE_OF_BIRTH: RiskLevel.HIGH,
    PIIType.PERSON_NAME: RiskLevel.LOW,
}


@dataclass
class AuditPolicy:
    name: str
    max_pii_per_record: int = 0          # 0 = no PII allowed
    allowed_pii_types: list[PIIType] = field(default_factory=list)
    block_critical: bool = True           # block records with critical PII
    require_anonymization: bool = True


# Pre-built policies
STRICT_POLICY = AuditPolicy(
    name="strict",
    max_pii_per_record=0,
    allowed_pii_types=[],
    block_critical=True,
)

MODERATE_POLICY = AuditPolicy(
    name="moderate",
    max_pii_per_record=2,
    allowed_pii_types=[PIIType.PERSON_NAME, PIIType.EMAIL],
    block_critical=True,
)


# ---------------------------------------------------------------------------
# Audit findings
# ---------------------------------------------------------------------------
@dataclass
class AuditFinding:
    record_index: int
    text_preview: str
    pii_type: PIIType
    risk_level: RiskLevel
    violation: str


@dataclass
class AuditReport:
    policy: AuditPolicy
    total_records: int
    clean_records: int
    violated_records: int
    findings: list[AuditFinding]
    scan_duration_ms: float
    timestamp: float = field(default_factory=time.time)

    @property
    def compliance_rate(self) -> float:
        return self.clean_records / self.total_records if self.total_records else 0

    def summary(self) -> str:
        lines = [
            f"Policy: {self.policy.name}",
            f"Records scanned: {self.total_records}",
            f"Clean: {self.clean_records} | Violated: {self.violated_records}",
            f"Compliance rate: {self.compliance_rate:.1%}",
            f"Scan time: {self.scan_duration_ms:.1f} ms",
            f"Findings: {len(self.findings)}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Auditor
# ---------------------------------------------------------------------------
class PrivacyAuditor:
    def __init__(self, policy: AuditPolicy):
        self.policy = policy
        self.detector = PIIDetector()

    def audit_dataset(self, records: list[str]) -> AuditReport:
        """Scan a list of text records against the audit policy."""
        start = time.perf_counter()
        findings: list[AuditFinding] = []
        violated_indices: set[int] = set()

        for idx, text in enumerate(records):
            entities = self.detector.scan(text)

            # Check: any PII at all when policy forbids it
            if self.policy.max_pii_per_record == 0 and entities:
                violated_indices.add(idx)
                for e in entities:
                    findings.append(AuditFinding(
                        record_index=idx,
                        text_preview=text[:50],
                        pii_type=e.pii_type,
                        risk_level=PII_RISK_MAP.get(e.pii_type, RiskLevel.MEDIUM),
                        violation="PII found in zero-tolerance policy",
                    ))
                continue

            # Check: too many PII entities
            if len(entities) > self.policy.max_pii_per_record:
                violated_indices.add(idx)
                findings.append(AuditFinding(
                    record_index=idx,
                    text_preview=text[:50],
                    pii_type=entities[0].pii_type,
                    risk_level=RiskLevel.MEDIUM,
                    violation=f"Exceeds max PII limit ({len(entities)} > {self.policy.max_pii_per_record})",
                ))

            # Check: disallowed PII types
            for e in entities:
                if e.pii_type not in self.policy.allowed_pii_types:
                    violated_indices.add(idx)
                    findings.append(AuditFinding(
                        record_index=idx,
                        text_preview=text[:50],
                        pii_type=e.pii_type,
                        risk_level=PII_RISK_MAP.get(e.pii_type, RiskLevel.MEDIUM),
                        violation=f"Disallowed PII type: {e.pii_type.value}",
                    ))

            # Check: critical PII when blocked
            if self.policy.block_critical:
                for e in entities:
                    if PII_RISK_MAP.get(e.pii_type) == RiskLevel.CRITICAL:
                        violated_indices.add(idx)
                        findings.append(AuditFinding(
                            record_index=idx,
                            text_preview=text[:50],
                            pii_type=e.pii_type,
                            risk_level=RiskLevel.CRITICAL,
                            violation=f"Critical PII detected: {e.pii_type.value}",
                        ))

        duration = (time.perf_counter() - start) * 1000
        return AuditReport(
            policy=self.policy,
            total_records=len(records),
            clean_records=len(records) - len(violated_indices),
            violated_records=len(violated_indices),
            findings=findings,
            scan_duration_ms=duration,
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Simulated dataset (e.g., LLM training data or inference logs)
    dataset = [
        "The capital of France is Paris.",
        "Contact john.doe@example.com for more details.",
        "Patient SSN: 987-65-4321, DOB: 05/12/1990.",
        "Order shipped to 123 Main St. Credit card: 4111-1111-1111-1111.",
        "The model achieved 95% accuracy on the test set.",
        "User IP 10.0.0.55 made 200 requests in the last hour.",
        "Dr. Alice Wang presented results at the conference.",
        "Refund issued. Call Mr. Bob Lee at 555-234-5678.",
    ]

    print(f"\n{'='*65}")
    print(f"  Privacy Audit Demo")
    print(f"{'='*65}\n")

    for policy in [STRICT_POLICY, MODERATE_POLICY]:
        auditor = PrivacyAuditor(policy)
        report = auditor.audit_dataset(dataset)

        print(f"  --- {policy.name.upper()} POLICY ---")
        print(f"  {report.summary()}")
        print()
        if report.findings:
            for f in report.findings[:6]:  # show first 6
                icon = "!!" if f.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) else ".."
                print(f"    [{icon}] record[{f.record_index}] {f.pii_type.value}: {f.violation}")
            if len(report.findings) > 6:
                print(f"    ... and {len(report.findings) - 6} more findings")
        print()
