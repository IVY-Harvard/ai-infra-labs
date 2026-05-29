"""AI Model Risk Assessment Tool - Evaluates model risk levels."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class ModelMetadata:
    name: str
    use_case: str
    data_sensitivity: str  # public, internal, confidential, restricted
    user_impact: str       # informational, operational, financial, safety
    autonomy_level: str    # human-in-loop, human-on-loop, autonomous
    data_volume: int       # number of records processed
    has_pii: bool = False
    has_bias_testing: bool = False


SENSITIVITY_SCORES = {"public": 1, "internal": 2, "confidential": 3, "restricted": 4}
IMPACT_SCORES = {"informational": 1, "operational": 2, "financial": 3, "safety": 4}
AUTONOMY_SCORES = {"human-in-loop": 1, "human-on-loop": 2, "autonomous": 3}


def assess_risk(model: ModelMetadata) -> dict:
    """Calculate composite risk score and determine risk level."""
    score = 0
    score += SENSITIVITY_SCORES.get(model.data_sensitivity, 2) * 2
    score += IMPACT_SCORES.get(model.user_impact, 2) * 3
    score += AUTONOMY_SCORES.get(model.autonomy_level, 1) * 2
    if model.has_pii:
        score += 3
    if not model.has_bias_testing:
        score += 2
    if model.data_volume > 100000:
        score += 1

    if score <= 8:
        level = RiskLevel.LOW
    elif score <= 14:
        level = RiskLevel.MEDIUM
    elif score <= 20:
        level = RiskLevel.HIGH
    else:
        level = RiskLevel.CRITICAL

    mitigations = []
    if not model.has_bias_testing:
        mitigations.append("Conduct bias and fairness testing")
    if model.has_pii:
        mitigations.append("Implement data anonymization pipeline")
    if model.autonomy_level == "autonomous":
        mitigations.append("Add human override mechanism")

    return {
        "model": model.name,
        "score": score,
        "level": level.value,
        "mitigations": mitigations,
    }


if __name__ == "__main__":
    samples = [
        ModelMetadata("search-ranker", "Search ranking", "public", "informational",
                      "human-in-loop", 50000, False, True),
        ModelMetadata("credit-scorer", "Credit decisions", "restricted", "financial",
                      "human-on-loop", 500000, True, False),
        ModelMetadata("content-moderator", "Safety filtering", "confidential", "safety",
                      "autonomous", 1000000, True, True),
    ]
    for m in samples:
        result = assess_risk(m)
        print(f"[{result['level']:>8}] {result['model']} (score={result['score']})")
        for fix in result["mitigations"]:
            print(f"           -> {fix}")
