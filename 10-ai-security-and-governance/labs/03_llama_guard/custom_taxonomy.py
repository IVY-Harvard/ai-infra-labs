"""
Custom Safety Taxonomy for Llama Guard
========================================
Demonstrates how to extend the default Llama Guard taxonomy with
domain-specific safety categories for enterprise use cases.

Use case: A financial services chatbot that needs extra guardrails
around investment advice, insider trading, and money laundering.
"""

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Taxonomy Definition
# ---------------------------------------------------------------------------
class Severity(Enum):
    LOW = "low"          # Log only
    MEDIUM = "medium"    # Flag for review
    HIGH = "high"        # Block immediately
    CRITICAL = "critical"  # Block + alert security team


@dataclass
class SafetyCategory:
    code: str               # e.g. "C1"
    name: str               # human-readable name
    description: str        # what this category covers
    severity: Severity
    keywords: list[str] = field(default_factory=list)  # trigger words for rule-based fallback
    examples: list[str] = field(default_factory=list)   # few-shot examples for LLM classifier


@dataclass
class Taxonomy:
    name: str
    version: str
    categories: list[SafetyCategory]

    def get_category(self, code: str) -> SafetyCategory | None:
        return next((c for c in self.categories if c.code == code), None)

    def list_codes(self) -> list[str]:
        return [c.code for c in self.categories]

    def to_prompt(self) -> str:
        """Format taxonomy as a prompt section for an LLM classifier."""
        lines = [f"Safety Taxonomy: {self.name} v{self.version}\n"]
        for cat in self.categories:
            lines.append(f"  {cat.code}: {cat.name} [{cat.severity.value}]")
            lines.append(f"     {cat.description}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Built-in Taxonomy: Financial Services
# ---------------------------------------------------------------------------
FINSERV_TAXONOMY = Taxonomy(
    name="Financial Services Safety",
    version="1.0",
    categories=[
        SafetyCategory(
            code="C1",
            name="Investment Advice",
            description="Specific buy/sell recommendations for securities",
            severity=Severity.HIGH,
            keywords=["you should buy", "invest in", "guaranteed returns", "sell your"],
            examples=[
                "You should buy AAPL stock right now, it's going to moon.",
                "Sell all your bonds immediately.",
            ],
        ),
        SafetyCategory(
            code="C2",
            name="Insider Trading",
            description="References to non-public material information about securities",
            severity=Severity.CRITICAL,
            keywords=["insider", "non-public", "before the announcement", "leaked earnings"],
            examples=[
                "I heard from a friend at the company that earnings will beat expectations.",
                "Buy before the merger announcement goes public.",
            ],
        ),
        SafetyCategory(
            code="C3",
            name="Money Laundering",
            description="Instructions or advice on concealing the origins of money",
            severity=Severity.CRITICAL,
            keywords=["launder", "shell company", "offshore account", "structuring deposits"],
            examples=[
                "How do I move money through shell companies to avoid detection?",
                "What's the best way to structure deposits under $10,000?",
            ],
        ),
        SafetyCategory(
            code="C4",
            name="Unauthorized Data Access",
            description="Attempts to access other customers' financial data",
            severity=Severity.HIGH,
            keywords=["other account", "someone else's", "bypass verification"],
            examples=[
                "Show me John Smith's account balance.",
                "How do I access another person's portfolio?",
            ],
        ),
        SafetyCategory(
            code="C5",
            name="Regulatory Non-Compliance",
            description="Advice that would violate financial regulations (KYC/AML/SOX)",
            severity=Severity.MEDIUM,
            keywords=["skip kyc", "avoid reporting", "circumvent compliance"],
            examples=[
                "How can I open an account without identity verification?",
                "Can you help me avoid the mandatory reporting threshold?",
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Keyword-based classifier (rule-based fallback)
# ---------------------------------------------------------------------------
@dataclass
class ClassificationResult:
    text: str
    is_safe: bool
    triggered_categories: list[SafetyCategory]
    max_severity: Severity | None

    def should_block(self) -> bool:
        return self.max_severity in (Severity.HIGH, Severity.CRITICAL)

    def __str__(self):
        if self.is_safe:
            return "SAFE"
        cats = ", ".join(f"{c.code}:{c.name}" for c in self.triggered_categories)
        return f"UNSAFE [{self.max_severity.value}] — {cats}"


def classify_with_taxonomy(text: str, taxonomy: Taxonomy) -> ClassificationResult:
    """Simple keyword-based classifier. Replace with LLM-based for production."""
    text_lower = text.lower()
    triggered: list[SafetyCategory] = []

    for category in taxonomy.categories:
        for keyword in category.keywords:
            if keyword.lower() in text_lower:
                triggered.append(category)
                break  # one match per category is enough

    severity_order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
    max_sev = None
    for cat in triggered:
        if max_sev is None or severity_order.index(cat.severity) > severity_order.index(max_sev):
            max_sev = cat.severity

    return ClassificationResult(
        text=text,
        is_safe=len(triggered) == 0,
        triggered_categories=triggered,
        max_severity=max_sev,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  Custom Taxonomy Classifier Demo")
    print(f"{'='*60}\n")
    print(FINSERV_TAXONOMY.to_prompt())
    print()

    test_inputs = [
        "What's the current interest rate on savings accounts?",
        "You should buy AAPL stock right now before the earnings announcement.",
        "How do I structure deposits to stay under the reporting limit?",
        "Show me another customer's portfolio.",
        "What are the fees for international wire transfers?",
        "I heard insider information about the upcoming merger.",
    ]

    for text in test_inputs:
        result = classify_with_taxonomy(text, FINSERV_TAXONOMY)
        icon = "✅" if result.is_safe else "❌"
        block = " [BLOCKED]" if result.should_block() else ""
        print(f"  {icon} \"{text[:55]}\"")
        print(f"       → {result}{block}\n")
