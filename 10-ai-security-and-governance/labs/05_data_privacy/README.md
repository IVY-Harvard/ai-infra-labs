# Lab 05 — Data Privacy for AI Systems

## Overview

AI systems process vast amounts of personal data during training and inference.
This lab covers techniques for **detecting**, **anonymizing**, and **auditing**
personally identifiable information (PII) to meet privacy regulations like
GDPR, CCPA, and HIPAA.

## What You Will Learn

| Topic | File |
|---|---|
| PII detection (regex + NER) | `pii_detector.py` |
| Data anonymization & masking | `data_anonymizer.py` |
| Privacy compliance auditing | `privacy_audit.py` |

## Key Concepts

- **PII (Personally Identifiable Information)** — Data that can identify a
  person: names, emails, phone numbers, SSNs, credit cards, IP addresses.
- **Data Minimization** — Only collect and retain data that is strictly necessary.
- **Anonymization** — Irreversibly removing identifying information.
- **Pseudonymization** — Replacing identifiers with tokens that can be reversed
  with a separate key (still considered personal data under GDPR).
- **Right to Erasure** — Users can request deletion of their data from training
  sets and logs.

## Architecture

```
Raw Data
   │
   ▼
┌──────────────┐     ┌──────────────────┐
│ PII Detector │────▶│ Data Anonymizer  │──▶ Safe Data
└──────────────┘     └──────────────────┘
   │                          │
   ▼                          ▼
┌──────────────┐     ┌──────────────────┐
│ Alert /      │     │ Privacy Audit    │──▶ Compliance Report
│ Block        │     │ Logger           │
└──────────────┘     └──────────────────┘
```

## Quick Start

```bash
python pii_detector.py
python data_anonymizer.py
python privacy_audit.py
```

## Regulatory Context

| Regulation | Key Requirement |
|---|---|
| GDPR (EU) | Lawful basis, data minimization, right to erasure |
| CCPA (California) | Right to know, right to delete, opt-out of sale |
| HIPAA (US Health) | De-identification of protected health information |
