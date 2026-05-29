# Lab 09: Audit Logging for AI Systems

## Objectives

- Implement async audit logging for LLM request/response cycles
- Build compliance verification for audit logs
- Generate structured audit reports

## Overview

Audit logging is critical for AI governance. This lab implements a complete
audit pipeline: capturing events, verifying compliance, and generating reports.

## Files

| File | Description |
|------|-------------|
| `audit_logger.py` | Async logger capturing requests, responses, and metadata |
| `compliance_checker.py` | Validates logs against compliance requirements |
| `report_generator.py` | Generates audit reports in multiple formats |

## Exercises

1. Run the audit logger and observe captured events
2. Configure compliance rules and validate logs
3. Generate a compliance report from collected audit data

## Running

```bash
pip install aiofiles pydantic
python audit_logger.py
python compliance_checker.py
python report_generator.py
```

## Key Concepts

- **Immutability**: Audit logs must not be modifiable after creation
- **Completeness**: Every LLM interaction must be logged
- **Traceability**: Each log entry links to a user, session, and model
- **Retention**: Logs must be kept for the configured retention period
