# Lab 04 — RBAC Design for AI Systems

## Overview

**Role-Based Access Control (RBAC)** restricts system access based on a user's
assigned role. In AI platforms this means controlling who can train models, deploy
endpoints, read inference logs, or manage API keys.

## What You Will Learn

| Topic | File |
|---|---|
| RBAC data model | `rbac_model.py` |
| Permission checking logic | `permission_checker.py` |
| API key lifecycle management | `api_key_manager.py` |

## Key Concepts

- **Principal** — A user or service identity requesting access.
- **Role** — A named collection of permissions (e.g., `ml_engineer`, `auditor`).
- **Permission** — A fine-grained action on a resource (e.g., `model:deploy`).
- **Resource** — The object being protected (model, dataset, endpoint).
- **Policy Evaluation** — The process of checking if a principal's roles grant
  the required permission on the target resource.

## Architecture

```
Request
   │
   ▼
┌──────────────┐     ┌────────────────┐
│ Auth Gateway │────▶│ Permission     │──▶ Allow / Deny
└──────────────┘     │ Checker        │
                     └───────┬────────┘
                             │
                     ┌───────▼────────┐
                     │ RBAC Model     │
                     │ (roles, perms) │
                     └────────────────┘
```

## Quick Start

```bash
python rbac_model.py
python permission_checker.py
python api_key_manager.py
```

## Design Principles

1. **Least Privilege** — Grant the minimum permissions needed.
2. **Separation of Duties** — No single role should control the full pipeline.
3. **Audit Trail** — Log every access decision for compliance.
4. **Default Deny** — If no rule explicitly allows, reject.
