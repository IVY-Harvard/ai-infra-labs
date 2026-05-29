# AI Gateway with Guardrails

A production-ready API gateway for AI model access with built-in security, compliance, and governance controls.

## Architecture

```
Client -> [Auth] -> [Rate Limiter] -> [Input Guard] -> [Router] -> AI Model
                                                                      |
Client <- [Audit Logger] <- [Output Guard] <- [Policy Engine] <------+
```

## Components

| Module | Purpose |
|--------|---------|
| `src/gateway/` | Reverse proxy, routing, rate limiting |
| `src/guardrails/` | Input/output content filtering and rule engine |
| `src/auth/` | RBAC, API key management, quota enforcement |
| `src/audit/` | Request logging and compliance reporting |
| `src/policy/` | Policy-as-code engine and loader |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn src.gateway.proxy:app --reload --port 8000

# Run with Docker
docker-compose -f deploy/docker-compose.yaml up
```

## API Endpoints

- `POST /v1/chat/completions` — Proxied chat completion with guardrails
- `GET /v1/health` — Health check
- `GET /v1/audit/report` — Compliance report (admin only)

## Configuration

Environment variables:
- `UPSTREAM_URL` — Target AI model API endpoint
- `API_KEY_SECRET` — Secret for API key hashing
- `RATE_LIMIT_RPM` — Requests per minute limit (default: 60)
- `LOG_LEVEL` — Logging verbosity (default: INFO)

## Testing

```bash
pytest tests/ -v
```
