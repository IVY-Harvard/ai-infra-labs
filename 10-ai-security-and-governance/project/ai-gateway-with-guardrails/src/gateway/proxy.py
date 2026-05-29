"""AI Gateway - Reverse Proxy with Guardrails."""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

from ..guardrails.input_guard import InputGuard
from ..guardrails.output_guard import OutputGuard
from ..auth.rbac import RBACManager
from ..auth.api_key import APIKeyManager
from ..auth.quota import QuotaManager
from ..audit.logger import AuditLogger


app = FastAPI(title="AI Gateway with Guardrails")

input_guard = InputGuard()
output_guard = OutputGuard()
rbac = RBACManager()
key_manager = APIKeyManager()
quota_manager = QuotaManager()
audit_logger = AuditLogger()


@app.post("/v1/chat/completions")
async def proxy_chat(request: Request):
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = key_manager.validate(api_key)
    if not user:
        raise HTTPException(401, "Invalid API key")

    if not rbac.check_permission(user, "chat"):
        raise HTTPException(403, "Permission denied")

    body = await request.json()
    model = body.get("model", "default")

    if not quota_manager.consume(user, estimated_tokens=100):
        raise HTTPException(429, "Quota exceeded")

    input_result = input_guard.check(body)
    if not input_result["safe"]:
        audit_logger.log(user, body, None, blocked=True, reason=input_result["reason"])
        raise HTTPException(400, f"Input blocked: {input_result['reason']}")

    backend_url = f"http://localhost:8000/v1/chat/completions"
    async with aiohttp.ClientSession() as session:
        async with session.post(backend_url, json=body) as resp:
            response_data = await resp.json()

    output_result = output_guard.check(response_data)
    if not output_result["safe"]:
        audit_logger.log(user, body, response_data, blocked=True, reason=output_result["reason"])
        raise HTTPException(400, f"Output blocked: {output_result['reason']}")

    audit_logger.log(user, body, response_data, blocked=False)
    return response_data
