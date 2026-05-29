"""SSE 流式输出工具"""
import json
import time


def create_sse_response(
    request_id: str,
    model: str,
    content: str,
    finish_reason: str = None,
) -> str:
    """创建 SSE 格式的响应"""
    data = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": content} if content else {},
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(data)}\n\n"
