"""Model Router - Route requests to appropriate model backends."""
from dataclasses import dataclass


@dataclass
class ModelEndpoint:
    name: str
    url: str
    weight: int = 1
    healthy: bool = True


class ModelRouter:
    def __init__(self):
        self.endpoints: dict[str, list[ModelEndpoint]] = {}
        self._rr_index: dict[str, int] = {}

    def register(self, model: str, endpoint: ModelEndpoint):
        self.endpoints.setdefault(model, []).append(endpoint)
        self._rr_index.setdefault(model, 0)

    def route(self, model: str) -> str:
        candidates = [e for e in self.endpoints.get(model, []) if e.healthy]
        if not candidates:
            raise ValueError(f"No healthy endpoint for model: {model}")
        idx = self._rr_index[model] % len(candidates)
        self._rr_index[model] = idx + 1
        return candidates[idx].url

    def mark_unhealthy(self, model: str, url: str):
        for ep in self.endpoints.get(model, []):
            if ep.url == url:
                ep.healthy = False

    def list_models(self) -> dict:
        return {m: [{"url": e.url, "healthy": e.healthy} for e in eps]
                for m, eps in self.endpoints.items()}
