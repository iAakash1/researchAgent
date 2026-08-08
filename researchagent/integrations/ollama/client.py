"""Thin async client for Ollama's management endpoints.

LangChain's ChatOllama covers inference but not introspection; readiness checks and
"is this model actually pulled?" need the raw HTTP API.
"""

from __future__ import annotations

import httpx

from researchagent.core.exceptions import ProviderUnavailableError
from researchagent.core.logging import get_logger

logger = get_logger(__name__)


class OllamaAdminClient:
    def __init__(self, base_url: str, *, timeout_seconds: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout_seconds)

    @property
    def base_url(self) -> str:
        return self._base_url

    async def list_models(self) -> list[str]:
        """Names of locally available models, e.g. ``["qwen3:8b", "nomic-embed-text"]``."""
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                "Cannot reach Ollama",
                base_url=self._base_url,
                reason=str(exc),
            ) from exc

        payload = response.json()
        return [entry["model"] for entry in payload.get("models", []) if "model" in entry]

    async def is_reachable(self) -> bool:
        try:
            response = await self._client.get("/api/version")
            return response.status_code == httpx.codes.OK
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()
