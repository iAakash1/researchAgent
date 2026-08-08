"""Shared async HTTP client for literature providers.

Every provider gets one of these instead of a raw httpx client, because they all need
the same three things and getting any of them wrong gets you blocked:

* **Rate limiting** — these are free public APIs. Crossref and OpenAlex ask for a polite
  request rate and a contact address; Semantic Scholar returns 429 quickly without one.
* **Uniform error mapping** — a 429 and a connection reset must become the same
  retryable domain error regardless of which provider produced them.
* **Streaming downloads** — PDFs are tens of megabytes; they are written to disk in
  chunks, never buffered in memory.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import TracebackType
from typing import Any

import httpx

from researchagent.core.constants import APP_NAME, APP_VERSION
from researchagent.core.exceptions import (
    SourceRateLimitedError,
    SourceResponseError,
    SourceUnavailableError,
)
from researchagent.core.logging import get_logger

logger = get_logger(__name__)

_PDF_CHUNK_BYTES = 64 * 1024


class RateLimiter:
    """Serialises requests so that at most ``rate`` start per second."""

    def __init__(self, requests_per_second: float) -> None:
        self._min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self._lock = asyncio.Lock()
        self._last_start = 0.0

    async def acquire(self) -> None:
        if self._min_interval <= 0:
            return
        async with self._lock:
            elapsed = time.monotonic() - self._last_start
            wait = self._min_interval - elapsed
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_start = time.monotonic()


class HttpClient:
    """Thin, rate-limited HTTP client that speaks the project's error vocabulary."""

    def __init__(
        self,
        source: str,
        *,
        base_url: str = "",
        timeout_seconds: float = 30.0,
        requests_per_second: float = 3.0,
        contact_email: str | None = None,
        headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._source = source
        # A real User-Agent with a contact address is what keeps these APIs friendly;
        # OpenAlex and Crossref explicitly reward it with the faster request pool.
        user_agent = f"{APP_NAME}/{APP_VERSION}"
        if contact_email:
            user_agent = f"{user_agent} (mailto:{contact_email})"

        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": user_agent, **(headers or {})},
            transport=transport,
        )
        self._limiter = RateLimiter(requests_per_second)

    async def get_json(self, url: str, *, params: dict[str, Any] | None = None) -> Any:
        response = await self._get(url, params=params)
        try:
            return response.json()
        except ValueError as exc:
            raise SourceResponseError(
                "Provider returned malformed JSON",
                source=self._source,
                url=str(response.url),
                reason=str(exc),
            ) from exc

    async def get_text(self, url: str, *, params: dict[str, Any] | None = None) -> str:
        return (await self._get(url, params=params)).text

    async def download(self, url: str, destination: Path) -> Path:
        """Stream a file to disk. Writes to a temp path first so an interrupted download
        never leaves a truncated PDF that later stages would treat as valid."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")

        await self._limiter.acquire()
        try:
            async with self._client.stream("GET", url) as response:
                self._raise_for_status(response, url)
                written = 0
                with temporary.open("wb") as handle:
                    async for chunk in response.aiter_bytes(_PDF_CHUNK_BYTES):
                        handle.write(chunk)
                        written += len(chunk)
        except httpx.HTTPError as exc:
            temporary.unlink(missing_ok=True)
            raise SourceUnavailableError(
                "Download failed", source=self._source, url=url, reason=str(exc)
            ) from exc
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

        temporary.replace(destination)
        logger.debug(
            "pdf_downloaded",
            source=self._source,
            destination=str(destination),
            bytes=written,
        )
        return destination

    async def is_reachable(self, url: str, *, params: dict[str, Any] | None = None) -> bool:
        try:
            await self._get(url, params=params)
        except (SourceUnavailableError, SourceRateLimitedError, SourceResponseError):
            return False
        return True

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> HttpClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def _get(self, url: str, *, params: dict[str, Any] | None) -> httpx.Response:
        await self._limiter.acquire()
        try:
            response = await self._client.get(url, params=_drop_none(params))
        except httpx.TimeoutException as exc:
            raise SourceUnavailableError(
                "Provider timed out", source=self._source, url=url, reason=str(exc)
            ) from exc
        except httpx.HTTPError as exc:
            raise SourceUnavailableError(
                "Provider unreachable", source=self._source, url=url, reason=str(exc)
            ) from exc

        self._raise_for_status(response, url)
        return response

    def _raise_for_status(self, response: httpx.Response, url: str) -> None:
        status = response.status_code
        if status == httpx.codes.TOO_MANY_REQUESTS:
            raise SourceRateLimitedError(
                "Provider rate limit reached",
                source=self._source,
                url=url,
                retry_after=response.headers.get("Retry-After"),
            )
        if status >= httpx.codes.INTERNAL_SERVER_ERROR:
            raise SourceUnavailableError(
                "Provider returned a server error",
                source=self._source,
                url=url,
                status_code=status,
            )
        if status >= httpx.codes.BAD_REQUEST:
            raise SourceResponseError(
                "Provider rejected the request",
                source=self._source,
                url=url,
                status_code=status,
            )


def _drop_none(params: dict[str, Any] | None) -> dict[str, Any] | None:
    """Providers differ on how they treat empty parameters; omit them entirely."""
    if params is None:
        return None
    return {key: value for key, value in params.items() if value is not None}
