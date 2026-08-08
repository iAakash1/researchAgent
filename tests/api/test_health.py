from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from researchagent.api.app import create_app
from researchagent.container import Container


@pytest.fixture
async def client(container: Container) -> AsyncIterator[AsyncClient]:
    app = create_app(container=container)
    transport = ASGITransport(app=app)
    # The lifespan context is what populates app.state.container.
    async with (
        AsyncClient(transport=transport, base_url="http://test") as http_client,
        app.router.lifespan_context(app),
    ):
        yield http_client


async def test_liveness_does_not_touch_dependencies(client: AsyncClient) -> None:
    response = await client.get("/health/live")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "ResearchAgent"
    assert body["environment"] == "ci"


async def test_readiness_reports_unpulled_models_as_not_ready(client: AsyncClient) -> None:
    # The fake provider only advertises "fake-model", so the real catalogue is unmet.
    response = await client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["ready"] is False
    assert body["providers"][0]["healthy"] is True
    assert {m["alias"] for m in body["models"]} == {"reasoning", "extraction", "fast"}
    assert all(m["pulled"] is False for m in body["models"])


async def test_openapi_is_served(client: AsyncClient) -> None:
    response = await client.get("/openapi.json")

    assert response.status_code == 200
    assert "/health/ready" in response.json()["paths"]
