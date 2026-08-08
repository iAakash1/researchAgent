from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from researchagent.api.app import create_app
from researchagent.config.loader import ConfigLoader
from researchagent.config.schemas import AgentConfig, ModelCatalog
from researchagent.container import Container
from researchagent.core.events import EventBus
from researchagent.core.settings import Settings
from researchagent.services.llm_service import LLMService
from tests.conftest import FakeLLMProvider


@pytest.fixture
def container(
    settings: Settings,
    config_loader: ConfigLoader,
    model_catalog: ModelCatalog,
    agent_config: AgentConfig,
    event_bus: EventBus,
    fake_provider: FakeLLMProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> Container:
    service = LLMService(model_catalog, settings, event_bus=event_bus)
    monkeypatch.setattr(service, "_provider", lambda _name: fake_provider)
    return Container(
        settings=settings,
        config_loader=config_loader,
        model_catalog=model_catalog,
        agent_config=agent_config,
        event_bus=event_bus,
        llm_service=service,
    )


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
