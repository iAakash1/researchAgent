from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from researchagent.api.app import create_app
from researchagent.container import Container
from researchagent.core.exceptions import ProviderUnavailableError
from tests.agents.test_planner import framing, strategy
from tests.conftest import FakeLLMProvider

GOAL = "Agentic AI in healthcare"


@pytest.fixture
def fake_provider() -> FakeLLMProvider:
    """Overrides the default fake so the real Planner has drafts to work with.

    Each request consumes two structured replies (framing, strategy).
    """
    return FakeLLMProvider(structured_sequence=[framing(), strategy(), framing(), strategy()])


@pytest.fixture
async def client(container: Container) -> AsyncIterator[AsyncClient]:
    app = create_app(container=container)
    transport = ASGITransport(app=app)
    async with (
        AsyncClient(transport=transport, base_url="http://test", timeout=30) as http_client,
        app.router.lifespan_context(app),
    ):
        yield http_client


async def test_plan_returns_a_complete_plan(client: AsyncClient) -> None:
    response = await client.post("/research/plan", json={"goal": GOAL})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["run_id"]
    assert body["plan"]["topic"] == "Agentic AI in clinical decision support"
    assert [q["id"] for q in body["plan"]["research_questions"]] == ["RQ1", "RQ2"]
    assert body["history"][0]["stage"] == "planning"
    assert body["failure"] is None


async def test_plan_accepts_constraints_and_feedback(client: AsyncClient) -> None:
    response = await client.post(
        "/research/plan",
        json={
            "goal": GOAL,
            "constraints": {"max_research_questions": 1, "year_from": 2021},
            "feedback": ["Needs newer work"],
        },
    )

    assert response.status_code == 200
    plan = response.json()["plan"]
    assert len(plan["research_questions"]) == 1
    assert plan["strategy"]["year_from"] == 2021


async def test_short_goal_is_rejected_by_validation(client: AsyncClient) -> None:
    response = await client.post("/research/plan", json={"goal": "ai"})

    assert response.status_code == 422


async def test_domain_failure_becomes_a_502(
    client: AsyncClient, fake_provider: FakeLLMProvider
) -> None:
    fake_provider.fail_times = 99
    fake_provider.error = ProviderUnavailableError("ollama is down")

    response = await client.post("/research/plan", json={"goal": GOAL})

    assert response.status_code == 502
    error = response.json()["error"]
    assert error["code"] == "workflow_execution_error"
    assert error["context"]["stage"] == "planning"
    assert error["context"]["agent"] == "planner"
    assert error["context"]["cause"] == "provider_unavailable"


async def test_unexpected_exception_is_captured_not_leaked(
    client: AsyncClient, fake_provider: FakeLLMProvider
) -> None:
    """A bug inside an agent must fail the run, not unwind the graph."""
    fake_provider.structured_sequence = []
    fake_provider.structured = None  # the fake raises a bare AssertionError

    response = await client.post("/research/plan", json={"goal": GOAL})

    assert response.status_code == 502
    assert response.json()["error"]["context"]["cause"] == "unexpected_error"


async def test_run_is_retrievable_after_completion(client: AsyncClient) -> None:
    created = await client.post("/research/plan", json={"goal": GOAL})
    run_id = created.json()["run_id"]

    response = await client.get(f"/research/runs/{run_id}")

    assert response.status_code == 200
    assert response.json()["run_id"] == run_id
    assert response.json()["plan"] is not None


async def test_unknown_run_returns_404(client: AsyncClient) -> None:
    response = await client.get("/research/runs/nope")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "run_not_found"


async def test_stream_emits_stage_then_done(client: AsyncClient) -> None:
    async with client.stream("POST", "/research/plan/stream", json={"goal": GOAL}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join([chunk async for chunk in response.aiter_text()])

    events = [block for block in body.split("\n\n") if block.strip()]
    assert events[0].startswith("event: stage")
    assert json.loads(events[0].split("data: ", 1)[1])["node"] == "planning"
    assert events[-1].startswith("event: done")
