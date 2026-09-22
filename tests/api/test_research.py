from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from researchagent.api.app import create_app
from researchagent.api.dependencies import get_research_service
from researchagent.container import Container
from researchagent.core.events import Event, EventType, StagePayload
from researchagent.core.exceptions import ProviderUnavailableError
from researchagent.schemas.result import ResearchResult, ResearchRunSnapshot
from researchagent.schemas.workflow import RunStatus
from researchagent.services.research_run import SequencedRunEvent
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


async def test_unknown_research_stream_returns_404(client: AsyncClient) -> None:
    response = await client.get("/research/runs/nope/stream")

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


async def test_frontend_is_served_by_the_api(client: AsyncClient) -> None:
    response = await client.get("/app/")

    assert response.status_code == 200
    assert "ResearchAgent" in response.text
    javascript = (await client.get("/app/app.js")).text
    assert 'fetch("/research/run"' in javascript
    assert "/research/runs/${encodeURIComponent(run.runId)}/stream" in javascript
    assert "researchagent.activeRun" in javascript
    assert "if(!phase)return" in javascript
    assert "if(index<highestStageIndex||index<0)return" in javascript
    assert 'phaseByEvent[item.type]||"Planning"' not in javascript
    assert "stageIndex" in javascript


async def test_reconnecting_to_stream_does_not_create_another_run(container: Container) -> None:
    expected = ResearchResult(
        run_id="run-1",
        research_goal=GOAL,
        status=RunStatus.COMPLETED,
        discovered_papers=4,
        processed_papers=3,
        final_summary="A verified synthesis.",
    )

    class StubResearchService:
        result = expected
        start_calls = 0

        def start(self, goal: str, **_: object) -> str:
            assert goal == GOAL
            self.start_calls += 1
            return self.result.run_id

        def get(self, run_id: str) -> ResearchRunSnapshot:
            assert run_id == "run-1"
            return ResearchRunSnapshot(
                run_id=run_id,
                status=RunStatus.COMPLETED,
                result=self.result,
            )

        async def stream(self, run_id: str, *, after: int = 0) -> AsyncIterator[SequencedRunEvent]:
            assert run_id == "run-1"
            if after < 1:
                yield SequencedRunEvent(
                    sequence=1,
                    event=Event(
                        type=EventType.WORKFLOW_STARTED,
                        payload=StagePayload(stage="planning"),
                        run_id=run_id,
                    ),
                )

    service = StubResearchService()
    app = create_app(container=container)
    app.dependency_overrides[get_research_service] = lambda: service
    transport = ASGITransport(app=app)
    async with (
        AsyncClient(transport=transport, base_url="http://test") as http_client,
        app.router.lifespan_context(app),
    ):
        response = await http_client.post("/research/run", json={"goal": GOAL})
        fetched = await http_client.get("/research/results/run-1")
        first_stream = await http_client.get("/research/runs/run-1/stream")
        reconnected = await http_client.get("/research/runs/run-1/stream?after=1")

    assert response.status_code == 202
    assert response.json() == {"run_id": "run-1", "status": "running"}
    assert fetched.json()["result"]["processed_papers"] == 3
    assert "event: progress" in first_stream.text
    assert "event: result" in first_stream.text
    assert "event: progress" not in reconnected.text
    assert "event: result" in reconnected.text
    assert service.start_calls == 1
