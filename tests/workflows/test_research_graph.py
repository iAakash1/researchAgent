from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from researchagent.agents.base import AgentContext, BaseAgent
from researchagent.agents.planner.schemas import PlannerInput, PlannerOutput
from researchagent.config.schemas import CheckpointerKind, DiscoverySettings, WorkflowConfig
from researchagent.core.exceptions import OutputParsingError, RunNotFoundError
from researchagent.core.interfaces.paper_source import PaperSource, SearchQuery, SourceHealth
from researchagent.core.prompts import PromptLibrary
from researchagent.core.retry import RetryPolicy
from researchagent.memory.checkpoints import build_checkpointer
from researchagent.models.paper import Paper, SourceName
from researchagent.models.research import (
    QuestionPriority,
    ResearchPlan,
    ResearchQuestion,
    SearchStrategy,
)
from researchagent.schemas.workflow import RunStatus, StageStatus, WorkflowStage
from researchagent.services.deduplication import PaperDeduplicator
from researchagent.services.discovery_service import DiscoveryService
from researchagent.services.document import DocumentIntelligenceService
from researchagent.services.knowledge import KnowledgeIntelligenceService
from researchagent.services.llm_service import BoundLLM
from researchagent.services.ranking import HeuristicScorer
from researchagent.workflows.research import build_research_graph
from researchagent.workflows.runner import WorkflowRunner

CONFIG = WorkflowConfig(checkpointer=CheckpointerKind.MEMORY, recursion_limit=10)


def a_plan(topic: str = "Agentic AI in healthcare") -> ResearchPlan:
    return ResearchPlan(
        topic=topic,
        framing="A scoped review of autonomous agents applied to clinical workflows.",
        research_questions=[
            ResearchQuestion(
                id="RQ1",
                question="Which agent architectures are used in clinical triage?",
                rationale="Architecture choice drives safety and auditability.",
                priority=QuestionPriority.HIGH,
                keywords=["agent architecture"],
            )
        ],
        strategy=SearchStrategy(queries=["agentic ai healthcare"]),
    )


class StubPlanner(BaseAgent[PlannerInput, PlannerOutput]):
    """Stands in for the real Planner: the graph is under test, not the reasoning."""

    name: ClassVar[str] = "planner"
    description: ClassVar[str] = "stub"
    input_schema: ClassVar[type[BaseModel]] = PlannerInput
    output_schema: ClassVar[type[BaseModel]] = PlannerOutput

    def __init__(self, llm: BoundLLM, spec: Any, prompts: PromptLibrary) -> None:
        super().__init__(llm, spec, prompts)
        self.seen: list[PlannerInput] = []
        self.error: Exception | None = None

    async def execute(self, payload: PlannerInput, context: AgentContext) -> PlannerOutput:
        self.seen.append(payload)
        if self.error is not None:
            raise self.error
        return PlannerOutput(plan=a_plan())


@pytest.fixture
def planner(bound_llm: BoundLLM, prompt_library: PromptLibrary) -> StubPlanner:
    from researchagent.config.schemas import AgentSpec

    return StubPlanner(bound_llm, AgentSpec(retry=RetryPolicy(max_attempts=1)), prompt_library)


class StubSource(PaperSource):
    """Returns a fixed paper set; the graph is under test, not any provider."""

    name = SourceName.MANUAL

    def __init__(self, papers: list[Paper] | None = None) -> None:
        self.papers = papers or []
        self.searches: list[SearchQuery] = []

    async def search(self, query: SearchQuery) -> list[Paper]:
        self.searches.append(query)
        return list(self.papers)

    async def get_paper(self, identifier: str) -> Paper | None:
        return None

    async def download_pdf(self, paper: Paper, destination: Path) -> Path:
        raise NotImplementedError

    async def health(self) -> SourceHealth:
        return SourceHealth(source=self.name, healthy=True)

    async def aclose(self) -> None:
        return None


@pytest.fixture
def source() -> StubSource:
    return StubSource(
        [
            Paper(
                id="arxiv:2401.00001",
                title="Coordination mechanisms for clinical multi-agent systems",
                provider=SourceName.ARXIV,
                year=2024,
            )
        ]
    )


@pytest.fixture
def discovery(source: StubSource) -> DiscoveryService:
    return DiscoveryService([source], PaperDeduplicator(), HeuristicScorer(), DiscoverySettings())


@pytest.fixture
def runner(
    planner: StubPlanner,
    discovery: DiscoveryService,
    document_service: DocumentIntelligenceService,
    knowledge_service: KnowledgeIntelligenceService,
) -> WorkflowRunner:
    graph = build_research_graph(
        planner=planner,
        discovery=discovery,
        documents=document_service,
        knowledge=knowledge_service,
        checkpointer=InMemorySaver(),
    )
    return WorkflowRunner(graph, CONFIG)


async def test_successful_run_completes_with_a_plan(runner: WorkflowRunner) -> None:
    state = await runner.run("Agentic AI in healthcare")

    assert state.status is RunStatus.COMPLETED
    assert state.succeeded is True
    assert state.plan is not None
    assert state.plan.topic == "Agentic AI in healthcare"
    assert state.failure is None


async def test_history_records_every_stage(runner: WorkflowRunner) -> None:
    state = await runner.run("Agentic AI in healthcare")

    assert [record.stage for record in state.history] == [
        WorkflowStage.PLANNING,
        WorkflowStage.DISCOVERY,
        WorkflowStage.DOCUMENT_INTELLIGENCE,
        WorkflowStage.KNOWLEDGE_EXTRACTION,
    ]
    assert [record.agent for record in state.history[:2]] == ["planner", "discovery_service"]
    assert all(record.status is StageStatus.OK for record in state.history[:2])
    # The stub source returns metadata-only papers, so the document stage is blocked by
    # its `requires_local_pdfs` guard rather than failing.
    # Both later stages are blocked by their guards, not failed: the stub source returns
    # metadata-only papers, so there is nothing to parse and nothing to extract from.
    assert state.history[2].status is StageStatus.BLOCKED
    assert state.history[3].status is StageStatus.BLOCKED


async def test_planner_failure_halts_before_discovery(
    runner: WorkflowRunner, planner: StubPlanner, source: StubSource
) -> None:
    """The conditional edge exists so a failed plan never sends garbage to the indexes."""
    planner.error = OutputParsingError("model returned junk")

    state = await runner.run("Agentic AI in healthcare")

    assert state.status is RunStatus.FAILED
    assert source.searches == []
    assert [record.stage for record in state.history] == [WorkflowStage.PLANNING]


async def test_discovery_receives_the_plan_queries(
    runner: WorkflowRunner, source: StubSource
) -> None:
    state = await runner.run("Agentic AI in healthcare")

    assert [query.text for query in source.searches] == ["agentic ai healthcare"]
    assert state.candidates
    assert state.discovery is not None
    assert state.discovery.candidates == 1


async def test_goal_and_constraints_reach_the_agent(
    runner: WorkflowRunner, planner: StubPlanner
) -> None:
    await runner.run(
        "Agentic AI in healthcare",
        constraints={"year_from": 2020, "focus_areas": ["triage"]},  # type: ignore[arg-type]
        feedback=["needs newer work"],
    )

    payload = planner.seen[0]
    assert payload.goal == "Agentic AI in healthcare"
    assert payload.constraints.year_from == 2020
    assert payload.feedback == ["needs newer work"]


async def test_agent_failure_is_recorded_not_raised(
    runner: WorkflowRunner, planner: StubPlanner
) -> None:
    planner.error = OutputParsingError("model returned junk")

    state = await runner.run("Agentic AI in healthcare")

    assert state.status is RunStatus.FAILED
    assert state.plan is None
    assert state.failure is not None
    assert state.failure.stage is WorkflowStage.PLANNING
    assert state.failure.agent == "planner"
    # The wrapper's code is useless for debugging; the original cause is preserved.
    assert state.failure.code == "output_parsing_error"
    assert state.history[0].status is StageStatus.FAILED


async def test_stream_emits_one_update_per_node(runner: WorkflowRunner) -> None:
    updates = [update async for update in runner.stream("Agentic AI in healthcare")]

    assert [u.node for u in updates] == [
        WorkflowStage.PLANNING.value,
        WorkflowStage.DISCOVERY.value,
        WorkflowStage.DOCUMENT_INTELLIGENCE.value,
        WorkflowStage.KNOWLEDGE_EXTRACTION.value,
    ]


async def test_stream_reports_failure(runner: WorkflowRunner, planner: StubPlanner) -> None:
    planner.error = OutputParsingError("bad json")

    updates = [update async for update in runner.stream("Agentic AI in healthcare")]

    assert updates[0].status is RunStatus.FAILED
    assert updates[0].failure is not None


async def test_run_is_retrievable_from_its_checkpoint(runner: WorkflowRunner) -> None:
    state = await runner.run("Agentic AI in healthcare", run_id="run-42")

    restored = await runner.get_state("run-42")

    assert restored.run_id == state.run_id
    assert restored.plan is not None
    assert restored.status is RunStatus.COMPLETED


async def test_unknown_run_id_raises(runner: WorkflowRunner) -> None:
    with pytest.raises(RunNotFoundError):
        await runner.get_state("does-not-exist")


async def test_runs_are_isolated_by_run_id(runner: WorkflowRunner) -> None:
    first = await runner.run("Agentic AI in healthcare", run_id="a")
    second = await runner.run("Agentic AI in radiology", run_id="b")

    assert first.run_id != second.run_id
    assert (await runner.get_state("a")).goal == "Agentic AI in healthcare"
    assert (await runner.get_state("b")).goal == "Agentic AI in radiology"


async def test_graph_without_checkpointer_reports_it(
    planner: StubPlanner,
    discovery: DiscoveryService,
    document_service: DocumentIntelligenceService,
    knowledge_service: KnowledgeIntelligenceService,
) -> None:
    graph = build_research_graph(
        planner=planner,
        discovery=discovery,
        documents=document_service,
        knowledge=knowledge_service,
        checkpointer=None,
    )
    runner = WorkflowRunner(graph, WorkflowConfig(checkpointer=CheckpointerKind.NONE))

    state = await runner.run("Agentic AI in healthcare")

    assert state.status is RunStatus.COMPLETED
    assert runner.checkpointing_enabled is False


def test_checkpointer_factory_honours_config() -> None:
    assert build_checkpointer(CheckpointerKind.NONE) is None
    assert isinstance(build_checkpointer(CheckpointerKind.MEMORY), InMemorySaver)
