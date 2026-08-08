"""Composition root.

The single place where concrete implementations are chosen and wired. Everything else
receives its collaborators by injection, which is what keeps the inner layers testable
without Docker, Ollama or a network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from researchagent.agents.registry import build_agent
from researchagent.config.loader import ConfigLoader
from researchagent.config.schemas import AgentConfig, ModelCatalog, SourcesConfig, WorkflowConfig
from researchagent.core.events import EventBus
from researchagent.core.interfaces.paper_source import PaperSource
from researchagent.core.logging import configure_logging, get_logger
from researchagent.core.prompts import PromptLibrary
from researchagent.core.settings import Settings, get_settings
from researchagent.integrations.sources import build_enabled_sources
from researchagent.memory.checkpoints import build_checkpointer
from researchagent.repositories.paper_repository import JsonPaperRepository
from researchagent.services.deduplication import PaperDeduplicator
from researchagent.services.discovery_service import DiscoveryService
from researchagent.services.llm_service import LLMService
from researchagent.services.ranking import HeuristicScorer
from researchagent.services.retrieval_service import RetrievalService
from researchagent.workflows.research import build_research_graph
from researchagent.workflows.runner import WorkflowRunner

logger = get_logger(__name__)


@dataclass(slots=True)
class Container:
    """Application-scoped singletons."""

    settings: Settings
    config_loader: ConfigLoader
    model_catalog: ModelCatalog
    agent_config: AgentConfig
    workflow_config: WorkflowConfig
    sources_config: SourcesConfig
    prompt_library: PromptLibrary
    event_bus: EventBus
    llm_service: LLMService
    paper_sources: list[PaperSource]
    paper_repository: JsonPaperRepository
    discovery_service: DiscoveryService
    retrieval_service: RetrievalService
    workflow_runner: WorkflowRunner

    async def aclose(self) -> None:
        await self.llm_service.aclose()
        for source in self.paper_sources:
            await source.aclose()


def build_container(settings: Settings | None = None) -> Container:
    """Wire the object graph. Cheap and synchronous: no network calls happen here."""
    settings = settings or get_settings()
    configure_logging(settings)

    loader = ConfigLoader(settings.config_dir)
    model_catalog = loader.load("models", ModelCatalog)
    agent_config = loader.load("agents", AgentConfig)
    workflow_config = loader.load("workflow", WorkflowConfig)
    sources_config = loader.load("sources", SourcesConfig)

    prompt_library = PromptLibrary(settings.prompts_dir)
    event_bus = EventBus()
    llm_service = LLMService(model_catalog, settings, event_bus=event_bus)

    paper_sources = build_enabled_sources(sources_config, settings.project_root)
    paper_repository = JsonPaperRepository(
        _resolve(sources_config.metadata_dir, settings.project_root)
    )
    discovery_service = DiscoveryService(
        paper_sources,
        PaperDeduplicator(sources_config.deduplication),
        HeuristicScorer(sources_config.ranking),
        sources_config.discovery_settings(),
        repository=paper_repository,
    )
    retrieval_service = RetrievalService(
        {source.name: source for source in paper_sources},
        paper_repository,
        _resolve(sources_config.download_dir, settings.project_root),
        sources_config.retrieval,
    )

    planner = build_agent(
        "planner",
        agent_config=agent_config,
        llm_service=llm_service,
        prompts=prompt_library,
        event_bus=event_bus,
    )
    graph = build_research_graph(
        planner=planner,
        discovery=discovery_service,
        checkpointer=build_checkpointer(workflow_config.checkpointer),
    )

    logger.info(
        "container_built",
        environment=settings.environment,
        config_dir=str(settings.config_dir),
        model_aliases=sorted(model_catalog.models),
        default_model=model_catalog.default,
        paper_sources=[source.name.value for source in paper_sources],
    )

    return Container(
        settings=settings,
        config_loader=loader,
        model_catalog=model_catalog,
        agent_config=agent_config,
        workflow_config=workflow_config,
        sources_config=sources_config,
        prompt_library=prompt_library,
        event_bus=event_bus,
        llm_service=llm_service,
        paper_sources=paper_sources,
        paper_repository=paper_repository,
        discovery_service=discovery_service,
        retrieval_service=retrieval_service,
        workflow_runner=WorkflowRunner(graph, workflow_config),
    )


def _resolve(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path
