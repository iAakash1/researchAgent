"""Composition root.

The single place where concrete implementations are chosen and wired. Everything else
receives its collaborators by injection, which is what keeps the inner layers testable
without Docker, Ollama or a network.
"""

from __future__ import annotations

from dataclasses import dataclass

from researchagent.agents.registry import build_agent
from researchagent.config.loader import ConfigLoader
from researchagent.config.schemas import AgentConfig, ModelCatalog, WorkflowConfig
from researchagent.core.events import EventBus
from researchagent.core.logging import configure_logging, get_logger
from researchagent.core.prompts import PromptLibrary
from researchagent.core.settings import Settings, get_settings
from researchagent.memory.checkpoints import build_checkpointer
from researchagent.services.llm_service import LLMService
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
    prompt_library: PromptLibrary
    event_bus: EventBus
    llm_service: LLMService
    workflow_runner: WorkflowRunner

    async def aclose(self) -> None:
        await self.llm_service.aclose()


def build_container(settings: Settings | None = None) -> Container:
    """Wire the object graph. Cheap and synchronous: no network calls happen here."""
    settings = settings or get_settings()
    configure_logging(settings)

    loader = ConfigLoader(settings.config_dir)
    model_catalog = loader.load("models", ModelCatalog)
    agent_config = loader.load("agents", AgentConfig)
    workflow_config = loader.load("workflow", WorkflowConfig)

    prompt_library = PromptLibrary(settings.prompts_dir)
    event_bus = EventBus()
    llm_service = LLMService(model_catalog, settings, event_bus=event_bus)

    planner = build_agent(
        "planner",
        agent_config=agent_config,
        llm_service=llm_service,
        prompts=prompt_library,
        event_bus=event_bus,
    )
    graph = build_research_graph(
        planner=planner,
        checkpointer=build_checkpointer(workflow_config.checkpointer),
    )

    logger.info(
        "container_built",
        environment=settings.environment,
        config_dir=str(settings.config_dir),
        model_aliases=sorted(model_catalog.models),
        default_model=model_catalog.default,
    )

    return Container(
        settings=settings,
        config_loader=loader,
        model_catalog=model_catalog,
        agent_config=agent_config,
        workflow_config=workflow_config,
        prompt_library=prompt_library,
        event_bus=event_bus,
        llm_service=llm_service,
        workflow_runner=WorkflowRunner(graph, workflow_config),
    )
