"""Composition root.

The single place where concrete implementations are chosen and wired. Everything else
receives its collaborators by injection, which is what keeps the inner layers testable
without Docker, Ollama or a network.
"""

from __future__ import annotations

from dataclasses import dataclass

from researchagent.config.loader import ConfigLoader
from researchagent.config.schemas import AgentConfig, ModelCatalog
from researchagent.core.events import EventBus
from researchagent.core.logging import configure_logging, get_logger
from researchagent.core.settings import Settings, get_settings
from researchagent.services.llm_service import LLMService

logger = get_logger(__name__)


@dataclass(slots=True)
class Container:
    """Application-scoped singletons."""

    settings: Settings
    config_loader: ConfigLoader
    model_catalog: ModelCatalog
    agent_config: AgentConfig
    event_bus: EventBus
    llm_service: LLMService

    async def aclose(self) -> None:
        await self.llm_service.aclose()


def build_container(settings: Settings | None = None) -> Container:
    """Wire the object graph. Cheap and synchronous: no network calls happen here."""
    settings = settings or get_settings()
    configure_logging(settings)

    loader = ConfigLoader(settings.config_dir)
    model_catalog = loader.load("models", ModelCatalog)
    agent_config = loader.load("agents", AgentConfig)
    event_bus = EventBus()

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
        event_bus=event_bus,
        llm_service=LLMService(model_catalog, settings, event_bus=event_bus),
    )
