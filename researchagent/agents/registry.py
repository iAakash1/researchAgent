"""Agent registry and construction.

Concrete agents self-register with ``@AGENTS.register("planner")``; the workflow builds
them by name so adding an agent never edits the orchestrator.
"""

from __future__ import annotations

from typing import Any

from researchagent.agents.base import BaseAgent
from researchagent.config.schemas import AgentConfig
from researchagent.core.events import EventBus
from researchagent.core.prompts import PromptLibrary
from researchagent.core.registry import Registry
from researchagent.services.llm_service import LLMService

AGENTS: Registry[type[BaseAgent[Any, Any]]] = Registry("agent")


def build_agent(
    name: str,
    *,
    agent_config: AgentConfig,
    llm_service: LLMService,
    prompts: PromptLibrary,
    event_bus: EventBus | None = None,
) -> BaseAgent[Any, Any]:
    """Instantiate a registered agent with its configured model, prompt and retry policy."""
    agent_cls = AGENTS.get(name)
    spec = agent_config.spec_for(name)
    return agent_cls(llm_service.get(spec.model), spec, prompts, event_bus=event_bus)
