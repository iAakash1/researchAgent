"""The agent roster.

Five agents, listed explicitly. They used to self-register through a decorator, which
meant the roster only existed once the right modules happened to be imported — an agent
was "registered" by an import side-effect somewhere else in the tree. A plain dict is
the whole wiring, visible in one place, and impossible to get wrong by import order.
"""

from __future__ import annotations

from typing import Any

from researchagent.agents.base import BaseAgent
from researchagent.agents.planner.agent import PlannerAgent
from researchagent.agents.reasoning.agent import ResearchReasoningAgent
from researchagent.agents.retrieval.agent import RetrievalAgent
from researchagent.agents.reviewer.agent import ReviewerAgent
from researchagent.agents.verification.agent import VerificationAgent
from researchagent.config.schemas import AgentConfig
from researchagent.core.events import EventBus
from researchagent.core.exceptions import ConfigurationError
from researchagent.core.prompts import PromptLibrary
from researchagent.services.llm_service import LLMService

AGENTS: dict[str, type[BaseAgent[Any, Any]]] = {
    "planner": PlannerAgent,  # research goal   -> research questions
    "retrieval": RetrievalAgent,  # question    -> evidence bundles
    "reasoning": ResearchReasoningAgent,  # evidence -> findings with citations
    "verification": VerificationAgent,  # finding -> adversarial verdict
    "reviewer": ReviewerAgent,  # findings      -> accept / revise / reject
}


def agent_class(name: str) -> type[BaseAgent[Any, Any]]:
    """Look up an agent, failing with the list of real names rather than a KeyError."""
    try:
        return AGENTS[name]
    except KeyError:
        raise ConfigurationError(f"Unknown agent {name!r}", known=sorted(AGENTS)) from None


def build_agent(
    name: str,
    *,
    agent_config: AgentConfig,
    llm_service: LLMService,
    prompts: PromptLibrary,
    event_bus: EventBus | None = None,
) -> BaseAgent[Any, Any]:
    """Instantiate an agent with its configured model, prompt version and retry policy."""
    spec = agent_config.spec_for(name)
    return agent_class(name)(llm_service.get(spec.model), spec, prompts, event_bus=event_bus)
