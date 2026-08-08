"""Typed schemas for the YAML files under ``config/``.

Changing a model, a temperature or an agent's model binding must never require a
code change — only an edit to the corresponding YAML file.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from researchagent.core.interfaces.llm import GenerationParams
from researchagent.core.retry import RetryPolicy


class ModelSpec(BaseModel):
    """One entry in ``config/models.yaml``: an alias bound to a provider + model."""

    # `protected_namespaces=()` so a field may be called `model_name` without the
    # Pydantic shadow-warning; nothing here collides with BaseModel's own API.
    model_config = {"populate_by_name": True, "protected_namespaces": ()}

    provider: str = "ollama"
    model_name: str = Field(alias="model", description="Provider-side id, e.g. 'qwen3:8b'")
    params: GenerationParams = Field(default_factory=GenerationParams)
    description: str | None = None


class ModelCatalog(BaseModel):
    """``config/models.yaml`` root."""

    default: str
    models: dict[str, ModelSpec]

    @model_validator(mode="after")
    def _validate_default(self) -> ModelCatalog:
        if not self.models:
            raise ValueError("models catalog must define at least one model")
        if self.default not in self.models:
            raise ValueError(
                f"default alias {self.default!r} is not defined in models ({sorted(self.models)})"
            )
        return self

    def spec_for(self, alias: str) -> ModelSpec:
        return self.models[alias]

    def resolve_alias(self, alias: str | None) -> str:
        return alias if alias is not None else self.default


class AgentSpec(BaseModel):
    """One entry in ``config/agents.yaml``."""

    model: str | None = None
    prompt_version: str = "v1"
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    timeout_seconds: float | None = Field(default=None, gt=0)
    enabled: bool = True
    # Agent-specific knobs (e.g. planner.max_research_questions) validated by the
    # owning agent, not here — keeps this schema stable as agents are added.
    options: dict[str, object] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    """``config/agents.yaml`` root."""

    defaults: AgentSpec = Field(default_factory=AgentSpec)
    agents: dict[str, AgentSpec] = Field(default_factory=dict)

    def spec_for(self, agent_name: str) -> AgentSpec:
        """Agent-specific settings layered over ``defaults``."""
        override = self.agents.get(agent_name)
        if override is None:
            return self.defaults
        explicit = override.model_dump(exclude_unset=True)
        return self.defaults.model_copy(update=explicit)


class CheckpointerKind(StrEnum):
    NONE = "none"
    MEMORY = "memory"


class WorkflowConfig(BaseModel):
    """``config/workflow.yaml`` root."""

    checkpointer: CheckpointerKind = CheckpointerKind.MEMORY
    # Hard ceiling on node executions per run; the reviewer loop makes cycles possible,
    # so this is the guard against a workflow that never converges.
    recursion_limit: int = Field(default=25, ge=1, le=200)
