"""Typed schemas for the YAML files under ``config/``.

Changing a model, a temperature or an agent's model binding must never require a
code change — only an edit to the corresponding YAML file.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from researchagent.core.interfaces.llm import GenerationParams
from researchagent.core.retry import RetryPolicy
from researchagent.models.paper import SourceName


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


class DeduplicationConfig(BaseModel):
    title_similarity_threshold: float = Field(default=0.93, ge=0.0, le=1.0)
    # Titles of very different length are never the same work; skipping the expensive
    # comparison keeps deduplication near-linear on realistic result sets.
    length_ratio_floor: float = Field(default=0.7, ge=0.0, le=1.0)
    compare_titles: bool = True


class RankingWeights(BaseModel):
    """Relative influence of each ranking signal. Benchmarked, not guessed, in v1.0."""

    title_match: float = Field(default=0.35, ge=0.0)
    abstract_match: float = Field(default=0.25, ge=0.0)
    keyword_overlap: float = Field(default=0.15, ge=0.0)
    recency: float = Field(default=0.15, ge=0.0)
    citations: float = Field(default=0.10, ge=0.0)

    def total(self) -> float:
        return (
            self.title_match
            + self.abstract_match
            + self.keyword_overlap
            + self.recency
            + self.citations
        )


class RankingConfig(BaseModel):
    weights: RankingWeights = Field(default_factory=RankingWeights)
    # Papers older than this contribute progressively less to the recency signal.
    recency_half_life_years: float = Field(default=4.0, gt=0.0)
    # Citation counts are power-law distributed; log-compress before normalising.
    citation_saturation: int = Field(default=500, ge=1)


class DiscoverySettings(BaseModel):
    """Runtime bounds for one discovery pass."""

    results_per_query: int = Field(default=15, ge=1, le=100)
    max_queries: int = Field(default=8, ge=1, le=50)
    max_candidates: int = Field(default=60, ge=1, le=500)
    require_retrievable: bool = False


class RetrievalSettings(BaseModel):
    max_concurrent_downloads: int = Field(default=4, ge=1, le=16)
    skip_existing: bool = True


class SourceSettings(BaseModel):
    """One provider entry in ``config/sources.yaml``."""

    enabled: bool = True
    # Public APIs, public limits. Semantic Scholar throttles hardest without a key;
    # NCBI blocks above 3/s. These defaults keep every provider inside its policy.
    requests_per_second: float = Field(default=3.0, gt=0.0, le=50.0)
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    max_results: int = Field(default=25, ge=1, le=200)


class SourcesConfig(BaseModel):
    """``config/sources.yaml`` root."""

    # Crossref and OpenAlex give faster, more reliable service to identified clients.
    contact_email: str | None = None
    results_per_query: int = Field(default=15, ge=1, le=100)
    max_queries: int = Field(default=8, ge=1, le=50)
    max_candidates: int = Field(default=60, ge=1, le=500)
    require_retrievable: bool = False

    manual_library_dir: Path = Path("storage/papers/raw/manual")
    download_dir: Path = Path("storage/papers/raw/downloaded")
    metadata_dir: Path = Path("storage/papers/metadata")

    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)

    sources: dict[SourceName, SourceSettings] = Field(default_factory=dict)

    def enabled_sources(self) -> list[SourceName]:
        return [name for name, settings in self.sources.items() if settings.enabled]

    def settings_for(self, name: SourceName) -> SourceSettings:
        return self.sources.get(name, SourceSettings())

    def discovery_settings(self) -> DiscoverySettings:
        return DiscoverySettings(
            results_per_query=self.results_per_query,
            max_queries=self.max_queries,
            max_candidates=self.max_candidates,
            require_retrievable=self.require_retrievable,
        )
