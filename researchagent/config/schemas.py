"""Typed schemas for the YAML files under ``config/``.

Changing a model, a temperature or an agent's model binding must never require a
code change — only an edit to the corresponding YAML file.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from researchagent.core.interfaces.llm import GenerationParams
from researchagent.core.retry import RetryPolicy
from researchagent.models.knowledge import KnowledgeKind
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

    def with_provider_override(self, provider: str | None, model: str | None) -> ModelCatalog:
        """Apply ``LLM_PROVIDER`` / ``LLM_MODEL`` on top of the catalogue.

        Per-alias ``provider:`` entries remain the primary mechanism — they are what lets
        one agent reason remotely while extraction stays local. This override exists for
        the coarser case of running the whole pipeline on one provider, and is a no-op
        when neither variable is set (the local-first default).
        """
        if provider is None and model is None:
            return self
        return self.model_copy(
            update={
                "models": {
                    alias: spec.model_copy(
                        update={
                            key: value
                            for key, value in (("provider", provider), ("model_name", model))
                            if value is not None
                        }
                    )
                    for alias, spec in self.models.items()
                }
            }
        )

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
        """Agent-specific settings layered over ``defaults``.

        Re-validated rather than ``model_copy``-ed: ``model_dump`` turns nested models
        into dicts and ``model_copy(update=...)`` does not validate, so an agent
        overriding a nested block — a per-agent ``retry`` policy, say — would receive a
        plain dict where the code expects a ``RetryPolicy``.
        """
        override = self.agents.get(agent_name)
        if override is None:
            return self.defaults
        merged = self.defaults.model_dump() | override.model_dump(exclude_unset=True)
        return AgentSpec.model_validate(merged)


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


class SectionDetectionConfig(BaseModel):
    """Heading-detection thresholds. Relative to each document's own body font, so these
    hold across two-column ACM papers and single-column preprints alike."""

    heading_size_ratio: float = Field(default=1.08, gt=1.0)
    max_heading_words: int = Field(default=12, ge=1)
    max_heading_chars: int = Field(default=120, ge=10)
    min_paragraph_chars: int = Field(default=2, ge=1)
    min_confidence: float = Field(default=0.35, ge=0.0, le=1.0)


class DocumentValidationConfig(BaseModel):
    """What counts as an acceptable document."""

    min_pages: int = Field(default=1, ge=1)
    # Below this, a "digital" PDF is almost certainly a scan; OCR is out of scope.
    min_characters_per_page: int = Field(default=200, ge=0)
    max_empty_page_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    min_sections: int = Field(default=2, ge=0)
    min_body_words: int = Field(default=500, ge=0)
    # Title agreement below this is reported: the index and the PDF disagree.
    title_similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    min_citation_resolution: float = Field(default=0.5, ge=0.0, le=1.0)


class DocumentPipelineSettings(BaseModel):
    max_concurrent_documents: int = Field(default=4, ge=1, le=16)
    # Re-parsing is deterministic; skip when the stored document came from the same bytes.
    skip_unchanged: bool = True
    max_documents_per_run: int = Field(default=25, ge=1, le=500)


class DocumentsConfig(BaseModel):
    """``config/documents.yaml`` root: the document intelligence engine's tuning."""

    documents_dir: Path = Path("storage/papers/documents")
    sections: SectionDetectionConfig = Field(default_factory=SectionDetectionConfig)
    validation: DocumentValidationConfig = Field(default_factory=DocumentValidationConfig)
    pipeline: DocumentPipelineSettings = Field(default_factory=DocumentPipelineSettings)


class KnowledgeValidationConfig(BaseModel):
    """What counts as an acceptable knowledge object."""

    min_name_chars: int = Field(default=2, ge=1)
    min_claim_chars: int = Field(default=25, ge=0)
    # Named entities should appear in their own supporting quote; claim-like kinds are
    # exempt because a limitation is a statement, not a name.
    require_name_in_quote: bool = True


class KnowledgePipelineSettings(BaseModel):
    max_concurrent_documents: int = Field(default=2, ge=1, le=8)
    max_documents_per_run: int = Field(default=10, ge=1, le=200)
    skip_unchanged: bool = True
    # How close a model-supplied quote must be to real document text. High on purpose:
    # below this the honest answer is "not found", and the extraction is discarded.
    grounding_similarity_threshold: float = Field(default=0.85, ge=0.5, le=1.0)


class KnowledgeConfig(BaseModel):
    """``config/knowledge.yaml`` root."""

    knowledge_dir: Path = Path("storage/papers/knowledge")
    # Extraction is near-deterministic work; the alias binds it to a low-temperature model.
    model: str | None = "extraction"
    prompt_version: str = "v1"
    enabled_extractors: tuple[str, ...] = (
        "method_extractor",
        "dataset_extractor",
        "metric_extractor",
        "result_extractor",
        "limitation_extractor",
        "future_work_extractor",
    )
    validation: KnowledgeValidationConfig = Field(default_factory=KnowledgeValidationConfig)
    pipeline: KnowledgePipelineSettings = Field(default_factory=KnowledgePipelineSettings)


class RetrievalWeights(BaseModel):
    """Relative influence of each retrieval signal.

    Deterministic and tunable today; v0.7 adds embedding similarity as another weighted
    signal rather than as a replacement, so lexical and semantic scores can be compared
    and benchmarked against each other.
    """

    name_match: float = Field(default=1.0, ge=0.0)
    text_match: float = Field(default=0.8, ge=0.0)
    validation_confidence: float = Field(default=0.6, ge=0.0)
    evidence_density: float = Field(default=0.4, ge=0.0)
    provenance_precision: float = Field(default=0.5, ge=0.0)
    cross_paper_agreement: float = Field(default=0.9, ge=0.0)


class ContradictionConfig(BaseModel):
    """How far two reported numbers must diverge before it counts as a disagreement."""

    numeric_tolerance: float = Field(default=0.05, ge=0.0, le=1.0)


class BundleSettings(BaseModel):
    max_objects: int = Field(default=25, ge=1, le=200)
    max_evidence_per_object: int = Field(default=3, ge=1, le=20)
    min_object_score: float = Field(default=0.05, ge=0.0, le=1.0)
    # Papers needed before a bundle stops being flagged as single-source.
    min_papers_for_confidence: int = Field(default=2, ge=1, le=20)
    corroboration_target: int = Field(default=3, ge=1, le=50)


class EvidencePipelineSettings(BaseModel):
    max_bundles_per_run: int = Field(default=8, ge=1, le=50)
    max_objects_per_bundle: int = Field(default=25, ge=1, le=200)
    # Empty means every kind; narrowing is a per-deployment tuning decision.
    bundle_kinds: tuple[KnowledgeKind, ...] = ()


class EvidenceConfig(BaseModel):
    """``config/evidence.yaml`` root."""

    evidence_dir: Path = Path("storage/papers/evidence")
    bundles_dir: Path = Path("storage/bundles")
    weights: RetrievalWeights = Field(default_factory=RetrievalWeights)
    contradictions: ContradictionConfig = Field(default_factory=ContradictionConfig)
    bundles: BundleSettings = Field(default_factory=BundleSettings)
    pipeline: EvidencePipelineSettings = Field(default_factory=EvidencePipelineSettings)


class FusionStrategy(StrEnum):
    RECIPROCAL_RANK = "reciprocal_rank"
    WEIGHTED_SCORE = "weighted_score"


class BM25Settings(BaseModel):
    """Standard BM25 parameters. k1 controls term-frequency saturation, b length
    normalisation; the defaults are the usual starting point, not a tuned result."""

    k1: float = Field(default=1.5, gt=0.0, le=3.0)
    b: float = Field(default=0.75, ge=0.0, le=1.0)


class FusionSettings(BaseModel):
    strategy: FusionStrategy = FusionStrategy.RECIPROCAL_RANK
    # Each component retrieves this many times the requested limit, so fusion has depth
    # to reorder rather than reordering an already-truncated list.
    candidate_multiplier: int = Field(default=3, ge=1, le=20)
    # RRF's damping constant. 60 is the value from the original paper.
    rrf_k: int = Field(default=60, ge=1, le=1000)
    lexical_weight: float = Field(default=1.0, ge=0.0)
    sparse_weight: float = Field(default=1.0, ge=0.0)
    dense_weight: float = Field(default=1.0, ge=0.0)


class EmbeddingSettings(BaseModel):
    """Never hardcode a model name: switching models is a config change that forces a
    new index version, because the model is part of the index identity."""

    enabled: bool = True
    provider: str = "ollama"
    model: str = "nomic-embed-text"
    batch_size: int = Field(default=32, ge=1, le=512)
    timeout_seconds: float = Field(default=120.0, gt=0)
    preprocessing_version: str = "1"


class VectorStoreSettings(BaseModel):
    # `memory` keeps the full semantic stack working offline and in tests; `qdrant` is
    # the production adapter. Both implement the same port.
    backend: str = "memory"
    url: str = "http://localhost:6333"
    collection_prefix: str = "researchagent_knowledge"
    api_key: str | None = None
    timeout_seconds: float = Field(default=30.0, gt=0)


class IndexSettings(BaseModel):
    # Bumped by hand when the index schema changes for a reason the model identity does
    # not already capture.
    schema_version: str = "1"
    rebuild_on_start: bool = False


class RetrievalConfig(BaseModel):
    """``config/retrieval.yaml`` root."""

    # The retriever the pipeline actually uses. `deterministic` is the v0.6 baseline and
    # remains the default: semantic retrieval is opt-in until the benchmark justifies it.
    active_retriever: str = "deterministic"
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    index: IndexSettings = Field(default_factory=IndexSettings)
    bm25: BM25Settings = Field(default_factory=BM25Settings)
    fusion: FusionSettings = Field(default_factory=FusionSettings)


class GraphNeo4jSettings(BaseModel):
    database: str = "neo4j"


class GraphBuildSettings(BaseModel):
    require_trusted_knowledge: bool = True
    require_provenance: bool = True


class GraphSchemaSettings(BaseModel):
    """Identity of a graph generation, independent of the corpus it was built from."""

    version: str = "1"
    extraction_version: str = "1"
    relation_version: str = "1"


class GraphConfig(BaseModel):
    """``config/graph.yaml`` root."""

    # `memory` is the default so the graph works offline and the test suite never needs a
    # running Neo4j. Persisting is a one-line config change.
    backend: Literal["memory", "json", "neo4j"] = "memory"
    neo4j: GraphNeo4jSettings = Field(default_factory=GraphNeo4jSettings)
    build: GraphBuildSettings = Field(default_factory=GraphBuildSettings)
    schema_identity: GraphSchemaSettings = Field(
        default_factory=GraphSchemaSettings, alias="schema"
    )

    model_config = {"populate_by_name": True}


class ResearchBudget(BaseModel):
    """Hard limits on an autonomous agent loop.

    An agent loop without a budget is a bill, not a system. Every limit here is a reason
    the run can terminate, and each one is reported rather than silently hit.
    """

    max_iterations: int = Field(default=3, ge=1, le=20)
    max_retrieval_attempts: int = Field(default=8, ge=1, le=100)
    max_tool_calls: int = Field(default=40, ge=1, le=1000)
    max_tokens_per_agent: int = Field(default=32_000, ge=256)
    max_total_tokens: int = Field(default=200_000, ge=1024)

    @model_validator(mode="after")
    def _agent_budget_fits_the_total(self) -> ResearchBudget:
        if self.max_tokens_per_agent > self.max_total_tokens:
            raise ValueError(
                f"max_tokens_per_agent ({self.max_tokens_per_agent}) exceeds "
                f"max_total_tokens ({self.max_total_tokens})"
            )
        return self


class ReasoningLoopSettings(BaseModel):
    max_questions_per_round: int = Field(default=3, ge=1, le=10)
    max_findings_to_verify: int = Field(default=6, ge=1, le=50)


class ReasoningReviewSettings(BaseModel):
    minimum_papers_per_finding: int = Field(default=2, ge=1, le=10)


class ReasoningConfig(BaseModel):
    """``config/reasoning.yaml`` root."""

    budget: ResearchBudget = Field(default_factory=ResearchBudget)
    loop: ReasoningLoopSettings = Field(default_factory=ReasoningLoopSettings)
    review: ReasoningReviewSettings = Field(default_factory=ReasoningReviewSettings)
