"""Composition root.

The single place where concrete implementations are chosen and wired. Everything else
receives its collaborators by injection, which is what keeps the inner layers testable
without Docker, Ollama or a network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from researchagent.agents.base import BaseAgent
from researchagent.agents.registry import AGENTS, build_agent
from researchagent.config.loader import ConfigLoader
from researchagent.config.schemas import (
    AgentConfig,
    DocumentsConfig,
    EvidenceConfig,
    GraphConfig,
    KnowledgeConfig,
    ModelCatalog,
    ReasoningConfig,
    RetrievalConfig,
    SourcesConfig,
    WorkflowConfig,
)
from researchagent.core.events import EventBus
from researchagent.core.interfaces.embeddings import EmbeddingModel
from researchagent.core.interfaces.graph_repository import GraphRepository
from researchagent.core.interfaces.paper_source import PaperSource
from researchagent.core.interfaces.retrieval import KnowledgeRetriever
from researchagent.core.interfaces.vector_store import VectorStore
from researchagent.core.logging import configure_logging, get_logger
from researchagent.core.prompts import PromptLibrary
from researchagent.core.settings import Settings, get_settings
from researchagent.integrations.memory_graph import InMemoryGraphRepository
from researchagent.integrations.neo4j import Neo4jGraphRepository
from researchagent.integrations.pymupdf import PyMuPDFLoader
from researchagent.integrations.sources import build_enabled_sources
from researchagent.memory.checkpoints import build_checkpointer
from researchagent.repositories.bundle_repository import JsonBundleRepository
from researchagent.repositories.document_repository import JsonDocumentRepository
from researchagent.repositories.evidence_repository import JsonEvidenceRepository
from researchagent.repositories.knowledge_repository import JsonKnowledgeRepository
from researchagent.repositories.paper_repository import JsonPaperRepository
from researchagent.services.audit import AuditTrailBuilder
from researchagent.services.deduplication import PaperDeduplicator
from researchagent.services.discovery_service import DiscoveryService
from researchagent.services.document import (
    CitationExtractor,
    DocumentAssembler,
    DocumentIntelligenceService,
    FigureTableDetector,
    MetadataExtractor,
    ReferenceExtractor,
    SectionDetector,
)
from researchagent.services.evidence import (
    AgreementCrossPaperRetriever,
    ContradictionDetector,
    EvidenceBundleBuilder,
    EvidenceIndexer,
    EvidenceIntelligenceService,
    LexicalKnowledgeRetriever,
    LinkedEvidenceRetriever,
    RepositoryDocumentRetriever,
    StoredBundleRetriever,
)
from researchagent.services.graph.builder import GraphBuilder
from researchagent.services.graph.mapper import GraphMapper
from researchagent.services.graph.queries import GraphQueries
from researchagent.services.graph.validator import GraphValidator
from researchagent.services.knowledge import KnowledgeIntelligenceService, RelationBuilder
from researchagent.services.knowledge.registry import build_extractors
from researchagent.services.llm_service import LLMService
from researchagent.services.ranking import HeuristicScorer
from researchagent.services.retrieval import KnowledgeIndexer
from researchagent.services.retrieval.registry import (
    build_embedding_model,
    build_retrieval_arms,
    build_vector_store,
    select_active,
)
from researchagent.services.retrieval_service import RetrievalService
from researchagent.services.tools import ServiceToolbox
from researchagent.workflows.reasoning_runner import ReasoningRunner
from researchagent.workflows.research import build_research_graph
from researchagent.workflows.runner import WorkflowRunner

logger = get_logger(__name__)

# Agents permitted an I/O surface. Reasoning and review reason over what they are handed;
# only retrieval and verification need to go and look.
_TOOLBOX_AGENTS = frozenset({"retrieval", "verification"})


@dataclass(slots=True)
class Container:
    """Application-scoped singletons."""

    settings: Settings
    config_loader: ConfigLoader
    model_catalog: ModelCatalog
    agent_config: AgentConfig
    workflow_config: WorkflowConfig
    sources_config: SourcesConfig
    documents_config: DocumentsConfig
    knowledge_config: KnowledgeConfig
    evidence_config: EvidenceConfig
    retrieval_config: RetrievalConfig
    prompt_library: PromptLibrary
    event_bus: EventBus
    llm_service: LLMService
    paper_sources: list[PaperSource]
    paper_repository: JsonPaperRepository
    discovery_service: DiscoveryService
    retrieval_service: RetrievalService
    document_repository: JsonDocumentRepository
    document_service: DocumentIntelligenceService
    knowledge_repository: JsonKnowledgeRepository
    knowledge_service: KnowledgeIntelligenceService
    evidence_repository: JsonEvidenceRepository
    bundle_repository: JsonBundleRepository
    evidence_service: EvidenceIntelligenceService
    # Retrieval layers exposed directly: the API and, from v0.9, the reasoning engine
    # query them without going through the pipeline that populated them.
    knowledge_retriever: LexicalKnowledgeRetriever
    evidence_retriever: LinkedEvidenceRetriever
    document_retriever: RepositoryDocumentRetriever
    cross_paper_retriever: AgreementCrossPaperRetriever
    bundle_retriever: StoredBundleRetriever
    # Every retrieval strategy stays constructed, whichever one is active. The benchmark
    # compares them, and switching is a config edit rather than a code change.
    embedding_model: EmbeddingModel
    vector_store: VectorStore
    knowledge_indexer: KnowledgeIndexer
    retrieval_arms: dict[str, KnowledgeRetriever]
    active_knowledge_retriever: KnowledgeRetriever
    # v0.8. A derived index: the knowledge and evidence repositories stay authoritative,
    # and `graph_builder.build()` reconstructs everything below from them.
    graph_config: GraphConfig
    graph_repository: GraphRepository
    graph_builder: GraphBuilder
    graph_queries: GraphQueries
    # v0.9. The toolbox is the agents' only I/O surface; agent factories bind it per
    # agent and iteration so every tool call is attributable.
    reasoning_config: ReasoningConfig
    toolbox: ServiceToolbox
    audit_trail: AuditTrailBuilder
    reasoning_runner: ReasoningRunner
    workflow_runner: WorkflowRunner

    async def aclose(self) -> None:
        await self.llm_service.aclose()
        await self.embedding_model.aclose()
        await self.vector_store.aclose()
        await self.graph_repository.aclose()
        for source in self.paper_sources:
            await source.aclose()


def build_container(settings: Settings | None = None) -> Container:
    """Wire the object graph. Cheap and synchronous: no network calls happen here."""
    settings = settings or get_settings()
    configure_logging(settings)

    loader = ConfigLoader(settings.config_dir)
    model_catalog = loader.load("models", ModelCatalog).with_provider_override(
        settings.llm_provider, settings.llm_model
    )
    agent_config = loader.load("agents", AgentConfig)
    workflow_config = loader.load("workflow", WorkflowConfig)
    sources_config = loader.load("sources", SourcesConfig)
    documents_config = loader.load("documents", DocumentsConfig)
    knowledge_config = loader.load("knowledge", KnowledgeConfig)
    evidence_config = loader.load("evidence", EvidenceConfig)
    retrieval_config = loader.load("retrieval", RetrievalConfig)

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
        event_bus=event_bus,
    )
    retrieval_service = RetrievalService(
        {source.name: source for source in paper_sources},
        paper_repository,
        _resolve(sources_config.download_dir, settings.project_root),
        sources_config.retrieval,
    )

    document_repository = JsonDocumentRepository(
        _resolve(documents_config.documents_dir, settings.project_root)
    )
    document_service = DocumentIntelligenceService(
        PyMuPDFLoader(),
        DocumentAssembler(
            SectionDetector(documents_config.sections),
            ReferenceExtractor(),
            CitationExtractor(),
            FigureTableDetector(),
            MetadataExtractor(),
        ),
        document_repository,
        paper_repository,
        documents_config.pipeline,
        documents_config.validation,
        event_bus=event_bus,
    )

    knowledge_repository = JsonKnowledgeRepository(
        _resolve(knowledge_config.knowledge_dir, settings.project_root)
    )
    knowledge_service = KnowledgeIntelligenceService(
        build_extractors(
            knowledge_config.enabled_extractors,
            llm_service.get(knowledge_config.model),
            prompt_library,
            prompt_version=knowledge_config.prompt_version,
        ),
        RelationBuilder(),
        knowledge_repository,
        document_repository,
        paper_repository,
        knowledge_config.pipeline,
        knowledge_config.validation,
        event_bus=event_bus,
    )

    evidence_repository = JsonEvidenceRepository(
        _resolve(evidence_config.evidence_dir, settings.project_root)
    )
    bundle_repository = JsonBundleRepository(
        _resolve(evidence_config.bundles_dir, settings.project_root)
    )
    knowledge_retriever = LexicalKnowledgeRetriever(knowledge_repository, evidence_config.weights)
    evidence_retriever = LinkedEvidenceRetriever(evidence_repository, evidence_config.weights)
    document_retriever = RepositoryDocumentRetriever(document_repository)
    cross_paper_retriever = AgreementCrossPaperRetriever(
        knowledge_retriever, evidence_config.weights
    )
    bundle_retriever = StoredBundleRetriever(bundle_repository)
    embedding_model = build_embedding_model(
        retrieval_config.embeddings, base_url=settings.ollama.base_url
    )
    vector_store = build_vector_store(retrieval_config.vector_store)
    knowledge_indexer = KnowledgeIndexer(
        embedding_model,
        vector_store,
        knowledge_repository,
        retrieval_config.index,
        event_bus=event_bus,
    )
    retrieval_arms = build_retrieval_arms(
        retrieval_config, knowledge_repository, knowledge_retriever, embedding_model, vector_store
    )
    active_knowledge_retriever = select_active(retrieval_arms, retrieval_config)

    evidence_service = EvidenceIntelligenceService(
        EvidenceIndexer(evidence_repository, event_bus=event_bus),
        EvidenceBundleBuilder(
            active_knowledge_retriever,
            evidence_retriever,
            cross_paper_retriever,
            ContradictionDetector(evidence_config.contradictions),
            evidence_config.bundles,
        ),
        knowledge_repository,
        bundle_repository,
        evidence_config.pipeline,
        event_bus=event_bus,
    )

    graph_config = loader.load("graph", GraphConfig)
    graph_repository = _build_graph_repository(graph_config, settings)
    graph_builder = GraphBuilder(
        knowledge_repository,
        graph_repository,
        GraphMapper(
            schema_version=graph_config.schema_identity.version,
            extraction_version=graph_config.schema_identity.extraction_version,
            relation_version=graph_config.schema_identity.relation_version,
        ),
        GraphValidator(require_provenance=graph_config.build.require_provenance),
        ContradictionDetector(evidence_config.contradictions),
        paper_repository,
        event_bus=event_bus,
    )
    graph_queries = GraphQueries(graph_repository)

    reasoning_config = loader.load("reasoning", ReasoningConfig)
    toolbox = ServiceToolbox(
        active_knowledge_retriever,
        evidence_service,
        knowledge_repository,
        evidence_repository,
        paper_repository,
        bundle_repository,
        graph_repository,
        graph_queries,
    )
    audit_trail = AuditTrailBuilder(bundle_repository, evidence_repository)

    def agent_for(name: str, iteration: int) -> BaseAgent[Any, Any]:
        """Build an agent bound to this iteration's toolbox view.

        Agents that take a toolbox receive one attributed to them, so the tool-call
        ledger records which agent asked for what without the agent knowing it is logged.
        """
        spec = agent_config.spec_for(name)
        agent_cls = AGENTS.get(name)
        kwargs: dict[str, Any] = {"event_bus": event_bus}
        if name in _TOOLBOX_AGENTS:
            kwargs["toolbox"] = toolbox.for_agent(name, iteration)
        return agent_cls(llm_service.get(spec.model), spec, prompt_library, **kwargs)

    reasoning_runner = ReasoningRunner(
        agent_for, bundle_repository, reasoning_config, event_bus=event_bus
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
        documents=document_service,
        knowledge=knowledge_service,
        evidence=evidence_service,
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
        documents_config=documents_config,
        knowledge_config=knowledge_config,
        evidence_config=evidence_config,
        retrieval_config=retrieval_config,
        prompt_library=prompt_library,
        event_bus=event_bus,
        llm_service=llm_service,
        paper_sources=paper_sources,
        paper_repository=paper_repository,
        discovery_service=discovery_service,
        retrieval_service=retrieval_service,
        document_repository=document_repository,
        document_service=document_service,
        knowledge_repository=knowledge_repository,
        knowledge_service=knowledge_service,
        evidence_repository=evidence_repository,
        bundle_repository=bundle_repository,
        evidence_service=evidence_service,
        knowledge_retriever=knowledge_retriever,
        evidence_retriever=evidence_retriever,
        document_retriever=document_retriever,
        cross_paper_retriever=cross_paper_retriever,
        bundle_retriever=bundle_retriever,
        embedding_model=embedding_model,
        vector_store=vector_store,
        knowledge_indexer=knowledge_indexer,
        retrieval_arms=retrieval_arms,
        active_knowledge_retriever=active_knowledge_retriever,
        graph_config=graph_config,
        graph_repository=graph_repository,
        graph_builder=graph_builder,
        graph_queries=graph_queries,
        reasoning_config=reasoning_config,
        toolbox=toolbox,
        audit_trail=audit_trail,
        reasoning_runner=reasoning_runner,
        workflow_runner=WorkflowRunner(graph, workflow_config),
    )


def _resolve(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _build_graph_repository(config: GraphConfig, settings: Settings) -> GraphRepository:
    """Choose the graph backend.

    Kept here rather than in a registry because there are exactly two and one of them is a
    test seam. Credentials come from ``Settings``; ``config/graph.yaml`` holds no secrets.
    """
    if config.backend == "neo4j":
        return Neo4jGraphRepository(
            uri=settings.neo4j.uri,
            user=settings.neo4j.user,
            password=settings.neo4j.password.get_secret_value(),
            database=config.neo4j.database,
        )
    return InMemoryGraphRepository()
