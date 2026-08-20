"""Retriever construction.

``config/retrieval.yaml`` names the active retriever as a string; this builds it. Every
arm stays available regardless of which is active, so the benchmark can compare them and
a bad result can be reverted with a one-line config change rather than a rollback.
"""

from __future__ import annotations

from researchagent.config.schemas import (
    EmbeddingSettings,
    RetrievalConfig,
    VectorStoreSettings,
)
from researchagent.core.interfaces.embeddings import EmbeddingModel
from researchagent.core.interfaces.repositories import KnowledgeRepository
from researchagent.core.interfaces.retrieval import KnowledgeRetriever
from researchagent.core.interfaces.vector_store import VectorStore
from researchagent.core.logging import get_logger
from researchagent.integrations.memory_store import InMemoryVectorStore
from researchagent.integrations.ollama import NullEmbeddingModel, OllamaEmbeddingModel
from researchagent.services.retrieval.bm25 import BM25KnowledgeRetriever
from researchagent.services.retrieval.fusion import (
    ComponentRole,
    HybridKnowledgeRetriever,
    RetrieverComponent,
)
from researchagent.services.retrieval.lexical import LexicalKnowledgeRetriever
from researchagent.services.retrieval.semantic import SemanticKnowledgeRetriever

logger = get_logger(__name__)


def build_embedding_model(settings: EmbeddingSettings, *, base_url: str) -> EmbeddingModel:
    if not settings.enabled:
        return NullEmbeddingModel()
    if settings.provider != "ollama":
        raise ValueError(f"unknown embedding provider {settings.provider!r}")
    return OllamaEmbeddingModel(
        settings.model,
        base_url=base_url,
        timeout_seconds=settings.timeout_seconds,
        batch_size=settings.batch_size,
        preprocessing_version=settings.preprocessing_version,
    )


def build_vector_store(settings: VectorStoreSettings) -> VectorStore:
    if settings.backend == "memory":
        return InMemoryVectorStore()
    if settings.backend == "qdrant":
        from researchagent.integrations.qdrant import QdrantVectorStore

        return QdrantVectorStore(
            url=settings.url,
            collection_prefix=settings.collection_prefix,
            api_key=settings.api_key,
            timeout_seconds=settings.timeout_seconds,
        )
    raise ValueError(f"unknown vector store backend {settings.backend!r}")


def build_retrieval_arms(
    config: RetrievalConfig,
    knowledge: KnowledgeRepository,
    lexical: LexicalKnowledgeRetriever,
    embeddings: EmbeddingModel,
    store: VectorStore,
) -> dict[str, KnowledgeRetriever]:
    """Every retrieval strategy, built and named. The benchmark compares these directly."""
    bm25 = BM25KnowledgeRetriever(knowledge, config.bm25)
    semantic = SemanticKnowledgeRetriever(embeddings, store, knowledge)
    hybrid = HybridKnowledgeRetriever(
        [
            RetrieverComponent(lexical, ComponentRole.LEXICAL, config.fusion.lexical_weight),
            RetrieverComponent(bm25, ComponentRole.SPARSE, config.fusion.sparse_weight),
            RetrieverComponent(semantic, ComponentRole.DENSE, config.fusion.dense_weight),
        ],
        config.fusion,
    )
    return {
        "deterministic": lexical,
        "bm25": bm25,
        "semantic": semantic,
        "hybrid": hybrid,
    }


def select_active(
    arms: dict[str, KnowledgeRetriever], config: RetrievalConfig
) -> KnowledgeRetriever:
    active = arms.get(config.active_retriever)
    if active is None:
        raise ValueError(
            f"unknown active_retriever {config.active_retriever!r}; available: {sorted(arms)}"
        )
    logger.info("active_retriever_selected", retriever=config.active_retriever)
    return active
