"""Retrieval implementations: lexical, sparse, dense and hybrid.

All are implementations of the same ports. The deterministic lexical retriever from v0.6
remains the baseline and is unchanged; these are alternatives to compare against it, not
replacements for it.
"""

from researchagent.services.retrieval.bm25 import BM25Index, BM25KnowledgeRetriever
from researchagent.services.retrieval.fusion import (
    ComponentRole,
    HybridKnowledgeRetriever,
    RetrieverComponent,
)
from researchagent.services.retrieval.indexer import IndexReport, KnowledgeIndexer
from researchagent.services.retrieval.representation import (
    REPRESENTATION_VERSION,
    RetrievalRepresentation,
    represent,
)
from researchagent.services.retrieval.semantic import SemanticKnowledgeRetriever

__all__ = [
    "REPRESENTATION_VERSION",
    "BM25Index",
    "BM25KnowledgeRetriever",
    "ComponentRole",
    "HybridKnowledgeRetriever",
    "IndexReport",
    "KnowledgeIndexer",
    "RetrievalRepresentation",
    "RetrieverComponent",
    "SemanticKnowledgeRetriever",
    "represent",
]
