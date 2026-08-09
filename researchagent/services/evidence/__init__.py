"""Evidence intelligence: retrieval over evidence, not text.

The system retrieves knowledge, walks to the evidence supporting it, and assembles
:class:`EvidenceBundle` objects — the canonical unit every future reasoning engine
consumes instead of arbitrary context.
"""

from researchagent.services.evidence.builder import EvidenceBundleBuilder
from researchagent.services.evidence.contradictions import ContradictionDetector
from researchagent.services.evidence.indexer import EvidenceIndexer
from researchagent.services.evidence.pipeline import EvidenceIntelligenceService
from researchagent.services.evidence.retrievers import (
    AgreementCrossPaperRetriever,
    LexicalKnowledgeRetriever,
    LinkedEvidenceRetriever,
    RepositoryDocumentRetriever,
    StoredBundleRetriever,
)

__all__ = [
    "AgreementCrossPaperRetriever",
    "ContradictionDetector",
    "EvidenceBundleBuilder",
    "EvidenceIndexer",
    "EvidenceIntelligenceService",
    "LexicalKnowledgeRetriever",
    "LinkedEvidenceRetriever",
    "RepositoryDocumentRetriever",
    "StoredBundleRetriever",
]
