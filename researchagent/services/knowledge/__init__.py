"""Knowledge intelligence: validated documents to evidence-backed knowledge objects.

Not summarisation. Each extractor produces typed facts whose quotes are located in the
source document before they are believed.
"""

from researchagent.services.knowledge.base import ExtractionOutcome, KnowledgeExtractor
from researchagent.services.knowledge.grounding import EvidenceGrounder, GroundedQuote
from researchagent.services.knowledge.pipeline import KnowledgeIntelligenceService
from researchagent.services.knowledge.relations import RelationBuilder

__all__ = [
    "EvidenceGrounder",
    "ExtractionOutcome",
    "GroundedQuote",
    "KnowledgeExtractor",
    "KnowledgeIntelligenceService",
    "RelationBuilder",
]
