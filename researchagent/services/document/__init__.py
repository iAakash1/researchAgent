"""Document intelligence: PDF bytes to a validated canonical paper.

Detection is pure logic over ``models.layout`` structures, so every rule here is
testable on synthetic pages without a PDF fixture.
"""

from researchagent.services.document.assembler import DocumentAssembler
from researchagent.services.document.figures import FigureTableDetector
from researchagent.services.document.metadata import MetadataExtractor
from researchagent.services.document.pipeline import DocumentIntelligenceService
from researchagent.services.document.references import CitationExtractor, ReferenceExtractor
from researchagent.services.document.sections import SectionDetector, classify_section

__all__ = [
    "CitationExtractor",
    "DocumentAssembler",
    "DocumentIntelligenceService",
    "FigureTableDetector",
    "MetadataExtractor",
    "ReferenceExtractor",
    "SectionDetector",
    "classify_section",
]
