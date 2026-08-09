"""Document validators.

One validator, one question. Each returns a :class:`ValidationResult` whose confidence
is built only from counts it measured itself — never a hand-picked number.

The chain is deliberately staged: an unreadable PDF is caught before section detection
wastes time on it, and a document with no body text is rejected before knowledge
extraction is asked to reason about nothing.
"""

from __future__ import annotations

from researchagent.config.schemas import DocumentValidationConfig
from researchagent.core.evidence import Evidence, EvidenceKind, SourceLocation
from researchagent.core.interfaces.validator import Validator
from researchagent.core.validation import (
    Confidence,
    ConfidenceSignal,
    ValidationIssue,
    ValidationResult,
)
from researchagent.models.document import PaperDocument, SectionKind
from researchagent.models.layout import RawDocument
from researchagent.models.paper import Paper, normalise_title

_SUBJECT_RAW = "RawDocument"
_SUBJECT_DOCUMENT = "PaperDocument"


class PDFValidator(Validator[RawDocument]):
    """Is this a usable digital PDF at all?

    The first gate. Everything downstream assumes extractable text exists, so a scanned
    document must be rejected here with an actionable remedy rather than silently
    producing an empty :class:`PaperDocument`.
    """

    name = "pdf_validator"
    subject_type = _SUBJECT_RAW

    def __init__(self, config: DocumentValidationConfig | None = None) -> None:
        self._config = config or DocumentValidationConfig()

    def check(self, subject: RawDocument) -> ValidationResult:
        issues: list[ValidationIssue] = []
        signals: list[ConfidenceSignal] = []
        evidence: list[Evidence] = []

        pages = subject.page_count
        characters = subject.character_count
        per_page = characters / pages if pages else 0.0
        empty_ratio = subject.empty_page_count / pages if pages else 1.0

        evidence.append(
            Evidence.structural(
                claim=f"Document has {pages} pages and {characters} extractable characters",
                document_id=subject.document_id,
                produced_by=self.name,
            )
        )

        if pages < self._config.min_pages:
            issues.append(
                ValidationIssue.fatal(
                    "pdf_no_pages", f"PDF has {pages} pages", field="pages", remedy="Re-download"
                )
            )
        if characters == 0:
            issues.append(
                ValidationIssue.fatal(
                    "pdf_no_text",
                    "No extractable text; the PDF is probably a scan",
                    field="characters",
                    remedy="Run OCR before ingestion (out of scope for this release)",
                )
            )
            evidence.append(
                Evidence.absence(
                    claim="No extractable text found on any page",
                    document_id=subject.document_id,
                    produced_by=self.name,
                )
            )
        elif per_page < self._config.min_characters_per_page:
            issues.append(
                ValidationIssue.warning(
                    "pdf_sparse_text",
                    f"Only {per_page:.0f} characters per page",
                    field="characters",
                    remedy="Check for embedded images or an unusual layout",
                )
            )

        if empty_ratio > self._config.max_empty_page_ratio:
            issues.append(
                ValidationIssue.warning(
                    "pdf_many_empty_pages",
                    f"{subject.empty_page_count} of {pages} pages have no text",
                    field="pages",
                )
            )

        signals.append(
            ConfidenceSignal(
                name="text_density",
                value=min(per_page / max(self._config.min_characters_per_page * 5, 1), 1.0),
                observation=f"{per_page:.0f} characters per page across {pages} pages",
            )
        )
        signals.append(
            ConfidenceSignal(
                name="page_coverage",
                value=1.0 - empty_ratio,
                observation=f"{pages - subject.empty_page_count} of {pages} pages carry text",
            )
        )

        return ValidationResult.decide(
            validator=self.name,
            subject_id=subject.document_id,
            subject_type=self.subject_type,
            confidence=Confidence.from_signals(signals),
            issues=issues,
            evidence=evidence,
        )


class SectionValidator(Validator[PaperDocument]):
    """Did section detection find a plausible paper structure?"""

    name = "section_validator"
    subject_type = _SUBJECT_DOCUMENT

    # A research paper that has none of these was almost certainly mis-segmented.
    _EXPECTED = (
        SectionKind.ABSTRACT,
        SectionKind.INTRODUCTION,
        SectionKind.METHODOLOGY,
        SectionKind.RESULTS,
        SectionKind.EVALUATION,
        SectionKind.CONCLUSION,
        SectionKind.REFERENCES,
    )

    def __init__(self, config: DocumentValidationConfig | None = None) -> None:
        self._config = config or DocumentValidationConfig()

    def check(self, subject: PaperDocument) -> ValidationResult:
        issues: list[ValidationIssue] = []
        evidence: list[Evidence] = []

        kinds = {section.kind for section in subject.sections}
        recognised = kinds & set(self._EXPECTED)
        body_words = sum(section.word_count for section in subject.body_sections)

        if len(subject.sections) < self._config.min_sections:
            issues.append(
                ValidationIssue.error(
                    "sections_missing",
                    f"Only {len(subject.sections)} sections detected",
                    field="sections",
                    remedy="The PDF layout may be unusual; inspect the heading detection",
                )
            )
        if not recognised:
            issues.append(
                ValidationIssue.error(
                    "sections_unrecognised",
                    "No canonical section (abstract, introduction, results, ...) was found",
                    field="sections",
                    remedy="Document may not be a research paper",
                )
            )
        if body_words < self._config.min_body_words:
            issues.append(
                ValidationIssue.warning(
                    "body_text_short",
                    f"Body text is only {body_words} words",
                    field="sections",
                )
            )

        if SectionKind.ABSTRACT in kinds:
            abstract = subject.first_section_of(SectionKind.ABSTRACT)
            if abstract is not None and abstract.text:
                evidence.append(
                    Evidence.from_text(
                        claim="Document contains an abstract",
                        quote=abstract.text[:400],
                        location=SourceLocation(
                            document_id=subject.paper_id,
                            page=abstract.page_start,
                            section_id=abstract.id,
                            section_title=abstract.title,
                        ),
                        produced_by=self.name,
                    )
                )
        else:
            evidence.append(
                Evidence.absence(
                    claim="No abstract section was detected",
                    document_id=subject.paper_id,
                    produced_by=self.name,
                )
            )

        mean_detection = (
            sum(section.detection_confidence for section in subject.sections)
            / len(subject.sections)
            if subject.sections
            else 0.0
        )
        signals = [
            ConfidenceSignal(
                name="canonical_sections",
                value=len(recognised) / len(self._EXPECTED),
                observation=(
                    f"{len(recognised)} of {len(self._EXPECTED)} canonical section kinds present: "
                    f"{sorted(k.value for k in recognised)}"
                ),
            ),
            ConfidenceSignal(
                name="heading_detection",
                value=mean_detection,
                observation=f"mean heading confidence {mean_detection:.2f} over "
                f"{len(subject.sections)} sections",
            ),
            ConfidenceSignal(
                name="body_volume",
                value=min(body_words / max(self._config.min_body_words * 4, 1), 1.0),
                observation=f"{body_words} words across {len(subject.body_sections)} body sections",
            ),
        ]

        return ValidationResult.decide(
            validator=self.name,
            subject_id=subject.paper_id,
            subject_type=self.subject_type,
            confidence=Confidence.from_signals(signals),
            issues=issues,
            evidence=evidence,
        )


class ReferenceValidator(Validator[PaperDocument]):
    """Were references found, and did any structure survive parsing?"""

    name = "reference_validator"
    subject_type = _SUBJECT_DOCUMENT

    def check(self, subject: PaperDocument) -> ValidationResult:
        issues: list[ValidationIssue] = []
        references = subject.references
        structured = [reference for reference in references if reference.is_structured]

        if not references:
            issues.append(
                ValidationIssue.warning(
                    "references_missing",
                    "No references were extracted",
                    field="references",
                    remedy="Check whether a references section was detected",
                )
            )
        elif not structured:
            issues.append(
                ValidationIssue.warning(
                    "references_unstructured",
                    f"All {len(references)} references are raw strings",
                    field="references",
                )
            )

        structure_rate = len(structured) / len(references) if references else 0.0
        with_identifier = sum(1 for r in references if r.doi or r.arxiv_id)

        signals = [
            ConfidenceSignal(
                name="reference_presence",
                value=min(len(references) / 20, 1.0),
                observation=f"{len(references)} bibliography entries extracted",
            ),
            ConfidenceSignal(
                name="reference_structure",
                value=structure_rate,
                observation=f"{len(structured)} of {len(references)} entries yielded fields",
            ),
            ConfidenceSignal(
                name="reference_identifiers",
                value=(with_identifier / len(references)) if references else 0.0,
                observation=f"{with_identifier} entries carry a DOI or arXiv id",
            ),
        ]

        return ValidationResult.decide(
            validator=self.name,
            subject_id=subject.paper_id,
            subject_type=self.subject_type,
            confidence=Confidence.from_signals(signals),
            issues=issues,
        )


class CitationValidator(Validator[PaperDocument]):
    """Do in-text citations line up with the bibliography?

    The resolution rate is the single best cheap proxy for parse quality: markers that
    point nowhere mean the references section was mis-detected or mis-split.
    """

    name = "citation_validator"
    subject_type = _SUBJECT_DOCUMENT

    def __init__(self, config: DocumentValidationConfig | None = None) -> None:
        self._config = config or DocumentValidationConfig()

    def check(self, subject: PaperDocument) -> ValidationResult:
        issues: list[ValidationIssue] = []
        citations = subject.citations
        resolved = [citation for citation in citations if citation.is_resolved]
        rate = len(resolved) / len(citations) if citations else 0.0

        if citations and rate < self._config.min_citation_resolution:
            issues.append(
                ValidationIssue.warning(
                    "citations_unresolved",
                    f"Only {len(resolved)} of {len(citations)} citations matched a reference",
                    field="citations",
                    remedy="Reference splitting is likely wrong for this layout",
                )
            )

        signals = [
            ConfidenceSignal(
                name="citation_resolution",
                value=rate,
                observation=f"{len(resolved)} of {len(citations)} in-text markers resolved",
            )
        ]
        if not citations:
            signals = [
                ConfidenceSignal(
                    name="citation_presence",
                    value=0.0,
                    observation="no in-text citation markers were found",
                )
            ]

        return ValidationResult.decide(
            validator=self.name,
            subject_id=subject.paper_id,
            subject_type=self.subject_type,
            confidence=Confidence.from_signals(signals),
            issues=issues,
        )


class MetadataValidator(Validator[PaperDocument]):
    """Does the document agree with what the indexes claimed about it?

    This is the clearest expression of zero trust in the codebase: discovery metadata is
    treated as an assertion by a third party, the PDF as an assertion by the document,
    and disagreement is reported rather than resolved by preferring one source. A
    mismatch usually means the wrong PDF was downloaded for the record.
    """

    name = "metadata_validator"
    subject_type = _SUBJECT_DOCUMENT

    def __init__(
        self, discovered: Paper | None = None, config: DocumentValidationConfig | None = None
    ) -> None:
        self._discovered = discovered
        self._config = config or DocumentValidationConfig()

    def check(self, subject: PaperDocument) -> ValidationResult:
        issues: list[ValidationIssue] = []
        evidence: list[Evidence] = []
        signals: list[ConfidenceSignal] = []
        metadata = subject.metadata

        if not metadata.title:
            issues.append(
                ValidationIssue.warning(
                    "document_title_missing",
                    "No title could be read from the document",
                    field="title",
                )
            )
        signals.append(
            ConfidenceSignal(
                name="self_metadata_completeness",
                value=_completeness(metadata),
                observation=(
                    f"title={bool(metadata.title)}, authors={len(metadata.authors)}, "
                    f"abstract={bool(metadata.abstract)}, doi={bool(metadata.doi)}"
                ),
            )
        )

        if self._discovered is not None:
            signals.append(self._title_agreement(subject, issues, evidence))
            signals.append(self._identifier_agreement(subject, issues))

        return ValidationResult.decide(
            validator=self.name,
            subject_id=subject.paper_id,
            subject_type=self.subject_type,
            confidence=Confidence.from_signals(signals),
            issues=issues,
            evidence=evidence,
        )

    def _title_agreement(
        self, subject: PaperDocument, issues: list[ValidationIssue], evidence: list[Evidence]
    ) -> ConfidenceSignal:
        assert self._discovered is not None  # noqa: S101 - guarded by the caller
        document_title = subject.metadata.title
        indexed_title = self._discovered.title

        if not document_title:
            return ConfidenceSignal(
                name="title_agreement",
                value=0.0,
                observation="document title unavailable, so no comparison was possible",
            )

        similarity = _title_similarity(document_title, indexed_title)
        if similarity < self._config.title_similarity_threshold:
            issues.append(
                ValidationIssue.warning(
                    "metadata_title_mismatch",
                    f"Document title {document_title!r} disagrees with the indexed title "
                    f"{indexed_title!r} (similarity {similarity:.2f})",
                    field="title",
                    remedy="The downloaded PDF may not be the paper that was discovered",
                )
            )
        else:
            evidence.append(
                Evidence(
                    kind=EvidenceKind.CROSS_SOURCE,
                    claim=f"Document and index agree on the title (similarity {similarity:.2f})",
                    location=SourceLocation(document_id=subject.paper_id, page=1),
                    produced_by=self.name,
                )
            )

        return ConfidenceSignal(
            name="title_agreement",
            value=similarity,
            observation=f"token overlap {similarity:.2f} between document and index titles",
        )

    def _identifier_agreement(
        self, subject: PaperDocument, issues: list[ValidationIssue]
    ) -> ConfidenceSignal:
        assert self._discovered is not None  # noqa: S101 - guarded by the caller
        document_doi = (subject.metadata.doi or "").lower()
        indexed_doi = (self._discovered.doi or "").lower()

        if not document_doi or not indexed_doi:
            return ConfidenceSignal(
                name="doi_agreement",
                value=0.5,
                observation="only one side reported a DOI, so agreement is unknown",
            )

        if document_doi != indexed_doi:
            issues.append(
                ValidationIssue.error(
                    "metadata_doi_mismatch",
                    f"Document DOI {document_doi} does not match indexed DOI {indexed_doi}",
                    field="doi",
                    remedy="Almost certainly the wrong PDF for this record; re-download",
                )
            )
            return ConfidenceSignal(
                name="doi_agreement",
                value=0.0,
                observation="DOIs differ between document and index",
            )

        return ConfidenceSignal(
            name="doi_agreement", value=1.0, observation=f"both report DOI {document_doi}"
        )


def _completeness(metadata: object) -> float:
    present = sum(
        bool(getattr(metadata, field)) for field in ("title", "authors", "abstract", "doi", "year")
    )
    return present / 5


def _title_similarity(left: str, right: str) -> float:
    """Token-overlap similarity; robust to the whitespace and casing noise PDFs add."""
    left_tokens = set(normalise_title(left).split())
    right_tokens = set(normalise_title(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens))
