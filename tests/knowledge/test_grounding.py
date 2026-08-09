"""Evidence grounding — the anti-hallucination gate.

These are the most important tests in v0.5. If grounding is wrong, an invented quote
becomes a fact with a page number attached, which is worse than having no provenance at
all.
"""

from __future__ import annotations

from researchagent.core.evidence import BoundingBox
from researchagent.models.document import (
    DocumentProvenance,
    DocumentStatistics,
    PaperDocument,
    Paragraph,
    Section,
    SectionKind,
)
from researchagent.services.knowledge.grounding import EvidenceGrounder, normalise

REAL_SENTENCE = (
    "We evaluate our approach on the MIMIC-III dataset and report an accuracy of 94.3% "
    "which outperforms the strongest baseline."
)
SECOND_SENTENCE = (
    "Metastable failures are triggered by a sustained overload that persists after the "
    "original trigger has been removed."
)


def document(*paragraph_texts: str) -> PaperDocument:
    paragraphs = tuple(
        Paragraph(
            index=index,
            text=text,
            page=index + 1,
            bounding_box=BoundingBox(x0=0, y0=0, x1=100, y1=20),
        )
        for index, text in enumerate(paragraph_texts)
    )
    section = Section(
        id="s001",
        kind=SectionKind.RESULTS,
        title="4 Results",
        level=1,
        order=0,
        paragraphs=paragraphs,
        page_start=1,
        page_end=max(len(paragraph_texts), 1),
    )
    return PaperDocument(
        paper_id="manual:01",
        provenance=DocumentProvenance(
            source_path="/tmp/x.pdf",  # noqa: S108 - fixture path, never opened
            source_sha256="abc123",
            source_bytes=10,
            loader="test",
        ),
        sections=(section,),
        statistics=DocumentStatistics(
            pages=1,
            characters=sum(len(t) for t in paragraph_texts),
            words=0,
            sections=1,
            paragraphs=len(paragraphs),
            figures=0,
            tables=0,
            references=0,
            citations=0,
            resolved_citations=0,
            empty_pages=0,
        ),
    )


class TestGrounding:
    def test_an_exact_quote_is_located(self) -> None:
        grounder = EvidenceGrounder(document(REAL_SENTENCE))

        grounded = grounder.ground(REAL_SENTENCE)

        assert grounded is not None
        assert grounded.exact is True
        assert grounded.similarity == 1.0
        assert grounded.location.page == 1
        assert grounded.location.section_title == "4 Results"
        assert grounded.location.paragraph_index == 0

    def test_a_fabricated_quote_is_rejected(self) -> None:
        """The failure the entire release exists to prevent."""
        grounder = EvidenceGrounder(document(REAL_SENTENCE))

        invented = grounder.ground(
            "We evaluate our approach on the PhysioNet corpus and report an F1 of 88.1%."
        )

        assert invented is None

    def test_no_evidence_means_no_knowledge_can_be_built(self) -> None:
        grounder = EvidenceGrounder(document(REAL_SENTENCE))

        evidence = grounder.evidence_for(
            claim="dataset",
            quote="A sentence that is nowhere in this paper at all.",
            produced_by="test",
        )

        assert evidence is None

    def test_a_grounded_quote_becomes_addressable_evidence(self) -> None:
        grounder = EvidenceGrounder(document("filler paragraph one", REAL_SENTENCE))

        evidence = grounder.evidence_for(
            claim="accuracy of 94.3%", quote=REAL_SENTENCE, produced_by="result_extractor"
        )

        assert evidence is not None
        assert evidence.quote == REAL_SENTENCE
        assert evidence.location.page == 2
        assert evidence.location.paragraph_index == 1
        assert evidence.produced_by == "result_extractor"
        assert "manual:01 p.2" in evidence.location.describe()

    def test_quotes_are_found_inside_longer_paragraphs(self) -> None:
        paragraph = f"Some preceding discussion. {REAL_SENTENCE} And some trailing text."
        grounder = EvidenceGrounder(document(paragraph))

        grounded = grounder.ground(REAL_SENTENCE)

        assert grounded is not None
        assert grounded.exact is True

    def test_pdf_noise_does_not_break_grounding(self) -> None:
        """Ligatures, soft hyphens and column wrapping must not reject a real quote."""
        printed = "The classiﬁer achieves signiﬁcant improvements on distri-\nbuted workloads."
        grounder = EvidenceGrounder(document(printed))

        grounded = grounder.ground(
            "The classifier achieves significant improvements on distributed workloads."
        )

        assert grounded is not None

    def test_near_misses_are_accepted_within_the_threshold(self) -> None:
        grounder = EvidenceGrounder(document(REAL_SENTENCE), similarity_threshold=0.85)

        # One word differs; still the same sentence in the paper.
        grounded = grounder.ground(
            "We evaluate our approach on the MIMIC-III dataset and report an accuracy of "
            "94.3% which outperforms the strongest baselines."
        )

        assert grounded is not None
        assert grounded.exact is False
        assert grounded.similarity >= 0.85

    def test_a_high_threshold_rejects_paraphrase(self) -> None:
        """Paraphrase is not a quote. The floor is what separates the two."""
        grounder = EvidenceGrounder(document(REAL_SENTENCE), similarity_threshold=0.95)

        paraphrase = grounder.ground(
            "Our method reaches 94.3 percent accuracy on MIMIC-III, beating all baselines."
        )

        assert paraphrase is None

    def test_fragments_are_not_grounded(self) -> None:
        """'accuracy' appears in every paper and proves nothing."""
        grounder = EvidenceGrounder(document(REAL_SENTENCE))

        assert grounder.ground("accuracy") is None
        assert grounder.ground("94.3%") is None

    def test_the_right_paragraph_is_chosen(self) -> None:
        grounder = EvidenceGrounder(document(SECOND_SENTENCE, "unrelated", REAL_SENTENCE))

        grounded = grounder.ground(SECOND_SENTENCE)

        assert grounded is not None
        assert grounded.location.paragraph_index == 0

    def test_empty_document_grounds_nothing(self) -> None:
        assert EvidenceGrounder(document()).ground(REAL_SENTENCE) is None


class TestNormalisation:
    def test_collapses_pdf_artefacts_without_changing_words(self) -> None:
        assert normalise("The  classiﬁer\nworks") == "the classifier works"
        assert normalise("distri-\nbuted systems") == "distributed systems"
        assert normalise("“quoted”") == '"quoted"'

    def test_is_case_insensitive(self) -> None:
        assert normalise("MIMIC-III") == normalise("mimic-iii")
