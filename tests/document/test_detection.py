"""Detection tests over synthetic layout — no PDF required.

This is the payoff of separating the loader from the detectors: every heading rule,
reference split and caption pattern is exercised on hand-built pages, so a failure points
at a rule rather than at a PDF.
"""

from __future__ import annotations

from researchagent.core.evidence import BoundingBox
from researchagent.models.document import SectionKind
from researchagent.models.layout import RawDocument, RawPage, TextBlock, TextStyle
from researchagent.services.document.figures import FigureTableDetector
from researchagent.services.document.metadata import MetadataExtractor
from researchagent.services.document.references import CitationExtractor, ReferenceExtractor
from researchagent.services.document.sections import SectionDetector, classify_section

BODY = 10.0
HEADING = 12.0
TITLE = 20.0


def block(
    text: str,
    *,
    page: int = 1,
    index: int = 0,
    size: float = BODY,
    bold: bool = False,
    top: float = 0.0,
) -> TextBlock:
    return TextBlock(
        text=text,
        page=page,
        index=index,
        bounding_box=BoundingBox(x0=0, y0=top, x1=400, y1=top + 20),
        style=TextStyle(size=size, font="Test", bold=bold),
    )


def document(*blocks: TextBlock, document_id: str = "doc-1") -> RawDocument:
    pages: dict[int, list[TextBlock]] = {}
    for item in blocks:
        pages.setdefault(item.page, []).append(item)
    return RawDocument(
        document_id=document_id,
        pages=tuple(
            RawPage(number=number, width=612, height=792, blocks=tuple(items))
            for number, items in sorted(pages.items())
        ),
        source_sha256="deadbeef",
        source_bytes=1024,
        loader="test",
    )


def paper_body(text: str, count: int = 1) -> list[TextBlock]:
    """Enough body text that the modal font size is unambiguous."""
    return [block(f"{text} {i}. " + "lorem ipsum dolor sit amet " * 6) for i in range(count)]


class TestSectionClassification:
    def test_canonical_names(self) -> None:
        assert classify_section("Abstract") is SectionKind.ABSTRACT
        assert classify_section("1 Introduction") is SectionKind.INTRODUCTION
        assert classify_section("References") is SectionKind.REFERENCES

    def test_variants_reach_the_same_kind(self) -> None:
        """Authors name these a dozen ways; downstream code must not have to care."""
        for title in ("5 Evaluation", "Evaluation and Analysis", "IV. Evaluation"):
            assert classify_section(title) is SectionKind.EVALUATION

        for title in ("Related Work", "2 Prior Work", "Literature Review"):
            assert classify_section(title) is SectionKind.RELATED_WORK

    def test_specific_patterns_beat_general_ones(self) -> None:
        # "Related Work" contains "work", and "Future Work" must not swallow it.
        assert classify_section("Related Work") is SectionKind.RELATED_WORK
        assert classify_section("7 Future Work") is SectionKind.FUTURE_WORK

    def test_unknown_headings_are_other_not_guessed(self) -> None:
        assert classify_section("Threats and Opportunities in Widget Design") is SectionKind.OTHER

    def test_numbering_is_stripped_before_matching(self) -> None:
        assert classify_section("3.2.1 Methodology") is SectionKind.METHODOLOGY
        assert classify_section("A.1 Appendix") is SectionKind.APPENDIX


class TestSectionDetection:
    def test_headings_split_the_document(self) -> None:
        raw = document(
            block("Abstract", size=HEADING, bold=True),
            *paper_body("abstract text", 2),
            block("1 Introduction", size=HEADING, bold=True),
            *paper_body("intro text", 3),
            block("2 Method", size=HEADING, bold=True),
            *paper_body("method text", 2),
        )

        sections = SectionDetector().detect(raw)

        assert [s.kind for s in sections] == [
            SectionKind.ABSTRACT,
            SectionKind.INTRODUCTION,
            SectionKind.METHODOLOGY,
        ]
        assert [len(s.paragraphs) for s in sections] == [2, 3, 2]

    def test_body_text_is_not_mistaken_for_headings(self) -> None:
        """The gate that matters: brevity alone must never create a section.

        Without a typographic requirement, every short line — table cells, author names,
        page furniture — becomes a heading and the section count explodes.
        """
        raw = document(
            block("1 Introduction", size=HEADING, bold=True),
            block("Short line"),
            block("Another short one"),
            block("Yet more brief text"),
            *paper_body("body", 2),
        )

        sections = SectionDetector().detect(raw)

        assert len(sections) == 1
        assert len(sections[0].paragraphs) == 5

    def test_larger_font_alone_is_enough(self) -> None:
        raw = document(
            *paper_body("body", 3),
            block("Results", size=HEADING),
            *paper_body("results", 2),
        )

        assert any(s.kind is SectionKind.RESULTS for s in SectionDetector().detect(raw))

    def test_numbering_alone_is_enough(self) -> None:
        raw = document(
            *paper_body("body", 3),
            block("4 Discussion"),
            *paper_body("discussion", 2),
        )

        assert any(s.kind is SectionKind.DISCUSSION for s in SectionDetector().detect(raw))

    def test_subsection_hierarchy_from_numbering(self) -> None:
        raw = document(
            block("2 Method", size=HEADING, bold=True),
            *paper_body("method", 2),
            block("2.1 Setup", size=HEADING, bold=True),
            *paper_body("setup", 2),
            block("2.2 Model", size=HEADING, bold=True),
            *paper_body("model", 2),
        )

        sections = SectionDetector().detect(raw)

        assert [s.level for s in sections] == [1, 2, 2]
        assert sections[1].parent_id == sections[0].id
        assert sections[2].parent_id == sections[0].id

    def test_page_ranges_are_recorded(self) -> None:
        raw = document(
            block("1 Introduction", page=1, size=HEADING, bold=True),
            block("intro " * 20, page=1),
            block("continues " * 20, page=2),
        )

        section = SectionDetector().detect(raw)[0]

        assert (section.page_start, section.page_end) == (1, 2)

    def test_a_document_with_no_headings_still_yields_addressable_text(self) -> None:
        """Better one OTHER section than nothing; the validator reports the problem."""
        raw = document(*paper_body("plain", 4))

        sections = SectionDetector().detect(raw)

        assert len(sections) == 1
        assert sections[0].kind is SectionKind.OTHER
        assert len(sections[0].paragraphs) == 4

    def test_empty_document(self) -> None:
        assert SectionDetector().detect(document()) == ()

    def test_canonical_names_raise_detection_confidence(self) -> None:
        raw = document(
            block("Conclusion", size=HEADING),
            *paper_body("c", 2),
            block("Widget Taxonomy", size=HEADING),
            *paper_body("w", 2),
        )

        sections = {s.kind: s for s in SectionDetector().detect(raw)}

        assert (
            sections[SectionKind.CONCLUSION].detection_confidence
            > sections[SectionKind.OTHER].detection_confidence
        )


class TestReferenceExtraction:
    def _sections(self, references_text: str) -> tuple:
        raw = document(
            block("1 Introduction", size=HEADING, bold=True),
            block("We build on prior work [1] and later results [2, 3]. " + "text " * 20),
            block("References", size=HEADING, bold=True),
            block(references_text),
        )
        return SectionDetector().detect(raw)

    def test_entries_split_on_inline_markers(self) -> None:
        """Bibliographies often arrive as one block with markers mid-line.

        Splitting only on line starts merges every entry into the first, which then makes
        every in-text citation unresolvable.
        """
        sections = self._sections(
            "[1] Joe Armstrong. Making reliable distributed systems. PhD thesis, 2003. "
            "[2] E. A. Brewer. Towards robust systems. PODC, 2000. "
            "[3] Betsy Beyer. Site Reliability Engineering. O'Reilly, 2016."
        )

        references = ReferenceExtractor().extract(sections)

        assert [r.marker for r in references] == ["1", "2", "3"]
        assert "Armstrong" in references[0].raw
        assert "Brewer" in references[1].raw

    def test_identifiers_are_recovered(self) -> None:
        sections = self._sections(
            "[1] A. Author. A paper title here. Proc. ACM, 2021. doi:10.1145/3458336.3465286 "
            "[2] B. Author. Another paper. arXiv:2401.12345, 2024."
        )

        references = ReferenceExtractor().extract(sections)

        assert references[0].doi == "10.1145/3458336.3465286"
        assert references[0].year == 2021
        assert references[1].arxiv_id == "2401.12345"

    def test_raw_text_is_always_preserved(self) -> None:
        """Parsing is lossy; v0.7 re-resolves from `raw` without re-reading the PDF."""
        sections = self._sections("[1] Something entirely unparseable %%% ###### 12345 xyz")

        references = ReferenceExtractor().extract(sections)

        assert references[0].raw.startswith("Something entirely unparseable")

    def test_no_references_section_yields_nothing(self) -> None:
        raw = document(block("1 Introduction", size=HEADING, bold=True), *paper_body("x", 2))

        assert ReferenceExtractor().extract(SectionDetector().detect(raw)) == ()


class TestCitationExtraction:
    def test_markers_link_to_references(self) -> None:
        raw = document(
            block("1 Introduction", size=HEADING, bold=True),
            block("Prior work [1] showed this, and [2] extended it. " + "filler " * 20),
            block("References", size=HEADING, bold=True),
            block("[1] A. Author. First paper. 2020. [2] B. Author. Second paper. 2021."),
        )
        sections = SectionDetector().detect(raw)
        references = ReferenceExtractor().extract(sections)

        citations = CitationExtractor().extract(sections, references)

        assert [c.marker for c in citations] == ["[1]", "[2]"]
        assert all(citation.is_resolved for citation in citations)
        assert citations[0].reference_id == references[0].id

    def test_grouped_markers_expand(self) -> None:
        raw = document(
            block("1 Introduction", size=HEADING, bold=True),
            block("Several works [1, 2] agree. " + "filler " * 20),
            block("References", size=HEADING, bold=True),
            block("[1] A. First paper. 2020. [2] B. Second paper. 2021."),
        )
        sections = SectionDetector().detect(raw)
        references = ReferenceExtractor().extract(sections)

        citations = CitationExtractor().extract(sections, references)

        assert [c.marker for c in citations] == ["[1]", "[2]"]

    def test_unmatched_markers_are_recorded_as_unresolved(self) -> None:
        """Dropping them would hide exactly the signal that measures parse quality."""
        raw = document(
            block("1 Introduction", size=HEADING, bold=True),
            block("An orphan citation [99] appears here. " + "filler " * 20),
            block("References", size=HEADING, bold=True),
            block("[1] A. Author. Only paper. 2020."),
        )
        sections = SectionDetector().detect(raw)
        references = ReferenceExtractor().extract(sections)

        citations = CitationExtractor().extract(sections, references)

        assert len(citations) == 1
        assert citations[0].is_resolved is False

    def test_bibliography_entries_are_not_counted_as_citations(self) -> None:
        raw = document(
            block("References", size=HEADING, bold=True),
            block("[1] A. Author. Only paper. 2020."),
        )
        sections = SectionDetector().detect(raw)
        references = ReferenceExtractor().extract(sections)

        assert CitationExtractor().extract(sections, references) == ()


class TestFigureTableDetection:
    def test_captions_are_detected(self) -> None:
        raw = document(
            block("Figure 1: System architecture overview."),
            block("Table 2. Latency measurements in milliseconds."),
            *paper_body("body", 2),
        )

        figures, tables = FigureTableDetector().detect(raw)

        assert figures[0].label == "Figure 1"
        assert figures[0].caption == "System architecture overview."
        assert tables[0].label == "Table 2"
        assert tables[0].page == 1

    def test_abbreviated_and_roman_labels(self) -> None:
        raw = document(block("Fig. 3 Throughput over time."), block("Table IV. Parameters."))

        figures, tables = FigureTableDetector().detect(raw)

        assert figures[0].label == "Figure 3"
        assert tables[0].label == "Table IV"

    def test_prose_beginning_with_table_is_not_a_caption(self) -> None:
        raw = document(block("Table lookups dominate the cost of this operation in practice."))

        figures, tables = FigureTableDetector().detect(raw)

        assert (figures, tables) == ((), ())

    def test_bounding_box_is_kept_for_later_extraction(self) -> None:
        raw = document(block("Figure 1: Overview.", top=100.0))

        figures, _ = FigureTableDetector().detect(raw)

        assert figures[0].bounding_box is not None
        assert figures[0].bounding_box.y0 == 100.0


class TestMetadataExtraction:
    def test_title_authors_and_abstract_from_the_first_page(self) -> None:
        raw = document(
            block("Metastable Failures in Distributed Systems", size=TITLE, top=80),
            block("Nathan Bronson, Abutalib Aghayev", top=110),
            block("Abstract", size=HEADING, bold=True, top=140),
            block(
                "We describe metastable failures, a failure pattern in distributed systems "
                "that manifests as black swan events.",
                top=160,
            ),
            *paper_body("body", 3),
        )

        metadata = MetadataExtractor().extract(raw)

        assert metadata.title == "Metastable Failures in Distributed Systems"
        assert "Nathan Bronson" in metadata.authors
        assert metadata.abstract is not None
        assert metadata.abstract.startswith("We describe metastable failures")

    def test_identifiers_are_read_from_the_front_page(self) -> None:
        raw = document(
            block("A Paper Title", size=TITLE),
            block("doi:10.1145/3458336.3465286 arXiv:2401.12345"),
            *paper_body("body", 2),
        )

        metadata = MetadataExtractor().extract(raw)

        assert metadata.doi == "10.1145/3458336.3465286"
        assert metadata.arxiv_id == "2401.12345"

    def test_inline_abstract_heading(self) -> None:
        raw = document(
            block("A Paper Title", size=TITLE),
            block(
                "Abstract: This paper studies metastable failures in large systems "
                "and how to avoid them."
            ),
            *paper_body("body", 2),
        )

        metadata = MetadataExtractor().extract(raw)

        assert metadata.abstract is not None
        assert metadata.abstract.startswith("This paper studies")

    def test_missing_metadata_stays_empty(self) -> None:
        metadata = MetadataExtractor().extract(document(*paper_body("body", 3)))

        assert metadata.doi is None
        assert metadata.abstract is None

    def test_empty_document(self) -> None:
        assert MetadataExtractor().extract(document()).title is None
