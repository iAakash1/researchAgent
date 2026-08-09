"""EvidenceBundle, retrieval layers, contradictions and the builder."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from researchagent.core.evidence import Evidence, SourceLocation
from researchagent.core.validation import Confidence, ConfidenceSignal, ValidationResult
from researchagent.models.bundle import (
    BundleCoverage,
    BundledEvidence,
    Contradiction,
    ContradictionKind,
    EvidenceBundle,
    EvidenceStance,
)
from researchagent.models.evidence import (
    EvidenceLink,
    EvidenceRecord,
    EvidenceRole,
    PaperEvidence,
    content_hash_for,
)
from researchagent.models.knowledge import (
    DatasetDetails,
    FutureWorkDetails,
    KnowledgeKind,
    KnowledgeObject,
    LimitationDetails,
    MethodDetails,
    MetricDetails,
    PaperKnowledge,
    ResultDetails,
)
from researchagent.models.query import QueryIntent, ResearchQuery
from researchagent.models.research import QuestionPriority, ResearchQuestion
from researchagent.repositories.evidence_repository import JsonEvidenceRepository
from researchagent.repositories.knowledge_repository import JsonKnowledgeRepository
from researchagent.schemas.knowledge import ValidatedKnowledge
from researchagent.services.evidence import (
    AgreementCrossPaperRetriever,
    ContradictionDetector,
    EvidenceBundleBuilder,
    EvidenceIndexer,
    LexicalKnowledgeRetriever,
    LinkedEvidenceRetriever,
)

QUOTE = "Metastable failures are triggered by sustained overload in distributed systems."


def evidence(quote: str = QUOTE, *, paper: str = "manual:01", paragraph: int = 2) -> Evidence:
    return Evidence.from_text(
        claim="test",
        quote=quote,
        location=SourceLocation(
            document_id=paper,
            page=4,
            section_id="s004",
            section_title="Results",
            paragraph_index=paragraph,
        ),
        produced_by="test",
    )


def an_object(
    kind: KnowledgeKind = KnowledgeKind.METHOD,
    *,
    name: str = "overload control",
    paper: str = "manual:01",
    details: object | None = None,
    quote: str = QUOTE,
    confidence: float = 0.8,
) -> KnowledgeObject:
    return KnowledgeObject.model_validate(
        {
            "id": f"{paper}#{kind.value}:{name}",
            "kind": kind,
            "paper_id": paper,
            "name": name,
            "description": "Described in the paper.",
            "details": details or _details_for(kind),
            "evidence": (evidence(quote, paper=paper),),
            "confidence": Confidence.from_signals(
                [ConfidenceSignal(name="test", value=confidence, observation="fixture confidence")]
            ),
            "extracted_by": "test",
        }
    )


def _details_for(kind: KnowledgeKind) -> object:
    """Details must match the object's kind — the model enforces it."""
    return {
        KnowledgeKind.METHOD: MethodDetails(),
        KnowledgeKind.DATASET: DatasetDetails(),
        KnowledgeKind.METRIC: MetricDetails(),
        KnowledgeKind.RESULT: ResultDetails(),
        KnowledgeKind.LIMITATION: LimitationDetails(),
        KnowledgeKind.FUTURE_WORK: FutureWorkDetails(),
    }[kind]


def a_verdict(success: bool = True) -> ValidationResult:
    return (
        ValidationResult.passed(
            validator="v",
            subject_id="b",
            subject_type="EvidenceBundle",
            confidence=Confidence.unknown(),
        )
        if success
        else ValidationResult.failed(
            validator="v",
            subject_id="b",
            subject_type="EvidenceBundle",
            issues=[],
        )
    )


class TestEvidenceBundle:
    def test_a_bundle_cannot_reference_absent_objects(self) -> None:
        """A bundle whose evidence points at a fact it does not carry is untraceable."""
        with pytest.raises(ValidationError, match="absent objects"):
            EvidenceBundle(
                id="b1",
                query=ResearchQuery(text="anything at all"),
                knowledge_objects=(),
                evidence=(
                    BundledEvidence(
                        evidence=evidence(),
                        paper_id="manual:01",
                        knowledge_object_id="ghost",
                    ),
                ),
                validation=a_verdict(),
                built_by="test",
            )

    def test_bundles_are_immutable(self) -> None:
        bundle = EvidenceBundle(
            id="b1",
            query=ResearchQuery(text="anything at all"),
            validation=a_verdict(),
            built_by="test",
        )

        with pytest.raises(ValidationError):
            bundle.id = "b2"  # type: ignore[misc]

    def test_citations_are_deduplicated_and_ordered(self) -> None:
        obj = an_object()
        bundle = EvidenceBundle(
            id="b1",
            query=ResearchQuery(text="overload control"),
            knowledge_objects=(obj,),
            evidence=(
                BundledEvidence(
                    evidence=evidence(), paper_id="manual:01", knowledge_object_id=obj.id
                ),
                BundledEvidence(
                    evidence=evidence(), paper_id="manual:01", knowledge_object_id=obj.id
                ),
            ),
            validation=a_verdict(),
            built_by="test",
        )

        assert bundle.citations() == ("manual:01 p.4 §Results ¶2",)

    def test_stance_partitions_the_evidence(self) -> None:
        obj = an_object()
        bundle = EvidenceBundle(
            id="b1",
            query=ResearchQuery(text="overload control"),
            knowledge_objects=(obj,),
            evidence=(
                BundledEvidence(
                    evidence=evidence(),
                    paper_id="manual:01",
                    knowledge_object_id=obj.id,
                    stance=EvidenceStance.SUPPORTING,
                ),
                BundledEvidence(
                    evidence=evidence(),
                    paper_id="manual:01",
                    knowledge_object_id=obj.id,
                    stance=EvidenceStance.CONTRADICTING,
                ),
            ),
            validation=a_verdict(),
            built_by="test",
        )

        assert len(bundle.by_stance(EvidenceStance.SUPPORTING)) == 1
        assert len(bundle.by_stance(EvidenceStance.CONTRADICTING)) == 1

    def test_validation_travels_inside_the_bundle(self) -> None:
        """A bundle leaves the system alone; its verdict must not be droppable."""
        bundle = EvidenceBundle(
            id="b1",
            query=ResearchQuery(text="anything at all"),
            validation=a_verdict(success=False),
            built_by="test",
        )

        assert bundle.is_trusted is False

    def test_coverage_reports_thinness(self) -> None:
        coverage = BundleCoverage(
            papers_represented=("manual:01",),
            objects_with_evidence=1,
            objects_without_evidence=3,
        )

        assert coverage.paper_count == 1
        assert coverage.evidence_completeness == 0.25


class TestResearchQuery:
    def test_a_planner_question_becomes_a_query(self) -> None:
        """The loop back to v0.2: questions become the retrieval requests."""
        question = ResearchQuestion(
            id="RQ1",
            question="What triggers metastable failures in distributed systems?",
            rationale="Triggers determine which mitigations apply at all.",
            priority=QuestionPriority.HIGH,
            keywords=["metastable", "overload"],
        )

        query = ResearchQuery.for_question(question, kinds=(KnowledgeKind.METHOD,))

        assert query.question_id == "RQ1"
        assert query.intent is QueryIntent.ANSWER
        assert query.terms == ("metastable", "overload")
        assert query.kinds == (KnowledgeKind.METHOD,)

    def test_empty_filters_mean_no_constraint(self) -> None:
        query = ResearchQuery(text="anything at all")

        assert query.matches_kind(KnowledgeKind.RESULT) is True
        assert query.matches_paper("manual:99") is True

    def test_blank_terms_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ResearchQuery(text="anything", terms=("  ",))


class TestEvidenceIndex:
    async def test_indexing_makes_evidence_independently_retrievable(
        self, evidence_repository: JsonEvidenceRepository
    ) -> None:
        """Knowledge is not the access path to evidence."""
        knowledge = PaperKnowledge(
            paper_id="manual:01",
            document_sha256="abc123",
            objects=(an_object(), an_object(kind=KnowledgeKind.DATASET, name="MIMIC-III")),
        )

        indexed = await EvidenceIndexer(evidence_repository).index(knowledge)

        assert len(indexed.records) >= 1
        stored = await evidence_repository.get_paper("manual:01")
        assert stored is not None
        found = await evidence_repository.get(indexed.records[0].id)
        assert found is not None

    async def test_one_sentence_supporting_two_facts_is_one_record_with_two_links(
        self, evidence_repository: JsonEvidenceRepository
    ) -> None:
        """The same observation twice is one piece of evidence, not two."""
        shared = QUOTE
        knowledge = PaperKnowledge(
            paper_id="manual:01",
            document_sha256="abc",
            objects=(
                an_object(name="overload control", quote=shared),
                an_object(kind=KnowledgeKind.DATASET, name="MIMIC-III", quote=shared),
            ),
        )

        indexed = await EvidenceIndexer(evidence_repository).index(knowledge)

        assert len(indexed.records) == 1
        assert len(indexed.records[0].links) == 2

    async def test_evidence_never_points_at_knowledge(self) -> None:
        """The association is the link record; evidence itself stays ignorant of it."""
        record = EvidenceRecord(evidence=evidence(), paper_id="manual:01", document_sha256="abc")

        assert not hasattr(record.evidence, "knowledge_object_id")
        assert record.knowledge_object_ids == ()

    async def test_links_are_added_without_mutating_the_record(self) -> None:
        record = EvidenceRecord(evidence=evidence(), paper_id="manual:01", document_sha256="abc")
        link = EvidenceLink(
            evidence_id=record.id,
            knowledge_object_id="obj-1",
            knowledge_kind=KnowledgeKind.METHOD,
            role=EvidenceRole.FOUNDING,
            linked_by="test",
        )

        updated = record.linked_to(link)

        assert record.links == ()
        assert updated.knowledge_object_ids == ("obj-1",)

    def test_content_hash_identifies_the_same_observation(self) -> None:
        left = evidence()
        right = evidence()

        assert content_hash_for("manual:01", left) == content_hash_for("manual:01", right)
        assert content_hash_for("manual:02", left) != content_hash_for("manual:01", right)

    async def test_search_finds_evidence_by_its_quoted_text(
        self, evidence_repository: JsonEvidenceRepository
    ) -> None:
        await evidence_repository.save_paper(
            PaperEvidence(
                paper_id="manual:01",
                document_sha256="abc",
                records=(
                    EvidenceRecord(
                        evidence=evidence(), paper_id="manual:01", document_sha256="abc"
                    ),
                ),
            )
        )

        found = await evidence_repository.search(("metastable", "overload"))

        assert len(found) == 1


class TestRetrievalLayers:
    async def _seed(self, repository: JsonKnowledgeRepository, *objects: KnowledgeObject) -> None:
        by_paper: dict[str, list[KnowledgeObject]] = {}
        for item in objects:
            by_paper.setdefault(item.paper_id, []).append(item)
        for paper_id, items in by_paper.items():
            await repository.save(
                ValidatedKnowledge(
                    value=PaperKnowledge(
                        paper_id=paper_id, document_sha256="abc", objects=tuple(items)
                    ),
                    validation=ValidationResult.passed(
                        validator="v",
                        subject_id=paper_id,
                        subject_type="PaperKnowledge",
                        confidence=Confidence.unknown(),
                    ),
                )
            )

    async def test_layer_one_finds_matching_facts(
        self, knowledge_repository: JsonKnowledgeRepository
    ) -> None:
        await self._seed(
            knowledge_repository,
            an_object(name="overload control"),
            an_object(name="pottery glazing", paper="manual:02"),
        )
        retriever = LexicalKnowledgeRetriever(knowledge_repository)

        result = await retriever.retrieve(ResearchQuery(text="overload control"))

        assert result.hits
        assert result.hits[0].item.name == "overload control"
        assert result.considered == 2

    async def test_hits_explain_their_score(
        self, knowledge_repository: JsonKnowledgeRepository
    ) -> None:
        await self._seed(knowledge_repository, an_object(name="overload control"))
        retriever = LexicalKnowledgeRetriever(knowledge_repository)

        result = await retriever.retrieve(ResearchQuery(text="overload control"))

        assert "name_match" in result.hits[0].explain()
        assert all(signal.observation for signal in result.hits[0].signals)

    async def test_untrusted_knowledge_is_not_retrievable(
        self, knowledge_repository: JsonKnowledgeRepository
    ) -> None:
        """Zero trust across stages: rejected knowledge stays rejected."""
        await knowledge_repository.save(
            ValidatedKnowledge(
                value=PaperKnowledge(
                    paper_id="manual:01",
                    document_sha256="abc",
                    objects=(an_object(name="overload control"),),
                ),
                validation=ValidationResult.failed(
                    validator="v",
                    subject_id="manual:01",
                    subject_type="PaperKnowledge",
                    issues=[],
                ),
            )
        )
        retriever = LexicalKnowledgeRetriever(knowledge_repository)

        result = await retriever.retrieve(ResearchQuery(text="overload control"))

        assert result.is_empty

    async def test_filters_are_applied(self, knowledge_repository: JsonKnowledgeRepository) -> None:
        await self._seed(
            knowledge_repository,
            an_object(kind=KnowledgeKind.METHOD, name="overload control"),
            an_object(kind=KnowledgeKind.DATASET, name="overload traces", details=DatasetDetails()),
        )
        retriever = LexicalKnowledgeRetriever(knowledge_repository)

        result = await retriever.retrieve(
            ResearchQuery(text="overload", kinds=(KnowledgeKind.DATASET,))
        )

        assert all(hit.item.kind is KnowledgeKind.DATASET for hit in result.hits)

    async def test_layer_four_rewards_cross_paper_agreement(
        self, knowledge_repository: JsonKnowledgeRepository
    ) -> None:
        """Two papers naming the same entity is a stronger finding than one naming it."""
        await self._seed(
            knowledge_repository,
            an_object(name="overload control", paper="manual:01"),
            an_object(name="overload control", paper="manual:02"),
            an_object(name="overload shedding", paper="manual:03"),
        )
        layer_one = LexicalKnowledgeRetriever(knowledge_repository)
        layer_four = AgreementCrossPaperRetriever(layer_one)

        result = await layer_four.retrieve(ResearchQuery(text="overload"))

        assert result.hits[0].item.name == "overload control"
        agreement = next(s for s in result.hits[0].signals if s.name == "cross_paper_agreement")
        assert "2 distinct paper(s)" in agreement.observation

    async def test_papers_mentioning_an_entity(
        self, knowledge_repository: JsonKnowledgeRepository
    ) -> None:
        await self._seed(
            knowledge_repository,
            an_object(name="overload control", paper="manual:01"),
            an_object(name="overload control", paper="manual:02"),
        )
        layer_four = AgreementCrossPaperRetriever(LexicalKnowledgeRetriever(knowledge_repository))

        assert await layer_four.papers_mentioning("overload control") == (
            "manual:01",
            "manual:02",
        )

    async def test_layer_two_walks_from_facts_to_their_support(
        self, evidence_repository: JsonEvidenceRepository
    ) -> None:
        obj = an_object()
        await EvidenceIndexer(evidence_repository).index(
            PaperKnowledge(paper_id="manual:01", document_sha256="abc", objects=(obj,))
        )
        retriever = LinkedEvidenceRetriever(evidence_repository)

        records = await retriever.for_objects((obj.id,))

        assert len(records) == 1
        assert records[0].location.page == 4


class TestContradictions:
    def test_conflicting_numbers_are_detected_and_both_sides_kept(self) -> None:
        left = an_object(
            KnowledgeKind.RESULT,
            name="accuracy",
            paper="manual:01",
            details=ResultDetails(
                metric_name="accuracy",
                dataset_name="MIMIC-III",
                value_text="94.3%",
                numeric_value=94.3,
            ),
        )
        right = an_object(
            KnowledgeKind.RESULT,
            name="accuracy",
            paper="manual:02",
            details=ResultDetails(
                metric_name="accuracy",
                dataset_name="MIMIC-III",
                value_text="71.0%",
                numeric_value=71.0,
            ),
        )

        found = ContradictionDetector().detect((left, right))

        assert len(found) == 1
        conflict = found[0]
        assert conflict.kind is ContradictionKind.VALUE_CONFLICT
        assert conflict.is_cross_paper is True
        assert conflict.left_evidence and conflict.right_evidence

    def test_close_numbers_are_not_a_contradiction(self) -> None:
        left = an_object(
            KnowledgeKind.RESULT,
            name="accuracy",
            paper="manual:01",
            details=ResultDetails(
                metric_name="accuracy", dataset_name="D", value_text="94.3", numeric_value=94.3
            ),
        )
        right = an_object(
            KnowledgeKind.RESULT,
            name="accuracy",
            paper="manual:02",
            details=ResultDetails(
                metric_name="accuracy", dataset_name="D", value_text="94.5", numeric_value=94.5
            ),
        )

        assert ContradictionDetector().detect((left, right)) == ()

    def test_different_metrics_never_conflict(self) -> None:
        left = an_object(
            KnowledgeKind.RESULT,
            name="accuracy",
            paper="manual:01",
            details=ResultDetails(
                metric_name="accuracy", dataset_name="D", value_text="94", numeric_value=94.0
            ),
        )
        right = an_object(
            KnowledgeKind.RESULT,
            name="latency",
            paper="manual:02",
            details=ResultDetails(
                metric_name="latency", dataset_name="D", value_text="12", numeric_value=12.0
            ),
        )

        assert ContradictionDetector().detect((left, right)) == ()

    def test_attribute_conflicts_are_cross_paper_only(self) -> None:
        """Disagreement inside one paper is usually an extraction slip, not a finding."""
        same_paper = (
            an_object(
                KnowledgeKind.DATASET,
                name="MIMIC-III",
                paper="manual:01",
                details=DatasetDetails(is_public=True),
            ),
            an_object(
                KnowledgeKind.DATASET,
                name="MIMIC-III",
                paper="manual:01",
                details=DatasetDetails(is_public=False),
            ),
        )
        cross_paper = (
            same_paper[0],
            an_object(
                KnowledgeKind.DATASET,
                name="MIMIC-III",
                paper="manual:02",
                details=DatasetDetails(is_public=False),
            ),
        )

        assert ContradictionDetector().detect(same_paper) == ()
        assert len(ContradictionDetector().detect(cross_paper)) == 1

    def test_a_contradiction_cannot_hold_one_object_twice(self) -> None:
        with pytest.raises(ValidationError):
            Contradiction(
                id="c1",
                kind=ContradictionKind.VALUE_CONFLICT,
                description="x",
                left_object_id="same",
                right_object_id="same",
                left_paper_id="manual:01",
                right_paper_id="manual:01",
                detected_by="test",
            )


class TestBundleBuilder:
    async def _builder(
        self,
        knowledge_repository: JsonKnowledgeRepository,
        evidence_repository: JsonEvidenceRepository,
        *objects: KnowledgeObject,
    ) -> EvidenceBundleBuilder:
        by_paper: dict[str, list[KnowledgeObject]] = {}
        for item in objects:
            by_paper.setdefault(item.paper_id, []).append(item)

        for paper_id, items in by_paper.items():
            knowledge = PaperKnowledge(
                paper_id=paper_id, document_sha256="abc", objects=tuple(items)
            )
            await knowledge_repository.save(
                ValidatedKnowledge(
                    value=knowledge,
                    validation=ValidationResult.passed(
                        validator="v",
                        subject_id=paper_id,
                        subject_type="PaperKnowledge",
                        confidence=Confidence.unknown(),
                    ),
                )
            )
            await EvidenceIndexer(evidence_repository).index(knowledge)

        layer_one = LexicalKnowledgeRetriever(knowledge_repository)
        return EvidenceBundleBuilder(
            layer_one,
            LinkedEvidenceRetriever(evidence_repository),
            AgreementCrossPaperRetriever(layer_one),
            ContradictionDetector(),
        )

    async def test_a_bundle_carries_facts_evidence_and_provenance(
        self,
        knowledge_repository: JsonKnowledgeRepository,
        evidence_repository: JsonEvidenceRepository,
    ) -> None:
        builder = await self._builder(
            knowledge_repository, evidence_repository, an_object(name="overload control")
        )

        bundle = await builder.build(ResearchQuery(text="overload control"))

        assert bundle.knowledge_objects
        assert bundle.evidence
        assert bundle.citations() == ("manual:01 p.4 §Results ¶2",)
        assert bundle.coverage.papers_represented == ("manual:01",)

    async def test_bundle_confidence_is_grounded(
        self,
        knowledge_repository: JsonKnowledgeRepository,
        evidence_repository: JsonEvidenceRepository,
    ) -> None:
        builder = await self._builder(
            knowledge_repository, evidence_repository, an_object(name="overload control")
        )

        bundle = await builder.build(ResearchQuery(text="overload control"))

        assert bundle.confidence.is_grounded is True
        names = {signal.name for signal in bundle.confidence.signals}
        assert {"object_confidence", "evidence_completeness", "corroboration"} <= names

    async def test_single_source_bundles_are_flagged_not_hidden(
        self,
        knowledge_repository: JsonKnowledgeRepository,
        evidence_repository: JsonEvidenceRepository,
    ) -> None:
        builder = await self._builder(
            knowledge_repository, evidence_repository, an_object(name="overload control")
        )

        bundle = await builder.build(ResearchQuery(text="overload control"))

        assert "single_source_coverage" in bundle.validation.issue_codes()

    async def test_an_empty_bundle_fails_validation(
        self,
        knowledge_repository: JsonKnowledgeRepository,
        evidence_repository: JsonEvidenceRepository,
    ) -> None:
        builder = await self._builder(knowledge_repository, evidence_repository)

        bundle = await builder.build(ResearchQuery(text="something nothing matches"))

        assert bundle.is_empty is True
        assert bundle.is_trusted is False
        assert "no_supporting_papers" in bundle.validation.issue_codes()

    async def test_bundle_ids_are_deterministic(
        self,
        knowledge_repository: JsonKnowledgeRepository,
        evidence_repository: JsonEvidenceRepository,
    ) -> None:
        """The same question over the same filters must not build a second bundle."""
        builder = await self._builder(
            knowledge_repository, evidence_repository, an_object(name="overload control")
        )
        query = ResearchQuery(text="overload control")

        first = await builder.build(query)
        second = await builder.build(query)

        assert first.id == second.id

    async def test_contradictions_reach_the_bundle(
        self,
        knowledge_repository: JsonKnowledgeRepository,
        evidence_repository: JsonEvidenceRepository,
    ) -> None:
        """Disagreement is carried into the context a reasoner will see."""
        builder = await self._builder(
            knowledge_repository,
            evidence_repository,
            an_object(
                KnowledgeKind.RESULT,
                name="accuracy",
                paper="manual:01",
                details=ResultDetails(
                    metric_name="accuracy",
                    dataset_name="MIMIC-III",
                    value_text="94.3%",
                    numeric_value=94.3,
                ),
            ),
            an_object(
                KnowledgeKind.RESULT,
                name="accuracy",
                paper="manual:02",
                details=ResultDetails(
                    metric_name="accuracy",
                    dataset_name="MIMIC-III",
                    value_text="71.0%",
                    numeric_value=71.0,
                ),
            ),
        )

        bundle = await builder.build(ResearchQuery(text="accuracy"))

        assert bundle.has_disagreement is True
        assert bundle.contradictions[0].is_cross_paper is True
