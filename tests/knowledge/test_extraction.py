"""Extractors, validators and relation building."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from researchagent.core.evidence import Evidence, EvidenceKind, SourceLocation
from researchagent.core.prompts import PromptLibrary
from researchagent.models.knowledge import (
    DatasetDetails,
    KnowledgeKind,
    KnowledgeObject,
    MethodDetails,
    MetricDetails,
    PaperKnowledge,
    RelationPredicate,
    ResultDetails,
)
from researchagent.services.knowledge.extractors import (
    DatasetExtractor,
    LimitationExtractor,
    MethodExtractor,
    MetricExtractor,
    ResultExtractor,
)
from researchagent.services.knowledge.extractors.dataset import DatasetBatch, DatasetDraft
from researchagent.services.knowledge.extractors.limitation import LimitationDraft
from researchagent.services.knowledge.extractors.method import MethodBatch, MethodDraft
from researchagent.services.knowledge.extractors.metric import MetricDraft
from researchagent.services.knowledge.extractors.result import ResultBatch, ResultDraft
from researchagent.services.knowledge.registry import EXTRACTORS, build_extractors
from researchagent.services.knowledge.relations import RelationBuilder
from researchagent.services.llm_service import BoundLLM
from researchagent.services.validation.knowledge import (
    CompletenessValidator,
    EvidenceValidator,
    KnowledgeCoverageValidator,
    RelationshipValidator,
    ResultValidator,
)

QUOTE = (
    "We evaluate ReAct on the MIMIC-III dataset and report an accuracy of 94.3% on the "
    "held-out split."
)


def evidence(quote: str = QUOTE, *, page: int = 4, paragraph: int = 2) -> tuple[Evidence, ...]:
    return (
        Evidence.from_text(
            claim="test",
            quote=quote,
            location=SourceLocation(
                document_id="manual:01",
                page=page,
                section_id="s004",
                section_title="Results",
                paragraph_index=paragraph,
            ),
            produced_by="test",
        ),
    )


def an_object(
    kind: KnowledgeKind = KnowledgeKind.DATASET,
    *,
    name: str = "MIMIC-III",
    details: object | None = None,
    quote: str = QUOTE,
    description: str = "A de-identified clinical database.",
    paragraph: int = 2,
) -> KnowledgeObject:
    return KnowledgeObject.model_validate(
        {
            "id": f"manual:01#{kind.value}:{name}",
            "kind": kind,
            "paper_id": "manual:01",
            "name": name,
            "description": description,
            "details": details or DatasetDetails(),
            "evidence": evidence(quote, paragraph=paragraph),
            "extracted_by": "test",
        }
    )


class TestKnowledgeObjectInvariant:
    def test_a_knowledge_object_cannot_exist_without_evidence(self) -> None:
        """The central guarantee of v0.5, enforced by the model itself."""
        with pytest.raises(ValidationError, match="no evidence"):
            KnowledgeObject.model_validate(
                {
                    "id": "x",
                    "kind": KnowledgeKind.DATASET,
                    "paper_id": "manual:01",
                    "name": "Invented",
                    "details": DatasetDetails(),
                    "evidence": (),
                    "extracted_by": "test",
                }
            )

    def test_details_must_match_the_declared_kind(self) -> None:
        with pytest.raises(ValidationError):
            KnowledgeObject.model_validate(
                {
                    "id": "x",
                    "kind": KnowledgeKind.METHOD,
                    "paper_id": "manual:01",
                    "name": "ReAct",
                    "details": DatasetDetails(),
                    "evidence": evidence(),
                    "extracted_by": "test",
                }
            )

    def test_provenance_is_reachable_from_any_object(self) -> None:
        assert an_object().cite() == "manual:01 p.4 §Results ¶2"

    def test_objects_are_immutable(self) -> None:
        with pytest.raises(ValidationError):
            an_object().name = "changed"  # type: ignore[misc]


class TestExtractorMapping:
    def test_method_draft_becomes_a_method_object(self, prompt_library: PromptLibrary) -> None:
        extractor = MethodExtractor.__new__(MethodExtractor)

        result = extractor.to_object(
            MethodDraft(
                name="ReAct",
                description="Interleaved reasoning and acting.",
                category="agent architecture",
                components=["reasoning", "acting"],
                is_novel=False,
                quote=QUOTE,
            ),
            paper_id="manual:01",
            index=0,
            evidence=evidence(),
        )

        assert result is not None
        assert result.kind is KnowledgeKind.METHOD
        assert result.name == "ReAct"
        assert isinstance(result.details, MethodDetails)
        assert result.details.components == ("reasoning", "acting")

    def test_a_nameless_draft_is_dropped(self) -> None:
        extractor = DatasetExtractor.__new__(DatasetExtractor)

        assert (
            extractor.to_object(
                DatasetDraft(name="   ", quote=QUOTE),
                paper_id="manual:01",
                index=0,
                evidence=evidence(),
            )
            is None
        )

    def test_a_result_whose_number_is_absent_from_its_quote_is_dropped(self) -> None:
        """A real sentence with an invented figure inside it — the subtlest failure."""
        extractor = ResultExtractor.__new__(ResultExtractor)

        result = extractor.to_object(
            ResultDraft(metric_name="accuracy", value="99.9%", quote=QUOTE),
            paper_id="manual:01",
            index=0,
            evidence=evidence(),
        )

        assert result is None

    def test_a_result_whose_number_is_present_is_kept(self) -> None:
        extractor = ResultExtractor.__new__(ResultExtractor)

        result = extractor.to_object(
            ResultDraft(
                metric_name="accuracy", dataset_name="MIMIC-III", value="94.3%", quote=QUOTE
            ),
            paper_id="manual:01",
            index=0,
            evidence=evidence(),
        )

        assert result is not None
        assert isinstance(result.details, ResultDetails)
        assert result.details.numeric_value == 94.3
        assert result.details.dataset_name == "MIMIC-III"
        assert result.name == "accuracy on MIMIC-III"

    def test_a_result_without_a_metric_is_dropped(self) -> None:
        extractor = ResultExtractor.__new__(ResultExtractor)

        assert (
            extractor.to_object(
                ResultDraft(metric_name="", value="94.3%", quote=QUOTE),
                paper_id="manual:01",
                index=0,
                evidence=evidence(),
            )
            is None
        )

    def test_a_trivial_limitation_is_dropped(self) -> None:
        extractor = LimitationExtractor.__new__(LimitationExtractor)

        assert (
            extractor.to_object(
                LimitationDraft(summary="slow", quote=QUOTE),
                paper_id="manual:01",
                index=0,
                evidence=evidence(),
            )
            is None
        )

    def test_extractors_read_different_sections(self) -> None:
        """Feeding every extractor the whole paper invites cross-contamination."""
        assert MethodExtractor.source_sections != LimitationExtractor.source_sections
        assert MetricExtractor.kind is KnowledgeKind.METRIC
        assert MethodExtractor.kind is KnowledgeKind.METHOD


class TestRegistry:
    def test_all_six_extractors_are_registered(self) -> None:
        assert len(EXTRACTORS) == 6

    def test_config_names_become_instances(
        self, bound_llm: BoundLLM, prompt_library: PromptLibrary
    ) -> None:
        extractors = build_extractors(
            ("method_extractor", "result_extractor"), bound_llm, prompt_library
        )

        assert [e.name for e in extractors] == ["method_extractor", "result_extractor"]

    def test_every_extractor_has_a_loadable_prompt(
        self, bound_llm: BoundLLM, prompt_library: PromptLibrary
    ) -> None:
        for name, _ in EXTRACTORS:
            extractor = EXTRACTORS.get(name)(bound_llm, prompt_library)
            assert set(extractor.prompt.section_names()) == {"system", "extract"}


class TestValidators:
    def test_evidence_validator_flags_a_name_absent_from_its_quote(self) -> None:
        verdict = EvidenceValidator().validate(an_object(name="PhysioNet"))

        assert "name_absent_from_evidence" in verdict.issue_codes()

    def test_evidence_validator_passes_a_named_entity_present_in_its_quote(self) -> None:
        verdict = EvidenceValidator().validate(an_object(name="MIMIC-III"))

        assert verdict.success is True
        assert verdict.confidence.is_grounded is True

    def test_completeness_rejects_placeholder_names(self) -> None:
        verdict = CompletenessValidator().validate(an_object(name="dataset"))

        assert verdict.success is False
        assert "name_too_generic" in verdict.issue_codes()

    def test_result_validator_rejects_an_unsupported_value(self) -> None:
        unsupported = an_object(
            KnowledgeKind.RESULT,
            name="accuracy",
            details=ResultDetails(metric_name="accuracy", value_text="99.9%"),
        )

        verdict = ResultValidator().validate(unsupported)

        assert verdict.success is False
        assert "result_value_unsupported" in verdict.issue_codes()

    def test_result_validator_accepts_a_supported_value(self) -> None:
        supported = an_object(
            KnowledgeKind.RESULT,
            name="accuracy",
            details=ResultDetails(metric_name="accuracy", value_text="94.3%", unit="%"),
        )

        assert ResultValidator().validate(supported).success is True

    def test_result_validator_abstains_on_other_kinds(self) -> None:
        verdict = ResultValidator().validate(an_object(KnowledgeKind.DATASET))

        assert verdict.success is True
        assert verdict.confidence.is_grounded is False  # abstained, did not endorse

    def test_coverage_validator_reports_an_empty_extraction(self) -> None:
        verdict = KnowledgeCoverageValidator().validate(
            PaperKnowledge(paper_id="manual:01", document_sha256="abc")
        )

        assert verdict.success is False
        assert "no_knowledge_extracted" in verdict.issue_codes()

    def test_every_confidence_signal_carries_an_observation(self) -> None:
        verdict = EvidenceValidator().validate(an_object())

        assert all(signal.observation for signal in verdict.confidence.signals)


class TestRelationships:
    def test_a_result_links_to_its_metric_and_dataset(self) -> None:
        dataset = an_object(KnowledgeKind.DATASET, name="MIMIC-III")
        metric = an_object(KnowledgeKind.METRIC, name="accuracy", details=MetricDetails())
        result = an_object(
            KnowledgeKind.RESULT,
            name="accuracy on MIMIC-III",
            details=ResultDetails(
                metric_name="accuracy", dataset_name="MIMIC-III", value_text="94.3%"
            ),
        )

        relations = RelationBuilder().build((dataset, metric, result))

        predicates = {relation.predicate for relation in relations}
        assert RelationPredicate.MEASURED_BY in predicates
        assert RelationPredicate.REPORTED_ON in predicates

    def test_relations_carry_evidence(self) -> None:
        metric = an_object(KnowledgeKind.METRIC, name="accuracy", details=MetricDetails())
        result = an_object(
            KnowledgeKind.RESULT,
            name="accuracy",
            details=ResultDetails(metric_name="accuracy", value_text="94.3%"),
        )

        relation = RelationBuilder().build((metric, result))[0]

        assert relation.evidence
        assert relation.confidence.is_grounded is True

    def test_no_relations_are_invented_between_unrelated_objects(self) -> None:
        """Relations come from evidence, never from a model's sense of what fits.

        These two are supported by *different* paragraphs and neither names the other, so
        nothing in the paper connects them — and nothing should connect them here.
        """
        method = an_object(KnowledgeKind.METHOD, name="ReAct", details=MethodDetails(), paragraph=2)
        dataset = an_object(KnowledgeKind.DATASET, name="MIMIC-III", paragraph=9)

        assert RelationBuilder().build((method, dataset)) == ()

    def test_objects_sharing_a_paragraph_are_linked_by_it(self) -> None:
        """Co-occurrence in one paragraph is real, weaker evidence — and scored as such."""
        method = an_object(KnowledgeKind.METHOD, name="ReAct", details=MethodDetails(), paragraph=5)
        dataset = an_object(KnowledgeKind.DATASET, name="MIMIC-III", paragraph=5)

        relations = RelationBuilder().build((method, dataset))

        assert [r.predicate for r in relations] == [RelationPredicate.EVALUATED_ON]
        assert relations[0].confidence.score < 1.0

    def test_relationship_validator_rejects_dangling_edges(self) -> None:
        from researchagent.models.knowledge import KnowledgeRelation

        knowledge = PaperKnowledge(
            paper_id="manual:01",
            document_sha256="abc",
            objects=(an_object(),),
            relations=(
                KnowledgeRelation(
                    id="r1",
                    predicate=RelationPredicate.MEASURED_BY,
                    subject_id="missing",
                    object_id="also-missing",
                ),
            ),
        )

        verdict = RelationshipValidator().validate(knowledge)

        assert verdict.success is False
        assert "relation_dangling" in verdict.issue_codes()

    def test_relationship_validator_rejects_mistyped_edges(self) -> None:
        from researchagent.models.knowledge import KnowledgeRelation

        dataset = an_object(KnowledgeKind.DATASET, name="MIMIC-III")
        metric = an_object(KnowledgeKind.METRIC, name="accuracy", details=MetricDetails())
        knowledge = PaperKnowledge(
            paper_id="manual:01",
            document_sha256="abc",
            objects=(dataset, metric),
            relations=(
                KnowledgeRelation(
                    id="r1",
                    # MEASURED_BY expects a result subject, not a dataset.
                    predicate=RelationPredicate.MEASURED_BY,
                    subject_id=dataset.id,
                    object_id=metric.id,
                ),
            ),
        )

        verdict = RelationshipValidator().validate(knowledge)

        assert "relation_domain_mismatch" in verdict.issue_codes()

    def test_a_relation_cannot_point_at_itself(self) -> None:
        from researchagent.models.knowledge import KnowledgeRelation

        with pytest.raises(ValidationError):
            KnowledgeRelation(
                id="r1",
                predicate=RelationPredicate.MEASURED_BY,
                subject_id="same",
                object_id="same",
            )


class TestBatchSchemas:
    def test_batches_default_to_empty(self) -> None:
        """An empty list is a correct answer; the prompts say so explicitly."""
        assert MethodBatch().methods == []
        assert DatasetBatch().datasets == []
        assert ResultBatch().results == []

    def test_drafts_tolerate_missing_optional_fields(self) -> None:
        draft = MetricDraft(name="F1", quote=QUOTE)

        assert draft.unit == ""
        assert draft.higher_is_better is None


def test_evidence_kind_is_textual_for_grounded_quotes() -> None:
    assert evidence()[0].kind is EvidenceKind.EXTRACTED_TEXT


class TestDeduplication:
    def test_the_same_entity_from_two_sections_becomes_one_object(self) -> None:
        """Extractors read overlapping sections; double-counting would corrupt v0.7's graph."""
        from researchagent.services.knowledge.pipeline import _deduplicate

        first = an_object(name="MIMIC-III", paragraph=2)
        second = an_object(name="mimic-iii", paragraph=7)

        merged = _deduplicate([first, second])

        assert len(merged) == 1
        # The survivor inherits both citations: more provenance for one fact.
        assert len(merged[0].evidence) == 2

    def test_different_entities_are_kept_apart(self) -> None:
        from researchagent.services.knowledge.pipeline import _deduplicate

        assert len(_deduplicate([an_object(name="MIMIC-III"), an_object(name="MedQA")])) == 2

    def test_the_same_name_in_different_kinds_is_not_merged(self) -> None:
        """'accuracy' is legitimately both a metric and part of a result."""
        from researchagent.services.knowledge.pipeline import _deduplicate

        metric = an_object(KnowledgeKind.METRIC, name="accuracy", details=MetricDetails())
        result = an_object(
            KnowledgeKind.RESULT,
            name="accuracy",
            details=ResultDetails(metric_name="accuracy", value_text="94.3%"),
        )

        assert len(_deduplicate([metric, result])) == 2
