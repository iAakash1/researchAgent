"""Integrity registry: Crossref parsing, classification, failure handling, caching.

Every HTTP response is a recorded Crossref fixture (``tests/fixtures/crossref/``) or a
synthetic payload under Crossref's test prefix 10.5555. No test touches the network.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from researchagent.config.loader import ConfigLoader
from researchagent.config.schemas import IntegrityConfig, PdfNoticeSettings
from researchagent.core.interfaces.integrity import LookupOutcome, RegistryLookup
from researchagent.integrations.crossref import CrossrefIntegrityRegistry
from researchagent.integrations.http import HttpClient
from researchagent.models.integrity import (
    IntegrityChange,
    IntegrityObservation,
    IntegrityStatus,
    ObservationBasis,
    UpdateNotice,
)
from researchagent.repositories.integrity_repository import JsonIntegrityRepository
from researchagent.services.integrity import (
    IntegrityService,
    RetractionNoticeScanner,
    UpdateClassifier,
    identifiers_from_paper_id,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "crossref"
WORKS_URL = "https://api.crossref.org/works"


def fixture(name: str) -> object:
    return json.loads((FIXTURES / name).read_text())


def registry_for(
    routes: dict[str, object | int], *, query_update_notices: bool = False
) -> tuple[CrossrefIntegrityRegistry, list[httpx.Request]]:
    """Route by the request path (+ filter). An int value is an HTTP status to return."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        key = request.url.path + (
            f"?{request.url.params['filter']}" if "filter" in request.url.params else ""
        )
        found = routes.get(key)
        if found is None:
            return httpx.Response(404, json={"status": "error"})
        if isinstance(found, int):
            return httpx.Response(found, json={"status": "error"})
        return httpx.Response(200, json=found)

    client = HttpClient("crossref", requests_per_second=0, transport=httpx.MockTransport(handler))
    return (
        CrossrefIntegrityRegistry(
            client, works_url=WORKS_URL, query_update_notices=query_update_notices
        ),
        seen,
    )


@pytest.fixture
def integrity_config(config_loader: ConfigLoader) -> IntegrityConfig:
    return config_loader.load("integrity", IntegrityConfig)


@pytest.fixture
def classifier(integrity_config: IntegrityConfig) -> UpdateClassifier:
    return UpdateClassifier(integrity_config)


async def classify(
    classifier: UpdateClassifier, doi: str, routes: dict[str, object | int]
) -> IntegrityObservation:
    registry, _ = registry_for(routes)
    lookup = await registry.lookup(doi)
    return classifier.classify(lookup, paper_id=f"doi:{doi}")


# --- parsing per status type --------------------------------------------------------------


async def test_retraction_from_retraction_watch(classifier: UpdateClassifier) -> None:
    doi = "10.1021/am300292v"
    observation = await classify(
        classifier, doi, {f"/works/{doi}": fixture("work_retracted_retraction_watch.json")}
    )

    assert observation.status is IntegrityStatus.RETRACTED
    assert observation.basis is ObservationBasis.REGISTRY_RECORD
    assert observation.registry == "crossref"
    assert observation.raw_update_type == "retraction"
    assert observation.source == "retraction-watch"
    assert observation.notice_doi == "10.1021/acsami.9b11759"
    assert observation.notice_date == "2019-07-26"
    assert observation.notices[0].record_id == "20946"
    assert observation.doi == doi


async def test_same_retraction_from_two_sources_is_kept_per_source(
    classifier: UpdateClassifier,
) -> None:
    doi = "10.1177/1758835920922055"
    observation = await classify(
        classifier, doi, {f"/works/{doi}": fixture("work_retracted_two_sources.json")}
    )

    assert observation.status is IntegrityStatus.RETRACTED
    assert {notice.source for notice in observation.notices} == {"retraction-watch", "publisher"}
    assert len(observation.notices) == 2


async def test_expression_of_concern_is_not_a_retraction(classifier: UpdateClassifier) -> None:
    doi = "10.1016/j.lungcan.2003.10.006"
    observation = await classify(
        classifier, doi, {f"/works/{doi}": fixture("work_expression_of_concern.json")}
    )

    assert observation.status is IntegrityStatus.EXPRESSION_OF_CONCERN
    assert observation.status is not IntegrityStatus.RETRACTED
    # Two dated entries for the same notice: the most recent one is decisive.
    assert observation.notice_date == "2026-05-01"


async def test_correction_is_not_a_retraction(classifier: UpdateClassifier) -> None:
    doi = "10.1088/1361-6595/aaebdb"
    observation = await classify(classifier, doi, {f"/works/{doi}": fixture("work_corrected.json")})

    assert observation.status is IntegrityStatus.CORRECTED
    assert observation.notice_doi == "10.1088/1361-6595/ab245f"


async def test_withdrawal_whose_notice_is_the_work_itself(classifier: UpdateClassifier) -> None:
    doi = "10.1016/j.adengl.2021.11.027"
    observation = await classify(
        classifier, doi, {f"/works/{doi}": fixture("work_withdrawn_self_notice.json")}
    )

    assert observation.status is IntegrityStatus.WITHDRAWN
    assert observation.notice_doi == doi
    # Read from both updated-by and the self-referencing update-to, deduplicated.
    assert len(observation.notices) == 1


async def test_record_without_updates_is_current(classifier: UpdateClassifier) -> None:
    doi = "10.1038/nature14539"
    observation = await classify(classifier, doi, {f"/works/{doi}": fixture("work_current.json")})

    assert observation.status is IntegrityStatus.CURRENT
    assert observation.basis is ObservationBasis.REGISTRY_RECORD


async def test_notice_record_is_not_itself_retracted(classifier: UpdateClassifier) -> None:
    # A retraction notice's `update-to` names the *retracted* work, not the notice.
    doi = "10.1177/17588359231172420"
    observation = await classify(
        classifier, doi, {f"/works/{doi}": fixture("notice_retraction.json")}
    )

    assert observation.status is IntegrityStatus.CURRENT


async def test_updates_filter_cross_check(classifier: UpdateClassifier) -> None:
    doi = "10.1177/1758835920922055"
    registry, seen = registry_for(
        {
            f"/works/{doi}": fixture("work_current.json"),
            f"/works?updates:{doi}": fixture("updates_retracted_two_sources.json"),
        },
        query_update_notices=True,
    )
    lookup = await registry.lookup(doi)
    observation = classifier.classify(lookup, paper_id=f"doi:{doi}")

    assert len(seen) == 2
    assert observation.status is IntegrityStatus.RETRACTED
    assert observation.notice_doi == "10.1177/17588359231172420"


async def test_most_severe_update_wins(classifier: UpdateClassifier) -> None:
    lookup = RegistryLookup(
        registry="crossref",
        doi="10.5555/racm-test-severity",
        outcome=LookupOutcome.FOUND,
        detail="synthetic",
        notices=(
            UpdateNotice(update_type="correction", relation="updated-by", notice_date="2020-01-01"),
            UpdateNotice(update_type="Retraction", relation="updated-by", notice_date="2021-01-01"),
            UpdateNotice(
                update_type="expression_of_concern", relation="updated-by", notice_date="2020-06-01"
            ),
        ),
    )
    observation = classifier.classify(lookup, paper_id="doi:10.5555/racm-test-severity")

    assert observation.status is IntegrityStatus.RETRACTED


async def test_unmapped_update_type_is_unknown_and_flagged(classifier: UpdateClassifier) -> None:
    lookup = RegistryLookup(
        registry="crossref",
        doi="10.5555/racm-test-partial",
        outcome=LookupOutcome.FOUND,
        detail="synthetic",
        notices=(UpdateNotice(update_type="partial_retraction", relation="updated-by"),),
    )
    observation = classifier.classify(lookup, paper_id="doi:10.5555/racm-test-partial")

    assert observation.status is IntegrityStatus.UNKNOWN
    assert observation.basis is ObservationBasis.UNMAPPED_UPDATE
    assert observation.flags_review


async def test_ignored_update_type_leaves_record_current(classifier: UpdateClassifier) -> None:
    lookup = RegistryLookup(
        registry="crossref",
        doi="10.5555/racm-test-version",
        outcome=LookupOutcome.FOUND,
        detail="synthetic",
        notices=(UpdateNotice(update_type="new_version", relation="updated-by"),),
    )
    assert (
        classifier.classify(lookup, paper_id="doi:10.5555/racm-test-version").status
        is IntegrityStatus.CURRENT
    )


# --- failures are unknown, never current --------------------------------------------------


@pytest.mark.parametrize(
    ("routes", "outcome"),
    [
        ({}, LookupOutcome.NOT_FOUND),
        ({"/works/10.5555/racm-test": 503}, LookupOutcome.FAILED),
        ({"/works/10.5555/racm-test": 429}, LookupOutcome.FAILED),
        ({"/works/10.5555/racm-test": 400}, LookupOutcome.FAILED),
        ({"/works/10.5555/racm-test": {"status": "ok", "message": "nope"}}, LookupOutcome.FAILED),
    ],
)
async def test_failed_lookup_is_unknown_never_current(
    classifier: UpdateClassifier, routes: dict[str, object | int], outcome: LookupOutcome
) -> None:
    registry, _ = registry_for(routes)
    lookup = await registry.lookup("10.5555/racm-test")
    observation = classifier.classify(lookup, paper_id="doi:10.5555/racm-test")

    assert lookup.outcome is outcome
    assert observation.status is IntegrityStatus.UNKNOWN
    assert observation.basis is ObservationBasis.LOOKUP_FAILED
    assert not observation.is_informative


async def test_network_error_is_unknown(classifier: UpdateClassifier) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable", request=request)

    client = HttpClient("crossref", requests_per_second=0, transport=httpx.MockTransport(boom))
    registry = CrossrefIntegrityRegistry(client, works_url=WORKS_URL)
    observation = classifier.classify(
        await registry.lookup("10.5555/racm-test"), paper_id="doi:10.5555/racm-test"
    )

    assert observation.status is IntegrityStatus.UNKNOWN


async def test_cross_check_failure_with_nothing_found_is_ambiguous(
    classifier: UpdateClassifier,
) -> None:
    doi = "10.1038/nature14539"
    registry, _ = registry_for(
        {f"/works/{doi}": fixture("work_current.json"), f"/works?updates:{doi}": 503},
        query_update_notices=True,
    )
    lookup = await registry.lookup(doi)

    assert lookup.outcome is LookupOutcome.FAILED
    assert classifier.classify(lookup, paper_id=f"doi:{doi}").status is IntegrityStatus.UNKNOWN


# --- the observation invariant ------------------------------------------------------------


def test_a_status_requires_its_observation() -> None:
    with pytest.raises(ValidationError):
        IntegrityObservation(
            paper_id="doi:10.5555/x",
            identifier="doi:10.5555/x",
            status=IntegrityStatus.RETRACTED,
            basis=ObservationBasis.REGISTRY_RECORD,
            registry="crossref",
            raw_update_type="retraction",
            observation="",
        )


@pytest.mark.parametrize(
    "basis",
    [
        ObservationBasis.LOOKUP_FAILED,
        ObservationBasis.NO_LOOKUP_IDENTIFIER,
        ObservationBasis.PDF_NOTICE,
        ObservationBasis.UNMAPPED_UPDATE,
        ObservationBasis.INJECTED,
    ],
)
def test_current_cannot_come_from_anything_but_a_registry_record(
    basis: ObservationBasis,
) -> None:
    with pytest.raises(ValidationError):
        IntegrityObservation(
            paper_id="doi:10.5555/x",
            identifier="doi:10.5555/x",
            status=IntegrityStatus.CURRENT,
            basis=basis,
            registry="crossref",
            observation="looks fine",
        )


def test_registry_status_must_name_its_update_type() -> None:
    with pytest.raises(ValidationError):
        IntegrityObservation(
            paper_id="doi:10.5555/x",
            identifier="doi:10.5555/x",
            status=IntegrityStatus.RETRACTED,
            basis=ObservationBasis.REGISTRY_RECORD,
            registry="crossref",
            observation="retracted, apparently",
        )


def _observation(status: IntegrityStatus, basis: ObservationBasis) -> IntegrityObservation:
    return IntegrityObservation(
        paper_id="doi:10.5555/x",
        identifier="doi:10.5555/x",
        status=status,
        basis=basis,
        registry="crossref",
        raw_update_type="retraction" if status is IntegrityStatus.RETRACTED else None,
        observation="synthetic",
    )


def test_material_change_semantics() -> None:
    current = _observation(IntegrityStatus.CURRENT, ObservationBasis.REGISTRY_RECORD)
    retracted = _observation(IntegrityStatus.RETRACTED, ObservationBasis.REGISTRY_RECORD)
    failed = _observation(IntegrityStatus.UNKNOWN, ObservationBasis.LOOKUP_FAILED)

    assert IntegrityChange(paper_id="p", before=None, after=retracted).is_material
    assert not IntegrityChange(paper_id="p", before=None, after=current).is_material
    assert IntegrityChange(paper_id="p", before=current, after=retracted).is_material
    assert not IntegrityChange(paper_id="p", before=retracted, after=retracted).is_material
    # A failed lookup moves nothing, whatever was known before.
    assert not IntegrityChange(paper_id="p", before=retracted, after=failed).is_material


def test_severity_must_rank_every_status(integrity_config: IntegrityConfig) -> None:
    with pytest.raises(ValidationError):
        IntegrityConfig.model_validate(
            integrity_config.model_dump() | {"severity": ["retracted", "withdrawn"]}
        )


# --- service: cache, identifiers, secondary signal ----------------------------------------


class CountingRegistry:
    """Stands in for the port; counts lookups so cache behaviour is observable."""

    name = "crossref"

    def __init__(self, respond: Callable[[str], RegistryLookup]) -> None:
        self._respond = respond
        self.lookups: list[str] = []

    async def lookup(self, doi: str) -> RegistryLookup:
        self.lookups.append(doi)
        return self._respond(doi)

    async def aclose(self) -> None:
        return None


def found(doi: str, *types: str) -> RegistryLookup:
    return RegistryLookup(
        registry="crossref",
        doi=doi,
        outcome=LookupOutcome.FOUND,
        detail="synthetic",
        notices=tuple(UpdateNotice(update_type=t, relation="updated-by") for t in types),
    )


def failed(doi: str) -> RegistryLookup:
    return RegistryLookup(
        registry="crossref", doi=doi, outcome=LookupOutcome.FAILED, detail="synthetic timeout"
    )


def service_for(
    registry: CountingRegistry, tmp_path: Path, integrity_config: IntegrityConfig, **kwargs: object
) -> IntegrityService:
    return IntegrityService(
        registry,  # type: ignore[arg-type]
        JsonIntegrityRepository(tmp_path / "integrity"),
        UpdateClassifier(integrity_config),
        integrity_config.cache,
        **kwargs,  # type: ignore[arg-type]
    )


async def test_cached_observation_is_reused_until_stale(
    tmp_path: Path, integrity_config: IntegrityConfig
) -> None:
    registry = CountingRegistry(lambda doi: found(doi))
    service = service_for(registry, tmp_path, integrity_config)
    t0 = datetime(2026, 10, 1, tzinfo=UTC)

    first = await service.refresh(["doi:10.5555/a"], now=t0)
    second = await service.refresh(["doi:10.5555/a"], now=t0 + timedelta(hours=1))
    stale = await service.refresh(
        ["doi:10.5555/a"],
        now=t0 + timedelta(hours=integrity_config.cache.recheck_after_hours + 1),
    )

    assert registry.lookups == ["10.5555/a", "10.5555/a"]
    assert not first[0].cached and second[0].cached and not stale[0].cached
    assert len(await service.history("doi:10.5555/a")) == 2


async def test_failed_refresh_keeps_the_last_informative_status(
    tmp_path: Path, integrity_config: IntegrityConfig
) -> None:
    answers = iter([found("10.5555/a", "retraction"), failed("10.5555/a")])
    registry = CountingRegistry(lambda _doi: next(answers))
    service = service_for(registry, tmp_path, integrity_config)

    await service.refresh(["doi:10.5555/a"])
    refreshed = (await service.refresh(["doi:10.5555/a"], force=True))[0]

    assert refreshed.lookup_failed
    assert refreshed.observed.status is IntegrityStatus.UNKNOWN
    assert refreshed.effective is not None
    assert refreshed.effective.status is IntegrityStatus.RETRACTED


async def test_paper_without_doi_is_unknown_not_looked_up(
    tmp_path: Path, integrity_config: IntegrityConfig
) -> None:
    registry = CountingRegistry(lambda doi: found(doi))
    service = service_for(registry, tmp_path, integrity_config)

    refreshed = (await service.refresh(["arxiv:2401.12345"]))[0]

    assert registry.lookups == []
    assert refreshed.observed.status is IntegrityStatus.UNKNOWN
    assert refreshed.observed.basis is ObservationBasis.NO_LOOKUP_IDENTIFIER
    assert refreshed.observed.identifier == "arxiv:2401.12345"
    assert refreshed.effective is None


def test_identifiers_are_recovered_from_paper_ids() -> None:
    assert identifiers_from_paper_id("doi:10.1021/AM300292V").doi == "10.1021/am300292v"
    assert identifiers_from_paper_id("arxiv:2401.12345").arxiv_id == "2401.12345"
    assert not identifiers_from_paper_id("manual:01").any_present()


class StubDocuments:
    def __init__(self, text: str) -> None:
        self._text = text

    async def get(self, paper_id: str) -> object:
        from types import SimpleNamespace

        document = SimpleNamespace(metadata=SimpleNamespace(title="A paper"), full_text=self._text)
        return SimpleNamespace(value=document)


async def test_pdf_notice_raises_current_to_flagged_unknown(
    tmp_path: Path, integrity_config: IntegrityConfig
) -> None:
    registry = CountingRegistry(lambda doi: found(doi))
    service = service_for(
        registry,
        tmp_path,
        integrity_config,
        documents=StubDocuments("RETRACTED ARTICLE. This article has been retracted."),
        scanner=RetractionNoticeScanner(integrity_config.pdf_notice),
    )

    refreshed = (await service.refresh(["doi:10.5555/a"]))[0]

    assert refreshed.observed.status is IntegrityStatus.UNKNOWN
    assert refreshed.observed.basis is ObservationBasis.PDF_NOTICE
    assert refreshed.observed.flags_review
    # The registry's own answer is kept in the record, under the flag.
    history = await service.history("doi:10.5555/a")
    assert [item.basis for item in history] == [
        ObservationBasis.REGISTRY_RECORD,
        ObservationBasis.PDF_NOTICE,
    ]


async def test_pdf_notice_never_overrides_a_registry_retraction(
    tmp_path: Path, integrity_config: IntegrityConfig
) -> None:
    registry = CountingRegistry(lambda doi: found(doi, "retraction"))
    service = service_for(
        registry,
        tmp_path,
        integrity_config,
        documents=StubDocuments("this article has been retracted"),
        scanner=RetractionNoticeScanner(integrity_config.pdf_notice),
    )

    refreshed = (await service.refresh(["doi:10.5555/a"]))[0]

    assert refreshed.observed.status is IntegrityStatus.RETRACTED


def test_scanner_disabled_without_patterns() -> None:
    scanner = RetractionNoticeScanner(PdfNoticeSettings(enabled=True, patterns=()))
    assert not scanner.enabled
