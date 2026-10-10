"""Crossref integrity registry adapter.

Reads editorial-update metadata from the Crossref REST API. The fields used here were
confirmed against Crossref's own documentation and against live records (recorded as test
fixtures under ``tests/fixtures/crossref/``):

* ``GET {works_url}/{doi}`` → ``message["updated-by"]``: on the updated work, one entry per
  update notice — ``DOI`` (the notice), ``type``, ``label``, ``source`` (``publisher`` or
  ``retraction-watch``), ``updated`` (``date-parts``) and, for Retraction Watch entries,
  ``record-id``. Crossref distributes the Retraction Watch database through this field.
* ``message["update-to"]``: on a notice, the same entry shape pointing at the work it
  updates. A notice can point at itself (a withdrawn work that is its own notice), so an
  ``update-to`` entry naming the work's own DOI is read as well.
* Optional second query ``GET {works_url}?filter=updates:{doi}``: notices whose
  ``update-to`` names the DOI. Off by default — on every record checked, ``updated-by``
  already carried the same entries.

Only the raw entries are returned. What an update *type* means is configuration, applied by
``services/integrity/classifier.py``.
"""

from __future__ import annotations

from typing import Any, ClassVar
from urllib.parse import quote

from researchagent.core.exceptions import (
    SourceRateLimitedError,
    SourceResponseError,
    SourceUnavailableError,
)
from researchagent.core.interfaces.integrity import (
    IntegrityRegistry,
    LookupOutcome,
    RegistryLookup,
)
from researchagent.core.logging import get_logger
from researchagent.integrations.http import HttpClient
from researchagent.models.integrity import UpdateNotice
from researchagent.models.paper import PaperIdentifiers

logger = get_logger(__name__)

_NOT_FOUND = 404


class CrossrefIntegrityRegistry(IntegrityRegistry):
    name: ClassVar[str] = "crossref"

    def __init__(
        self,
        client: HttpClient,
        *,
        works_url: str,
        query_update_notices: bool = False,
        notice_rows: int = 20,
    ) -> None:
        self._client = client
        self._works_url = works_url.rstrip("/")
        self._query_update_notices = query_update_notices
        self._notice_rows = notice_rows

    async def lookup(self, doi: str) -> RegistryLookup:
        wanted = _normalise_doi(doi)
        try:
            payload = await self._client.get_json(f"{self._works_url}/{quote(wanted, safe='/')}")
        except SourceResponseError as exc:
            if exc.context.get("status_code") == _NOT_FOUND:
                return self._result(
                    wanted, LookupOutcome.NOT_FOUND, "crossref has no record (HTTP 404)"
                )
            return self._result(
                wanted, LookupOutcome.FAILED, f"crossref rejected the request: {exc}"
            )
        except (SourceUnavailableError, SourceRateLimitedError) as exc:
            return self._result(
                wanted, LookupOutcome.FAILED, f"crossref unreachable: {exc.message}"
            )

        message = payload.get("message") if isinstance(payload, dict) else None
        if not isinstance(message, dict):
            return self._result(
                wanted, LookupOutcome.FAILED, "crossref payload has no work message"
            )

        notices = [
            *_entries(message.get("updated-by"), relation="updated-by"),
            # Only entries that point back at this work: a notice's `update-to` names the
            # works it updates, and a self-referencing withdrawal is its own notice.
            *(
                notice
                for notice in _entries(message.get("update-to"), relation="update-to")
                if notice.notice_doi == wanted
            ),
        ]
        detail = (
            f"crossref record for {wanted}: {len(notices)} update entr(ies) in updated-by/update-to"
        )

        if self._query_update_notices:
            listed, failure = await self._notices_updating(wanted)
            if failure is not None and not notices:
                # Nothing from the record and the cross-check failed: ambiguous, not clean.
                return self._result(wanted, LookupOutcome.FAILED, failure)
            notices.extend(listed)
            detail += f"; updates-filter listed {len(listed)}" + (
                f" ({failure})" if failure else ""
            )

        return self._result(wanted, LookupOutcome.FOUND, detail, tuple(_dedupe(notices)))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _notices_updating(self, doi: str) -> tuple[list[UpdateNotice], str | None]:
        try:
            payload = await self._client.get_json(
                self._works_url,
                params={"filter": f"updates:{doi}", "rows": self._notice_rows},
            )
        except (SourceResponseError, SourceUnavailableError, SourceRateLimitedError) as exc:
            return [], f"updates-filter query failed: {exc.message}"

        items = ((payload or {}).get("message") or {}).get("items") or []
        found: list[UpdateNotice] = []
        for item in items:
            notice_doi = _normalise_doi(str(item.get("DOI") or "")) or None
            for entry in item.get("update-to") or []:
                if _normalise_doi(str(entry.get("DOI") or "")) != doi:
                    continue
                notice = _notice(entry, relation="update-to")
                if notice is not None:
                    found.append(notice.model_copy(update={"notice_doi": notice_doi}))
        return found, None

    def _result(
        self,
        doi: str,
        outcome: LookupOutcome,
        detail: str,
        notices: tuple[UpdateNotice, ...] = (),
    ) -> RegistryLookup:
        logger.info(
            "integrity_lookup",
            registry=self.name,
            doi=doi,
            outcome=outcome.value,
            notices=len(notices),
        )
        return RegistryLookup(
            registry=self.name, doi=doi, outcome=outcome, notices=notices, detail=detail
        )


def _entries(raw: Any, *, relation: str) -> list[UpdateNotice]:
    if not isinstance(raw, list):
        return []
    notices = [_notice(entry, relation=relation) for entry in raw if isinstance(entry, dict)]
    return [notice for notice in notices if notice is not None]


def _notice(entry: dict[str, Any], *, relation: str) -> UpdateNotice | None:
    update_type = str(entry.get("type") or "").strip()
    if not update_type:
        return None
    record_id = entry.get("record-id")
    return UpdateNotice(
        notice_doi=_normalise_doi(str(entry.get("DOI") or "")) or None,
        update_type=update_type,
        source=_text(entry.get("source")),
        label=_text(entry.get("label")),
        notice_date=_partial_date(entry.get("updated")),
        record_id=str(record_id) if record_id not in (None, "") else None,
        relation=relation,
    )


def _partial_date(raw: Any) -> str | None:
    """``{"date-parts": [[2019, 7, 26]]}`` → ``"2019-07-26"``; year-only stays year-only."""
    if not isinstance(raw, dict):
        return None
    parts = raw.get("date-parts")
    first = parts[0] if isinstance(parts, list) and parts and isinstance(parts[0], list) else []
    numbers: list[int] = []
    for value in first[:3]:
        try:
            numbers.append(int(value))
        except (TypeError, ValueError):
            break
    if not numbers:
        return None
    year, *rest = numbers
    return "-".join([f"{year:04d}", *(f"{value:02d}" for value in rest)])


def _dedupe(notices: list[UpdateNotice]) -> list[UpdateNotice]:
    """The same notice arrives from several sources and both relation directions.

    Distinct sources are kept (who asserted it is part of the observation); exact repeats
    of (notice, type, source, date) are not.
    """
    seen: set[tuple[str | None, str, str | None, str | None]] = set()
    unique: list[UpdateNotice] = []
    for notice in notices:
        key = (notice.notice_doi, notice.update_type.lower(), notice.source, notice.notice_date)
        if key in seen:
            continue
        seen.add(key)
        unique.append(notice)
    return unique


def _normalise_doi(value: str) -> str:
    return PaperIdentifiers(doi=value).doi or ""


def _text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None
