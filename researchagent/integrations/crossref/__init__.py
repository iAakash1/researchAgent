"""Crossref adapters: literature discovery and integrity (retraction/update) lookups."""

from researchagent.integrations.crossref.integrity import CrossrefIntegrityRegistry
from researchagent.integrations.crossref.source import CrossrefSource

__all__ = ["CrossrefIntegrityRegistry", "CrossrefSource"]
