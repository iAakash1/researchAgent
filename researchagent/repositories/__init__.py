"""Persistence adapters implementing ports from ``core.interfaces``.

Agents and services never touch a driver or write SQL/Cypher directly.
"""

from researchagent.repositories.paper_repository import JsonPaperRepository

__all__ = ["JsonPaperRepository"]
