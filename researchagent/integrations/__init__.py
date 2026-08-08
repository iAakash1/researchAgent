"""Outbound adapters: one subpackage per external system.

Each adapter implements a port from ``researchagent.core.interfaces`` and is the only
place a vendor SDK may be imported. Present: ``ollama``. Planned as their subsystems
land: ``qdrant``, ``neo4j``, ``arxiv``, ``semantic_scholar``, ``crossref``,
``openalex``, ``pubmed``.
"""
