"""Persistence boundary: PaperRepository, SessionRepository, VectorRepository, ...

Agents and services never touch a driver or write SQL/Cypher directly; repositories
implement ports from ``core.interfaces``. Populated from v0.3.
"""
