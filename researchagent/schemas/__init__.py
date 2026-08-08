"""Cross-boundary contracts: agent I/O schemas and workflow state.

Agents exchange validated Pydantic models, never dictionaries. Populated per agent
from v0.2 onward (PlannerOutput, DiscoveryOutput, VerificationOutput, ...).
"""
