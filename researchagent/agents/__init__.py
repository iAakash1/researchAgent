"""Agents: single-responsibility reasoning units.

One subpackage per agent, always the same shape::

    agents/<name>/
        agent.py      # BaseAgent subclass — reasoning only
        schemas.py    # typed input/output contract
        prompt.py     # prompt assembly from prompts/<name>/<version>.md

Agents never perform I/O directly; they call services, which call integrations.
"""

from researchagent.agents.base import AgentContext, AgentResult, BaseAgent
from researchagent.agents.registry import AGENTS

__all__ = ["AGENTS", "AgentContext", "AgentResult", "BaseAgent"]
