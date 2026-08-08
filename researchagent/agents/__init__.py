"""Agents: single-responsibility reasoning units.

One subpackage per agent, always the same shape::

    agents/<name>/
        agent.py      # BaseAgent subclass — reasoning only
        schemas.py    # typed input/output contract
        prompt.py     # message assembly from prompts/<name>/<version>.md

Agents never perform I/O directly; they call services, which call integrations.

Importing this package registers the built-in agents in ``AGENTS``.
"""

from researchagent.agents.base import AgentContext, AgentResult, BaseAgent

# Imported for its registration side-effect: this is what puts "planner" in AGENTS.
from researchagent.agents.planner import PlannerAgent
from researchagent.agents.registry import AGENTS, build_agent

__all__ = ["AGENTS", "AgentContext", "AgentResult", "BaseAgent", "PlannerAgent", "build_agent"]
