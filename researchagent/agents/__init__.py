"""Agents: single-responsibility reasoning units.

One subpackage per agent, always the same shape::

    agents/<name>/
        agent.py      # BaseAgent subclass — reasoning only
        schemas.py    # typed input/output contract
        prompt.py     # message assembly from prompts/<name>/<version>.md

Agents decide; services do the I/O. The roster lives in ``registry.py``.
"""
