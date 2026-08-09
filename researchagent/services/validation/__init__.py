"""Validation services: one validator, one question, one ValidationResult.

Deliberately no aggregate re-exports. Importing this package must not drag in every
validator (and through them, every subsystem they validate) — callers import the module
that owns the validator they need.
"""
