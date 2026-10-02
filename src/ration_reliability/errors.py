"""Exception types shared across the engine."""

from __future__ import annotations

__all__ = [
    "EngineError",
    "UnitError",
    "InvalidProblemError",
    "ConfigValidationError",
    "LeakageError",
]


class EngineError(Exception):
    """Base class of all engine errors."""


class UnitError(EngineError, ValueError):
    """Raised for an undefined unit, a dimension mismatch or an illegal basis conversion."""


class InvalidProblemError(EngineError, ValueError):
    """Raised when a problem definition (ingredients, nutrients, constraints) is inconsistent."""


class ConfigValidationError(EngineError, ValueError):
    """Raised by the YAML validator (contract T9).  ``issues`` holds every error found."""

    def __init__(self, issues: list[str]):
        self.issues = list(issues)
        msg = "configuration rejected ({} issue(s)):\n  - ".format(len(self.issues))
        super().__init__(msg + "\n  - ".join(self.issues))


class LeakageError(EngineError, RuntimeError):
    """Raised when information that must be hidden at decision time reaches a decision step.

    Example: a method that fits on the ``opt`` random stream receives ``test`` draws.
    """
