"""Typed error hierarchy with stable numeric codes (E100-E500).

Every error carries ``code`` so callers/tests can assert on identity, not message text.
"""

from __future__ import annotations

__all__ = [
    "ERROR_CODES",
    "ConfigError",
    "DataError",
    "DecompositionError",
    "MetricError",
    "PipelineError",
    "QuantForgeError",
    "QuantizationError",
    "ShapeError",
    "ValidationError",
    "error_code_of",
]


class QuantForgeError(Exception):
    """Base class. ``code`` is a stable string like ``E200``."""

    code = "E000"

    def __init__(self, message: str, **context: object) -> None:
        self.context = dict(context)
        suffix = "".join(f" [{k}={v!r}]" for k, v in sorted(self.context.items()))
        super().__init__(f"{self.code}: {message}{suffix}")


class ConfigError(QuantForgeError):
    code = "E100"


class ValidationError(QuantForgeError):
    code = "E110"


class DataError(QuantForgeError):
    code = "E200"


class ShapeError(QuantForgeError):
    code = "E210"


class QuantizationError(QuantForgeError):
    code = "E300"


class DecompositionError(QuantForgeError):
    code = "E310"


class MetricError(QuantForgeError):
    code = "E400"


class PipelineError(QuantForgeError):
    code = "E500"


ERROR_CODES: dict[str, type[QuantForgeError]] = {
    cls.code: cls
    for cls in (
        QuantForgeError,
        ConfigError,
        ValidationError,
        DataError,
        ShapeError,
        QuantizationError,
        DecompositionError,
        MetricError,
        PipelineError,
    )
}


def error_code_of(exc: BaseException) -> str:
    """Best-effort extraction of a QuantForge error code from any exception."""
    if isinstance(exc, QuantForgeError):
        return exc.code
    return "E999"
