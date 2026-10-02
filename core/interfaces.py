"""Abstract base classes / protocols every QuantForge component implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np

from .types import LayerBundle, QuantResult

__all__ = ["DataSource", "Evaluator", "Metric", "Quantizer"]


class DataSource(ABC):
    """Produces a reproducible :class:`LayerBundle`."""

    @abstractmethod
    def build(self, seed: int) -> LayerBundle:
        """Return the bundle for dataset ``seed`` (must be deterministic)."""

    @abstractmethod
    def describe(self) -> dict[str, object]:
        """Return JSON-serialisable metadata about the generated workload."""


class Quantizer(ABC):
    """Maps ``(W, X_calib)`` to a :class:`QuantResult`."""

    #: Stable identifier used in benchmark records.
    method: str = "quantizer"

    @abstractmethod
    def quantize(self, w: np.ndarray, x_calib: np.ndarray) -> QuantResult:
        """Quantize weight matrix ``w`` of shape ``(d_out, d_in)``."""

    def available(self) -> bool:
        """Whether optional acceleration dependencies are importable."""
        return True


class Metric(ABC):
    """Scalar error metric; ``lower_is_better`` drives aggregation direction."""

    name: str = "metric"
    lower_is_better: bool = True

    @abstractmethod
    def __call__(self, w: np.ndarray, w_hat: np.ndarray, x_calib: np.ndarray) -> float:
        """Compute the metric for reference/quantized weight matrices."""


@runtime_checkable
class Evaluator(Protocol):
    """Structural interface used by the pipeline to stay backend-agnostic."""

    def evaluate(self, bundle: LayerBundle, layer: int, w_hat: np.ndarray) -> dict[str, float]:
        """Return a metric dict for one layer."""
        ...
