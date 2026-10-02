"""Typed contracts shared by every QuantForge subpackage.

Call graph is strictly acyclic::

    cli -> pipeline -> {data, quant, eval} -> core

Nothing in :mod:`quantforge.core` may import a sibling subpackage.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np

from .errors import ShapeError

__all__ = [
    "Benchmark",
    "Granularity",
    "LayerBundle",
    "Method",
    "QuantConfig",
    "QuantResult",
    "RunRecord",
    "Scheme",
    "as_2d_float64",
    "as_float64",
]

Scheme = Literal["symmetric", "asymmetric"]
Method = Literal["rtn_tensor", "rtn_group", "awq", "quantforge"]
Granularity = Literal["per_tensor", "per_channel", "per_group"]


def as_float64(array: Any) -> np.ndarray:
    """Return ``array`` as a C-contiguous float64 ndarray (never a view of an int array)."""
    out = np.asarray(array, dtype=np.float64)
    return np.ascontiguousarray(out)


def as_2d_float64(array: Any, name: str = "array") -> np.ndarray:
    out = as_float64(array)
    if out.ndim != 2:
        raise ValueError(f"{name} must be 2-D (d_out, d_in), got shape {out.shape}")
    return out


@dataclass(frozen=True, slots=True)
class QuantConfig:
    """Immutable quantizer configuration.

    All fields are validated in ``__post_init__``; no silent clamping.
    """

    bits: int = 4
    scheme: Scheme = "symmetric"
    granularity: Granularity = "per_group"
    group_size: int = 128
    scale_bits: int = 16
    zero_bits: int = 4
    # GPTQ knobs
    #: relative damping ``lam = percdamp * mean(diag(H))``, i.e. ``mu/lam = 1/percdamp``.
    #: The default is ``1/SAFETY_TARGET``; see :data:`quantforge.core.config.SAFETY_TARGET`
    #: for the measurement behind it.
    percdamp: float = 1.0 / 3.0
    blocksize: int = 128
    act_order: bool = True
    static_groups: bool = True
    # AWQ knobs
    awq_alpha_grid: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    awq_clip: tuple[float, float] = (1.0 / 3.0, 3.0)
    # MSE-optimal clip search
    clip_grid: tuple[float, ...] = tuple(round(0.50 + 0.01 * i, 2) for i in range(51))
    #: search a per-group MSE-optimal clipping ratio instead of plain min-max
    use_mse_clip: bool = True
    #: closed-form least-squares refit of group scales after GPTQ (recovers weight NMSE)
    refit_scales: bool = True
    # bookkeeping
    name: str = "default"

    def __post_init__(self) -> None:
        if not 2 <= self.bits <= 8:
            raise ValueError(f"bits must be in [2, 8], got {self.bits}")
        if self.scheme not in ("symmetric", "asymmetric"):
            raise ValueError(f"unknown scheme {self.scheme!r}")
        if self.granularity not in ("per_tensor", "per_channel", "per_group"):
            raise ValueError(f"unknown granularity {self.granularity!r}")
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")
        if self.blocksize <= 0:
            raise ValueError(f"blocksize must be positive, got {self.blocksize}")
        if not 0.0 < self.percdamp < 1.0:
            raise ValueError(f"percdamp must be in (0, 1), got {self.percdamp}")
        if not self.awq_alpha_grid:
            raise ValueError("awq_alpha_grid must be non-empty (alpha=0 guarantees RTN parity)")
        if min(self.awq_alpha_grid) < 0.0 or max(self.awq_alpha_grid) > 1.0:
            raise ValueError("awq_alpha_grid entries must lie in [0, 1]")

    def evolve(self, **changes: Any) -> QuantConfig:
        """Return a copy with ``changes`` applied (config is frozen)."""
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class QuantResult:
    """Output of any quantizer: dequantized weights plus quantization metadata."""

    w_hat: np.ndarray
    codes: np.ndarray
    scales: np.ndarray
    zeros: np.ndarray
    bits_per_weight: float
    aux: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.w_hat.ndim != 2:
            raise ValueError("w_hat must be 2-D")


@dataclass(frozen=True, slots=True)
class LayerBundle:
    """A synthetic MLP layer stack: weights, activations and calibration sets.

    ``x_calib[i]`` are the calibration activations *feeding* layer ``i``; because layer
    widths differ in a deep MLP, each layer needs its own calibration matrix. The
    property :attr:`calib` exposes layer 0's, which is what single-layer experiments use.
    """

    weights: tuple[np.ndarray, ...]
    acts: tuple[np.ndarray, ...]
    x_calib: np.ndarray
    name: str = "mlp"
    per_layer_calib: tuple[np.ndarray, ...] = ()

    @property
    def n_layers(self) -> int:
        return len(self.weights)

    def layer(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(W, X_calib)`` for layer ``index``.

        The calibration matrix always has ``d_in`` rows, matching ``W``'s column count.
        """
        if not 0 <= index < self.n_layers:
            raise IndexError(f"layer index {index} out of range (n_layers={self.n_layers})")
        x = self.per_layer_calib[index] if self.per_layer_calib else self.x_calib
        w = self.weights[index]
        if x.shape[0] != w.shape[1]:
            raise ShapeError(
                "calibration width does not match layer input",
                calib_rows=int(x.shape[0]),
                d_in=int(w.shape[1]),
                layer=index,
            )
        return w, x


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One (seed, model, layer, bits, group_size, method) measurement."""

    dataset_seed: int
    model: str
    layer: int
    bits: int
    group_size: int
    method: str
    nmse: float
    act_err: float
    sqnr_db: float
    bits_per_weight: float
    compression_ratio: float
    wall_sec: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_seed": self.dataset_seed,
            "model": self.model,
            "layer": self.layer,
            "bits": self.bits,
            "group_size": self.group_size,
            "method": self.method,
            "nmse": self.nmse,
            "act_err": self.act_err,
            "sqnr_db": self.sqnr_db,
            "bits_per_weight": self.bits_per_weight,
            "compression_ratio": self.compression_ratio,
            "wall_sec": self.wall_sec,
        }


@dataclass(frozen=True, slots=True)
class Benchmark:
    """Aggregate report emitted by the pipeline."""

    records: tuple[RunRecord, ...]
    method_summary: Mapping[str, Mapping[str, float]]
    baseline: str
    flagship: str
    relative_nmse_drop_pct: float
    significant: bool
    ablations: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    failures: tuple[str, ...] = ()
    meta: Mapping[str, Any] = field(default_factory=dict)
