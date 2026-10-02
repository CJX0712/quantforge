"""Error metrics. Definitions are fixed and must not drift.

* ``NMSE = ‖W - Ŵ‖_F² / ‖W‖_F²``  -- weight domain, primary metric, lower is better
* ``ActErr = ‖ŴX - WX‖_F² / ‖WX‖_F²``  -- output domain, lower is better
* ``SQNR_dB = 10 log10(Σw² / Σ(w-ŵ)²) = -10 log10(NMSE)``  -- higher is better

An all-zero reference has no defined NMSE; these functions then return ``inf`` rather
than raising, so a degenerate layer shows up as a visibly bad score instead of a crash.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import MetricError
from ..core.interfaces import Metric

__all__ = ["NMSE", "SQNR", "ActErr", "act_err", "all_metrics", "compression_ratio", "nmse", "sqnr_db"]

_EPS = 1e-30


def _check_pair(w: np.ndarray, w_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(w, dtype=np.float64)
    b = np.asarray(w_hat, dtype=np.float64)
    if a.shape != b.shape:
        raise MetricError("shape mismatch between reference and estimate", reference=a.shape, estimate=b.shape)
    return a, b


def nmse(w: np.ndarray, w_hat: np.ndarray, x_calib: np.ndarray | None = None) -> float:
    """Weight-domain normalized mean squared error."""
    a, b = _check_pair(w, w_hat)
    denom = float(np.sum(a * a))
    err = float(np.sum((a - b) ** 2))
    return err / denom if denom > _EPS else float("inf")


def act_err(w: np.ndarray, w_hat: np.ndarray, x_calib: np.ndarray) -> float:
    """Output-domain normalized error ``‖ŴX - WX‖²_F / ‖WX‖²_F``."""
    a, b = _check_pair(w, w_hat)
    x = np.asarray(x_calib, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != a.shape[1]:
        raise MetricError("x_calib must be (d_in, N) matching w", x_shape=x.shape, d_in=a.shape[1])
    ref = a @ x
    approx = b @ x
    denom = float(np.sum(ref * ref))
    err = float(np.sum((approx - ref) ** 2))
    return err / denom if denom > _EPS else float("inf")


def sqnr_db(w: np.ndarray, w_hat: np.ndarray, x_calib: np.ndarray | None = None) -> float:
    """Signal-to-quantization-noise ratio in dB; exactly ``-10 log10(NMSE)``."""
    value = nmse(w, w_hat, x_calib)
    if not np.isfinite(value) or value <= 0.0:
        return float("inf") if value <= 0.0 else float("-inf")
    return float(-10.0 * np.log10(value))


def compression_ratio(bits_per_weight: float, fp32_bits: int = 32) -> float:
    """``fp32_bits / bits_per_weight``."""
    if bits_per_weight <= 0.0:
        raise MetricError("bits_per_weight must be positive", bits_per_weight=bits_per_weight)
    return fp32_bits / float(bits_per_weight)


class NMSE(Metric):
    name = "nmse"
    lower_is_better = True

    def __call__(self, w: np.ndarray, w_hat: np.ndarray, x_calib: np.ndarray | None = None) -> float:
        return nmse(w, w_hat, x_calib)


class ActErr(Metric):
    name = "act_err"
    lower_is_better = True

    def __call__(self, w: np.ndarray, w_hat: np.ndarray, x_calib: np.ndarray | None = None) -> float:
        if x_calib is None:
            raise MetricError("ActErr requires calibration activations")
        return act_err(w, w_hat, x_calib)


class SQNR(Metric):
    name = "sqnr_db"
    lower_is_better = False

    def __call__(self, w: np.ndarray, w_hat: np.ndarray, x_calib: np.ndarray | None = None) -> float:
        return sqnr_db(w, w_hat, x_calib)


def all_metrics(
    w: np.ndarray, w_hat: np.ndarray, x_calib: np.ndarray | None = None, bits_per_weight: float = 4.0
) -> dict[str, float]:
    """Compute every reported metric in one pass."""
    out: dict[str, float] = {"nmse": nmse(w, w_hat, x_calib), "sqnr_db": sqnr_db(w, w_hat, x_calib)}
    if x_calib is not None:
        out["act_err"] = act_err(w, w_hat, x_calib)
    out["bits_per_weight"] = float(bits_per_weight)
    out["compression_ratio"] = compression_ratio(bits_per_weight)
    return out
