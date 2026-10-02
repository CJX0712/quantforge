"""AWQ (Lin et al., MLSys 2024) activation-aware per-input-channel scaling.

The scaling is exactly invertible in floating point: with ``s > 0``,

    W' = W diag(s),   X' = diag(s)⁻¹ X    =>    W' X' = W X

so the forward pass is unchanged while the quantizer sees a differently *distributed*
weight matrix. After mapping back to the original coordinates the per-element error obeys
``|E_kj| <= Δ_g(s) / (2 s_j)``: amplifying a column by ``s_j`` buys it more grid points
at a cost shared by the whole group. Hence the objective being searched is

    J(s) = (1/12) Σ_g Δ_g(s)² Σ_{j in g} E‖x_j‖² / s_j²

The salience uses ``mean(|x|)`` (not ``mean(x²)``) so a few outlier tokens cannot
dominate, and is normalised by its own mean so the scaling is dimensionless.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import QuantizationError
from ..core.types import QuantConfig
from .scale import dequantize_matrix, quantize_matrix

__all__ = ["apply_awq_scaling", "awq_available", "awq_search", "channel_salience"]


def awq_available() -> bool:
    """AWQ needs only numpy; reported through the same probe as other backends."""
    return True


def channel_salience(x_calib: np.ndarray, alpha: float, clip: tuple[float, float] = (1.0 / 3.0, 3.0)) -> np.ndarray:
    """Return per-input-channel scales ``s = (mean|X| / mean|X|)^alpha`` clipped to ``clip``.

    ``alpha=0`` gives ``s == 1`` exactly, which is why the search grid always contains
    0.0: AWQ under an identical quantizer can then never lose to RTN.
    """
    x = np.asarray(x_calib, dtype=np.float64)
    if x.ndim != 2:
        raise QuantizationError("x_calib must be 2-D (d_in, N)", shape=x.shape)
    if alpha < 0.0:
        raise QuantizationError("alpha must be non-negative", alpha=alpha)
    imp = np.abs(x).mean(axis=1)
    imp = imp / (imp.mean() + 1e-12)
    s = imp**alpha
    lo, hi = clip
    if lo <= 0.0 or lo >= hi:
        raise QuantizationError("clip must satisfy 0 < lo < hi", clip=clip)
    return np.clip(s, lo, hi)


def awq_search(
    w: np.ndarray,
    x_calib: np.ndarray,
    config: QuantConfig | None = None,
    *,
    objective: str = "acterr",
) -> dict[str, object]:
    """Grid-search the AWQ exponent.

    ``objective="acterr"`` minimises the output error ``‖Ŵ'X' - WX‖²`` (the published
    AWQ criterion); ``objective="nmse"`` minimises the weight-domain NMSE instead, which
    is what the QuantForge flagship uses because GPTQ runs afterwards and re-derives the
    output error from the same Hessian.
    """
    cfg = config or QuantConfig()
    arr = np.asarray(w, dtype=np.float64)
    x = np.asarray(x_calib, dtype=np.float64)
    if arr.ndim != 2 or x.ndim != 2:
        raise QuantizationError("w and x_calib must be 2-D", w_shape=arr.shape, x_shape=x.shape)
    if arr.shape[1] != x.shape[0]:
        raise QuantizationError("d_in mismatch between w and x_calib", d_in=arr.shape[1], calib=x.shape[0])
    if objective not in ("acterr", "nmse"):
        raise QuantizationError("unknown objective", objective=objective)

    ref = arr @ x  # computed once outside the loop: reuse avoids re-associating sums
    ref_energy = float(np.sum(ref * ref))
    w_energy = float(np.sum(arr * arr))
    best = (0.0, np.inf, None, None, None, None)
    for alpha in sorted(set(cfg.awq_alpha_grid) | {0.0}):
        s = channel_salience(x, float(alpha), cfg.awq_clip)
        w_scaled = arr * s[None, :]
        codes, scales, zeros = quantize_matrix(w_scaled, cfg)
        rec = dequantize_matrix(codes, scales, zeros)
        if objective == "acterr":
            approx = rec @ (x / s[:, None])
            err = float(np.sum((approx - ref) ** 2))
            score = err / ref_energy if ref_energy > 0.0 else err
        else:
            err = float(np.sum((rec - w_scaled) ** 2))
            score = err / w_energy if w_energy > 0.0 else err
        if score < best[1]:
            best = (float(alpha), score, s, rec, scales, zeros)

    alpha, score, s, rec, scales, zeros = best
    if alpha == 0.0 or s is None:
        s = np.ones(arr.shape[1], dtype=np.float64)
    return {
        "alpha": alpha,
        "score": float(score),
        "scale": np.asarray(s, dtype=np.float64),
        "w_hat": rec if s is not None else dequantize_matrix(*quantize_matrix(arr, cfg)),
        "scales": scales,
        "zeros": zeros,
    }


def apply_awq_scaling(w: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale ``W`` columns by ``s`` and return the inverse scale for activations.

    Absorption into the previous layer is ``W_prev / s[:, None]`` and ``b_prev / s``,
    which is exact because the composition is a no-op in floating point up to rounding.
    """
    arr = np.asarray(w, dtype=np.float64)
    scale = np.asarray(s, dtype=np.float64)
    if scale.shape != (arr.shape[1],):
        raise QuantizationError("scale must have one entry per input channel", scale_shape=scale.shape, d_in=arr.shape[1])
    if np.any(scale <= 0.0):
        raise QuantizationError("awq scale must be strictly positive")
    return arr * scale[None, :], 1.0 / scale
