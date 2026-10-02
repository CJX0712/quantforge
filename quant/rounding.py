"""Deterministic rounding primitives.

The whole project uses ``floor(x + 0.5)`` (round-half-up) instead of ``np.round``.
Rationale: ``np.round`` is round-half-to-even (banker's rounding), so ``0.5 -> 0``
and ``1.5 -> 2``. That is unbiased but makes "what does an exact tie do?" depend on
the value's parity, which is surprising when debugging a quantizer. Round-half-up is
one rule, easy to state, and platform independent. It is applied identically during
calibration and export so a codebook never drifts between the two phases.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import QuantizationError

__all__ = [
    "dequantize_affine",
    "qmin_qmax",
    "quantize_affine",
    "quantize_with_params",
    "round_half_up",
]


def qmin_qmax(bits: int, scheme: str = "symmetric") -> tuple[int, int]:
    """Return the integer representable range ``(qmin, qmax)``.

    Symmetric keeps the grid centred on zero (``qmin == -qmax``) so ``0`` is exactly
    representable and no zero-point has to be stored. Asymmetric uses the full
    ``[-2^(b-1), 2^(b-1)-1]`` range, which matters for post-ReLU activations whose
    minimum is exactly ``0``.
    """
    if not 2 <= bits <= 8:
        raise QuantizationError("bits must be in [2, 8]", bits=bits)
    qmax = 2 ** (bits - 1) - 1
    if scheme == "symmetric":
        return -qmax, qmax
    if scheme == "asymmetric":
        return -(2 ** (bits - 1)), qmax
    raise QuantizationError("unknown scheme", scheme=scheme)


def round_half_up(x: np.ndarray) -> np.ndarray:
    """Round half away from +inf, elementwise. Never uses Python's ``int()``.

    ``int(-1.2) == -1`` (truncation toward zero) while ``floor(-1.2) == -2``; mixing
    the two silently corrupts asymmetric zero-points on the negative half axis.
    """
    return np.floor(np.asarray(x, dtype=np.float64) + 0.5)


def quantize_affine(x: np.ndarray, scale: float, zero: int, bits: int, scheme: str = "symmetric") -> np.ndarray:
    """Quantize ``x`` to integer codes ``clip(round(x/s) + z, qmin, qmax)``."""
    qmin, qmax = qmin_qmax(bits, scheme)
    s = float(scale)
    if not np.isfinite(s) or s <= 0.0:
        raise QuantizationError("scale must be finite and positive", scale=s)
    codes = round_half_up(np.asarray(x, dtype=np.float64) / s) + int(zero)
    return np.clip(codes, qmin, qmax).astype(np.int64)


def dequantize_affine(codes: np.ndarray, scale: np.ndarray | float, zero: np.ndarray | int) -> np.ndarray:
    """Inverse of :func:`quantize_affine`; ``scale``/``zero`` broadcast against ``codes``."""
    return (np.asarray(codes, dtype=np.float64) - np.asarray(zero, dtype=np.float64)) * np.asarray(
        scale, dtype=np.float64
    )


def quantize_with_params(
    x: np.ndarray, scale: np.ndarray | float, zero: np.ndarray | int, bits: int, scheme: str = "symmetric"
) -> np.ndarray:
    """Quantize with per-group scales broadcast over the last axis.

    ``x`` has shape ``(d_out, d_in)``; ``scale`` has shape ``(d_out, n_groups)`` or
    ``(n_groups,)``. Scales are tiled to match column blocks, so column ``j`` always
    uses the scale of the group that owns it (never a shifted group).
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise QuantizationError("quantize_with_params expects a 2-D array", shape=arr.shape)
    d_out, d_in = arr.shape
    s = np.asarray(scale, dtype=np.float64)
    z = np.asarray(zero, dtype=np.int64)

    n_groups = 1 if s.ndim == 0 else s.shape[-1]
    if s.ndim == 0:
        s = np.full((d_out, 1), float(s))
    elif s.ndim == 1:
        s = np.broadcast_to(s, (d_out, n_groups))
    if z.ndim == 0:
        z = np.full((d_out, n_groups), int(z), dtype=np.int64)
    elif z.ndim == 1:
        z = np.broadcast_to(z, (d_out, n_groups))
    # When scale is already 2-D it *is* the (d_out, n_groups) layout, so compare against
    # its own group count -- not against a count derived from the same array, which
    # would make the check vacuous.
    expected = (d_out, s.shape[-1] if s.ndim == 2 else n_groups)
    if s.shape != expected or z.shape != expected:
        raise QuantizationError(
            "scale/zero shape mismatch", scale_shape=s.shape, zero_shape=z.shape, expected=expected
        )
    n_groups = expected[1]
    if not 1 <= n_groups <= d_in:
        raise QuantizationError(
            "n_groups must lie in [1, d_in]", n_groups=n_groups, d_in=d_in, scale_shape=s.shape
        )

    # column -> owning group, via ceil so a short trailing group still maps correctly
    owner = np.clip(
        np.searchsorted(np.linspace(0, d_in, n_groups + 1), np.arange(d_in), side="right") - 1,
        0,
        n_groups - 1,
    )
    s_cols = s[:, owner]
    z_cols = z[:, owner]

    qmin, qmax = qmin_qmax(bits, scheme)
    codes = round_half_up(arr / s_cols) + z_cols
    return np.clip(codes, qmin, qmax).astype(np.int64)
