"""Scale/zero-point computation for the three granularities, plus MSE-optimal clipping.

Granularity semantics (quantization runs along the *column* / input-channel axis of a
``(d_out, d_in)`` weight matrix):

* ``per_tensor``  -- one scale for the whole matrix
* ``per_channel`` -- one scale per input channel (column), shape ``(d_out, 1)`` after tiling
* ``per_group``   -- one scale per ``group_size`` consecutive columns

Storage overhead (``n`` = number of weights) with fp16 scales and a 4-bit zero-point::

    per_tensor  b + (16 + 4) / n
    per_channel b + (16 + 4) / d_in
    per_group   b + (16 + 4) / g_eff          g_eff = min(group_size, d_in) for the tail
"""

from __future__ import annotations

import numpy as np

from ..core.errors import QuantizationError
from ..core.types import QuantConfig
from .rounding import dequantize_affine, qmin_qmax, quantize_with_params, round_half_up

__all__ = [
    "bits_per_weight",
    "compute_scale",
    "dequantize_matrix",
    "group_bounds",
    "n_groups_for",
    "quantize_matrix",
    "search_clip_alpha",
]


def group_bounds(d_in: int, group_size: int) -> list[tuple[int, int]]:
    """Return ``[(k0, k1), ...]`` column slices; the tail group is short, never padded.

    Padding the weight matrix with zeros would raise ``max|x|`` and therefore change
    the scale of the *whole* trailing group, corrupting its codebook. The trailing
    group is simply shorter.
    """
    if group_size <= 0:
        raise QuantizationError("group_size must be positive", group_size=group_size)
    if d_in <= 0:
        raise QuantizationError("d_in must be positive", d_in=d_in)
    return [(k0, min(k0 + group_size, d_in)) for k0 in range(0, d_in, group_size)]


def n_groups_for(d_in: int, config: QuantConfig) -> int:
    if config.granularity == "per_tensor":
        return 1
    if config.granularity == "per_channel":
        return d_in
    return len(group_bounds(d_in, config.group_size))


def compute_scale(
    block: np.ndarray, bits: int, scheme: str = "symmetric", clip_alpha: float = 1.0
) -> tuple[float, int]:
    """Return ``(scale, zero)`` for one block of values.

    ``clip_alpha`` shrinks the min-max range; ``alpha=1.0`` disables clipping. A few
    outlier weights inflate ``max|x|``, squeezing the bulk of the distribution into a
    handful of levels, so searching ``alpha`` usually buys real accuracy at low bit
    widths.
    """
    arr = np.asarray(block, dtype=np.float64)
    if arr.size == 0:
        raise QuantizationError("cannot compute a scale from an empty block")
    qmin, qmax = qmin_qmax(bits, scheme)
    if not 0.0 < clip_alpha <= 1.0:
        raise QuantizationError("clip_alpha must lie in (0, 1]", clip_alpha=clip_alpha)

    if scheme == "symmetric":
        amax = float(np.abs(arr).max()) * clip_alpha
        if amax == 0.0:  # all-zero block (dead channel / masked layer)
            return 1.0, 0
        scale = amax / qmax
        return scale, 0

    lo = float(arr.min()) * clip_alpha
    hi = float(arr.max()) * clip_alpha
    if hi == lo:
        return 1.0, int(np.clip(qmin, qmin, qmax))
    scale = (hi - lo) / (qmax - qmin)
    if scale <= 0.0 or not np.isfinite(scale):
        return 1.0, int(np.clip(qmin, qmin, qmax))
    zero = int(np.clip(round_half_up(np.array(-lo / scale)), qmin, qmax))
    return scale, zero


def search_clip_alpha(
    block: np.ndarray,
    bits: int,
    scheme: str = "symmetric",
    grid: tuple[float, ...] | np.ndarray | None = None,
) -> tuple[float, float]:
    """Grid-search the clipping ratio minimising NMSE. Returns ``(alpha, nmse)``.

    ``alpha=1.0`` is always in the grid, so "do not clip" is always a candidate and
    the search can never do worse than plain min-max. The curve is piecewise smooth
    with kinks where values cross a clipping boundary or hit a rounding tie, so a fine
    grid plus one local refinement is used instead of a golden section that could step
    over a kink.
    """
    arr = np.asarray(block, dtype=np.float64)
    if grid is None:
        grid = tuple(round(0.50 + 0.01 * i, 2) for i in range(51))
    alphas = np.asarray(sorted({1.0, *[float(a) for a in grid]}), dtype=np.float64)
    alphas = alphas[(alphas > 0.0) & (alphas <= 1.0)]

    def nmse(alpha: float) -> float:
        scale, zero = compute_scale(arr, bits, scheme, alpha)
        codes = quantize_with_params(arr, scale, zero, bits, scheme)
        rec = dequantize_affine(codes, scale, zero)
        denom = float(np.sum(arr * arr))
        err = float(np.sum((arr - rec) ** 2))
        return err / denom if denom > 0.0 else err

    scores = np.array([nmse(a) for a in alphas], dtype=np.float64)
    best = int(np.argmin(scores))
    alpha = float(alphas[best])
    # local refinement around the coarse optimum (step 0.01 -> 0.001)
    fine = np.arange(max(0.50, alpha - 0.01), min(1.0, alpha + 0.01) + 1e-9, 0.001)
    for a in fine:
        val = nmse(float(a))
        if val < scores[best]:
            alpha, scores[best] = float(a), val
    return alpha, float(scores[best])


def quantize_matrix(
    w: np.ndarray, config: QuantConfig, clip_alpha: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize ``w`` per config. Returns ``(codes, scales, zeros)``.

    ``scales`` has shape ``(d_out, n_groups)`` so a non-square matrix keeps
    per-output-row scales distinguishable from per-input-column scales.
    """
    arr = np.asarray(w, dtype=np.float64)
    if arr.ndim != 2:
        raise QuantizationError("quantize_matrix expects a 2-D array", shape=arr.shape)
    d_out, d_in = arr.shape
    if config.granularity == "per_tensor":
        bounds = [(0, d_in)]
    elif config.granularity == "per_channel":
        bounds = [(j, j + 1) for j in range(d_in)]
    else:
        bounds = group_bounds(d_in, config.group_size)

    n_groups = len(bounds)
    codes = np.zeros((d_out, d_in), dtype=np.int64)
    scales = np.zeros((d_out, n_groups), dtype=np.float64)
    zeros = np.zeros((d_out, n_groups), dtype=np.int64)
    for gi, (k0, k1) in enumerate(bounds):
        block = arr[:, k0:k1]
        s, z = compute_scale(block, config.bits, config.scheme, clip_alpha)
        scales[:, gi] = s
        zeros[:, gi] = z
        codes[:, k0:k1] = quantize_with_params(block, s, z, config.bits, config.scheme)
    return codes, scales, zeros


def dequantize_matrix(
    codes: np.ndarray, scales: np.ndarray, zeros: np.ndarray, bounds: list[tuple[int, int]] | None = None
) -> np.ndarray:
    """Rebuild float weights from integer codes (exact inverse of :func:`quantize_matrix`).

    ``scales``/``zeros`` are ``(d_out, n_groups)`` and get tiled across the columns each
    group owns, so column ``j`` is reconstructed with the scale of its own group. Pass
    ``bounds`` to guarantee the owner mapping matches the slicing exactly.
    """
    c = np.asarray(codes, dtype=np.int64)
    s = np.asarray(scales, dtype=np.float64)
    z = np.asarray(zeros, dtype=np.int64)
    d_in = c.shape[1]
    if bounds is not None:
        owner = column_owner(bounds, d_in)
    else:
        n_groups = s.shape[1] if s.ndim == 2 else int(s.size)
        owner = _column_owner(d_in, n_groups)
    return dequantize_affine(c, s[:, owner], z[:, owner])


def column_owner(bounds: list[tuple[int, int]], d_in: int) -> np.ndarray:
    """Map each column index to the group that owns it.

    Derived from the *same* ``[(k0, k1), ...]`` list used to slice the matrix, so the
    owner mapping can never disagree with the slicing. Deriving it independently (e.g.
    via ``np.linspace``) produces fractional edges when ``group_size`` does not divide
    ``d_in`` and silently assigns columns to the wrong group.
    """
    owner = np.zeros(d_in, dtype=np.int64)
    for gi, (k0, k1) in enumerate(bounds):
        owner[k0:k1] = gi
    return owner


def _column_owner(d_in: int, n_groups: int) -> np.ndarray:
    """Fallback owner mapping when only a group *count* is known.

    Splits ``d_in`` into ``n_groups`` near-equal contiguous blocks, distributing the
    remainder to the first blocks.
    """
    if n_groups <= 0:
        raise QuantizationError("n_groups must be positive", n_groups=n_groups)
    base, rem = divmod(d_in, n_groups)
    owner = np.empty(d_in, dtype=np.int64)
    start = 0
    for g in range(n_groups):
        width = base + (1 if g < rem else 0)
        owner[start : start + width] = g
        start += width
    return owner


def bits_per_weight(config: QuantConfig, n_weights: int, d_in: int) -> float:
    """Effective bits per weight, counting scale and zero-point storage.

    One scale (and, for asymmetric, one zero-point) is stored per *group*, shared by
    all output rows -- that is the layout GPTQ/AWQ kernels assume. Symmetric weights
    store no zero-point (it is identically 0), so only the fp16 scale is charged::

        bpw = bits + n_groups * (scale_bits + zero_bits) / n_weights

    For ``per_group`` with ``g`` covering ``d_in`` inputs this reduces to
    ``bits + 16*ceil(d_in/g)/d_in`` (e.g. 4.125 at bits=4, g=128, d_in=128).
    """
    if n_weights <= 0:
        raise QuantizationError("n_weights must be positive", n_weights=n_weights)
    n_groups = n_groups_for(d_in, config)
    zero_bits = 0 if config.scheme == "symmetric" else config.zero_bits
    stored_bits = n_groups * (config.scale_bits + zero_bits)
    return config.bits + stored_bits / float(n_weights)
