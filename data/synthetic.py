"""Synthetic calibration workloads with deliberate activation outliers.

Why the outliers matter: an activation-aware scale is only useful when channel
importances are *unequal*. On clean Gaussian activations every channel has the same
energy, the AWQ scaling collapses to a constant, and methods A/B/C/D all tie -- the
experiment would show nothing. So a small fraction of channels get a large gain, and
only a few calibration samples carry large values in those channels. That combination
(a few high-energy channels + a few samples) is what makes ``H = 2XXᵀ`` ill-conditioned
and gives GPTQ something to compensate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.errors import DataError
from ..core.interfaces import DataSource
from ..core.types import LayerBundle

__all__ = ["MODEL_SHAPES", "MODEL_SPECS", "SyntheticConfig", "SyntheticMLP", "make_layer", "make_mlp"]

#: name -> (d_in, d_out) for the first layer; deeper layers follow the previous width.
MODEL_SPECS: dict[str, tuple[int, int]] = {
    "m0": (64, 32),
    "m1": (32, 16),
    "m2": (128, 64),
    "m3": (96, 48),
}

#: name -> ((d_out, d_in), ...) weight-matrix shape of each layer, matching the
#: ``(d_out, d_in)`` convention of :class:`QuantConfig` and every quantizer.
MODEL_SHAPES: dict[str, tuple[tuple[int, int], ...]] = {
    "m0": ((32, 64), (32, 32)),
    "m1": ((16, 32), (16, 16)),
    "m2": ((64, 128), (64, 64)),
    "m3": ((48, 96), (48, 48)),
}


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    """Parameters of the synthetic MLP workload."""

    d_in: int = 64
    d_out: int = 32
    depth: int = 2
    act_scale: float = 1.0
    outlier_dim: int = 4
    outlier_frac: float = 0.04
    outlier_gain: float = 20.0
    n_calib: int = 512
    n_eval: int = 128

    def __post_init__(self) -> None:
        if self.d_in <= 0 or self.d_out <= 0 or self.depth <= 0:
            raise DataError("d_in, d_out and depth must be positive", d_in=self.d_in, d_out=self.d_out, depth=self.depth)
        if self.n_calib <= 0 or self.n_eval <= 0:
            raise DataError("n_calib and n_eval must be positive", n_calib=self.n_calib, n_eval=self.n_eval)
        if not 0.0 <= self.outlier_frac < 1.0:
            raise DataError("outlier_frac must lie in [0, 1)", outlier_frac=self.outlier_frac)
        if self.outlier_dim < 0 or self.outlier_dim > self.d_in:
            raise DataError("outlier_dim must lie in [0, d_in]", outlier_dim=self.outlier_dim, d_in=self.d_in)
        if self.outlier_gain < 1.0:
            raise DataError("outlier_gain must be >= 1", outlier_gain=self.outlier_gain)


def _inject_outliers(
    acts: np.ndarray, rng: np.random.Generator, cfg: SyntheticConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(acts_with_outliers, outlier_mask)``.

    A handful of channels are amplified by ``outlier_gain`` and only
    ``outlier_frac`` of the samples receive that large value, reproducing the
    "few outlier tokens on a few outlier channels" structure of real LLM activations.
    """
    n_samples = acts.shape[1]
    if cfg.outlier_dim == 0 or cfg.outlier_frac <= 0.0:
        return acts, np.zeros(acts.shape, dtype=bool)

    channels = rng.choice(acts.shape[0], size=cfg.outlier_dim, replace=False)
    channels.sort()
    out = acts.copy()
    mask = np.zeros(acts.shape, dtype=bool)
    n_hot = max(1, round(cfg.outlier_frac * n_samples))
    for ch in channels:
        # per-channel hot samples, not one shared set: real outlier channels trip on
        # different tokens, and sharing one set would correlate them artificially.
        hot = rng.choice(n_samples, size=n_hot, replace=False)
        out[ch, hot] *= cfg.outlier_gain
        mask[ch, hot] = True
    return out, mask


def make_layer(
    seed: int,
    d_in: int,
    d_out: int,
    n_calib: int = 512,
    act_scale: float = 1.0,
    outlier_dim: int = 4,
    outlier_frac: float = 0.04,
    outlier_gain: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one layer. Returns ``(weights, acts, x_calib, outlier_mask)``.

    ``acts`` are *input* activations of shape ``(d_in, n_calib)``; ``x_calib`` is the
    same matrix as consumed by GPTQ/AWQ. Weights are heavy-tailed (student-t-ish via a
    normal/chi mix) because Gaussian weights make clipping look useless.
    """
    cfg = SyntheticConfig(
        d_in=d_in,
        d_out=d_out,
        outlier_dim=outlier_dim,
        outlier_frac=outlier_frac,
        outlier_gain=outlier_gain,
        n_calib=n_calib,
    )
    rng = np.random.default_rng(seed)
    weights = rng.standard_normal((d_out, d_in)) * 1.0 / np.sqrt(d_in)
    # heavy tail: mix a few large entries into a Gaussian background
    heavy = rng.random((d_out, d_in)) < 0.02
    weights[heavy] *= 6.0
    acts = rng.standard_normal((d_in, n_calib)) * act_scale
    acts = np.maximum(acts, 0.0)  # post-ReLU, matching a real hidden layer
    acts, mask = _inject_outliers(acts, rng, cfg)
    return weights, acts, acts, mask


def make_mlp(
    seed: int,
    d_in: int = 64,
    d_out: int = 32,
    depth: int = 2,
    act_scale: float = 1.0,
    outlier_dim: int = 4,
    outlier_frac: float = 0.04,
    outlier_gain: float = 20.0,
    n_calib: int = 512,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], np.ndarray]:
    """Build a depth-``depth`` ReLU MLP. Returns ``(weights, acts, x_calib)``.

    ``weights[i]`` is ``(d_out_i, d_in_i)``; ``acts[i]`` is the *input* activation of
    layer ``i``. All randomness comes from ``np.random.default_rng(seed)``, so the same
    seed always reproduces the same workload bit for bit.
    """
    if depth <= 0:
        raise DataError("depth must be positive", depth=depth)
    widths = [d_in]
    for _ in range(depth):
        widths.append(d_out if len(widths) == 1 else widths[-1])
    shapes = [(widths[i + 1], widths[i]) for i in range(depth)]

    weights: list[np.ndarray] = []
    acts: list[np.ndarray] = []
    for i, (d_o, d_i) in enumerate(shapes):
        # per-layer seed keeps layers independent yet reproducible
        w, a, x, _ = make_layer(
            seed + 1000 * i,
            d_i,
            d_o,
            n_calib=n_calib,
            act_scale=act_scale,
            outlier_dim=min(outlier_dim, d_i),
            outlier_frac=outlier_frac,
            outlier_gain=outlier_gain,
        )
        weights.append(w)
        acts.append(a)
        if i == 0:
            x_calib = x
    return tuple(weights), tuple(acts), x_calib


class SyntheticMLP(DataSource):
    """A :class:`DataSource` over the synthetic MLP family."""

    def __init__(self, model: str = "m0", depth: int = 2, n_calib: int = 512, **kwargs: object) -> None:
        if model not in MODEL_SHAPES:
            raise DataError("unknown model", model=model, known=sorted(MODEL_SHAPES))
        self.model = model
        self.depth = depth
        self.n_calib = n_calib
        self.kwargs = dict(kwargs)

    def build(self, seed: int) -> LayerBundle:
        (d_out, d_in), *_ = MODEL_SHAPES[self.model]
        weights, acts, x_calib = make_mlp(
            seed,
            d_in=d_in,
            d_out=d_out,
            depth=self.depth,
            n_calib=self.n_calib,
            **self.kwargs,  # type: ignore[arg-type]
        )
        return LayerBundle(
            weights=weights, acts=acts, x_calib=x_calib, name=self.model, per_layer_calib=tuple(acts)
        )

    def describe(self) -> dict[str, object]:
        shapes = MODEL_SHAPES[self.model]
        return {
            "model": self.model,
            "shapes": [list(s) for s in shapes],
            "depth": self.depth,
            "n_calib": self.n_calib,
            "outliers": self.kwargs,
        }
