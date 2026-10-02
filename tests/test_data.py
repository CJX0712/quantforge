"""Data layer: reproducibility, outlier injection, model shapes."""

from __future__ import annotations

import numpy as np
import pytest
from quantforge.core import DataError
from quantforge.data import MODEL_SHAPES, SyntheticConfig, SyntheticMLP, make_layer, make_mlp


def test_make_mlp_is_deterministic() -> None:
    a = make_mlp(7, d_in=64, d_out=32, depth=2, n_calib=64)
    b = make_mlp(7, d_in=64, d_out=32, depth=2, n_calib=64)
    for wa, wb in zip(a[0], b[0], strict=True):
        assert np.array_equal(wa, wb)
    assert np.array_equal(a[2], b[2])


def test_make_mlp_seed_changes_workload() -> None:
    a = make_mlp(7, d_in=32, d_out=16, depth=1, n_calib=32)
    b = make_mlp(8, d_in=32, d_out=16, depth=1, n_calib=32)
    assert not np.array_equal(a[0][0], b[0][0])


def test_make_mlp_shapes_follow_depth() -> None:
    weights, acts, x = make_mlp(3, d_in=64, d_out=32, depth=3, n_calib=32)
    assert len(weights) == 3
    assert weights[0].shape == (32, 64)
    assert weights[1].shape == (32, 32)
    assert weights[2].shape == (32, 32)
    assert x.shape == (64, 32)
    assert acts[0].shape == (64, 32)


def test_activations_are_post_relu() -> None:
    _, acts, x = make_mlp(1, d_in=32, d_out=16, depth=1, n_calib=64)
    assert np.all(acts[0] >= 0.0)
    assert np.all(x >= 0.0)


def test_outlier_injection_creates_sparse_large_channels() -> None:
    _, _, acts, mask = make_layer(
        seed=5, d_in=64, d_out=32, n_calib=512, outlier_dim=4, outlier_frac=0.04, outlier_gain=20.0
    )
    assert mask.any()
    n_hot_channels = int(mask.any(axis=1).sum())
    assert n_hot_channels == 4
    # only a small fraction of samples carry the large values
    hot_fraction = float(mask[mask.any(axis=1)].mean())
    assert 0.0 < hot_fraction < 0.2
    plain = acts[~mask.any(axis=1)]
    hot = acts[mask.any(axis=1)]
    assert hot.max() > 10.0 * max(plain.max(), 1e-9)


def test_no_outliers_when_disabled() -> None:
    _, _, _, mask = make_layer(seed=5, d_in=32, d_out=16, n_calib=64, outlier_dim=0)
    assert not mask.any()


def test_weights_are_heavy_tailed() -> None:
    w, _, _, _ = make_layer(seed=2, d_in=128, d_out=64, n_calib=32)
    # population kurtosis (ddof=0), Gaussian == 3.0
    kurtosis = float(np.mean(w**4) / np.mean(w**2) ** 2)
    assert kurtosis > 4.0  # clearly heavier than Gaussian


@pytest.mark.parametrize("model", sorted(MODEL_SHAPES))
def test_synthetic_mlp_bundle(model: str) -> None:
    bundle = SyntheticMLP(model=model, n_calib=64).build(7)
    expected = MODEL_SHAPES[model]
    for i, (d_out, d_in) in enumerate(expected):
        w, x = bundle.layer(i)
        assert w.shape == (d_out, d_in)
        assert x.shape[0] == d_in
    assert bundle.n_layers == len(expected)


def test_synthetic_mlp_rejects_unknown_model() -> None:
    with pytest.raises(DataError):
        SyntheticMLP(model="nope")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d_in": 0},
        {"d_out": -1},
        {"depth": 0},
        {"n_calib": 0},
        {"outlier_frac": 1.5},
        {"outlier_dim": 999},
        {"outlier_gain": 0.5},
    ],
)
def test_synthetic_config_rejects_invalid(kwargs: dict) -> None:
    with pytest.raises(DataError):
        SyntheticConfig(**kwargs)


def test_outlier_dim_clamped_to_width() -> None:
    """A model narrower than the requested outlier count must still build."""
    weights, _, x = make_mlp(1, d_in=16, d_out=8, depth=2, n_calib=32, outlier_dim=64)
    assert weights[1].shape[1] == 8
    assert x.shape[0] == 16


def test_describe_is_serialisable() -> None:
    import json

    desc = SyntheticMLP("m0", n_calib=32).describe()
    json.dumps(desc)  # must not raise
    assert desc["model"] == "m0"
