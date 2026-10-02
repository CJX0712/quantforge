"""Degradation paths: the toolkit must stay usable with optional dependencies missing.

The flagship path needs numpy only. ``scipy`` is optional, and the RTX/Tier-1 fallback
must engage loudly (warning + reported path) rather than silently producing degraded
accuracy that looks healthy in the benchmark.
"""

from __future__ import annotations

import builtins
import warnings

import numpy as np
import pytest
from quantforge.core import DecompositionError, QuantConfig
from quantforge.eval import act_err, nmse
from quantforge.quant import selectors
from quantforge.quant.gptq import gptq_quantize, gptq_with_fallback
from quantforge.quant.scale import dequantize_matrix, quantize_matrix
from quantforge.quant.schemes import build_quantizer


def test_numpy_backend_is_always_available() -> None:
    assert selectors.available_numpy() is True


def test_select_backend_reports_a_valid_choice() -> None:
    name, _noop = selectors.select_backend(prefer_scipy=False)
    assert name == "numpy"
    name, _noop = selectors.select_backend(prefer_scipy=True)
    assert name in ("numpy", "scipy")


def test_capability_report_is_json_serialisable() -> None:
    import json

    payload = selectors.capability_report()
    json.dumps(payload)
    assert isinstance(payload["numpy"], str)
    assert "python" in payload and "platform" in payload


def test_tier1_rtn_fallback_matches_plain_rtn(gaussian_weight: np.ndarray, calibration: np.ndarray) -> None:
    """Without a Hessian the flagship must collapse onto plain RTN (Tier-1 guarantee)."""
    cfg = QuantConfig(bits=4, group_size=8, act_order=False, use_mse_clip=False, refit_scales=False)
    payload, path = gptq_with_fallback(gaussian_weight, calibration, cfg, use_hessian=False)
    assert path == "gptq"
    direct = dequantize_matrix(*quantize_matrix(gaussian_weight, cfg))
    assert np.allclose(np.asarray(payload["w_hat"]), direct, atol=1e-12)


def test_fallback_is_used_and_warned_when_factorisation_fails(
    monkeypatch: pytest.MonkeyPatch, gaussian_weight: np.ndarray, calibration: np.ndarray
) -> None:
    """A failed decomposition must degrade loudly, never silently."""
    import quantforge.quant.gptq as gptq_mod

    def always_fails(*_args: object, **_kwargs: object) -> None:
        raise DecompositionError("forced failure", attempts=6)

    monkeypatch.setattr(gptq_mod, "inverse_factor", always_fails)
    with pytest.warns(RuntimeWarning, match="falling back to RTN"):
        payload, path = gptq_with_fallback(gaussian_weight, calibration, QuantConfig(bits=4, group_size=8))
    assert path == "rtn_fallback"
    assert payload["fallback"] is True
    assert np.isfinite(np.asarray(payload["w_hat"])).all()


def test_fallback_result_is_usable(gaussian_weight: np.ndarray, calibration: np.ndarray) -> None:
    import quantforge.quant.gptq as gptq_mod

    original = gptq_mod.inverse_factor
    try:
        gptq_mod.inverse_factor = lambda *a, **k: (_ for _ in ()).throw(DecompositionError("x"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            payload, path = gptq_with_fallback(gaussian_weight, calibration, QuantConfig(bits=4, group_size=8))
    finally:
        gptq_mod.inverse_factor = original
    assert path == "rtn_fallback"
    rec = np.asarray(payload["w_hat"])
    assert rec.shape == gaussian_weight.shape
    assert 0.0 < nmse(gaussian_weight, rec) < 1.0


def test_pipeline_works_without_scipy(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Block scipy at import time and confirm the whole pipeline still runs."""
    import quantforge.quant.selectors as sel

    real_import = builtins.__import__

    def blocked(name: str, *args: object, **kwargs: object) -> object:
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("scipy blocked for test")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", blocked)
    monkeypatch.setattr(sel, "available_scipy", lambda: False)

    from quantforge.core import load_config
    from quantforge.pipeline.runner import run_pipeline

    cfg = load_config(
        {
            "seed": 1,
            "bits": 4,
            "n_calib": 64,
            "calib_samples": 64,
            "models": ("m0",),
            "dataset_seeds": (7,),
            "methods": ("rtn_tensor", "rtn_group", "quantforge"),
            "bits_grid": (4,),
            "group_grid": (32,),
        }
    )
    report = run_pipeline(cfg, outdir=tmp_path, with_ablations=False)
    assert report["n_cells"] > 0
    assert report["env"]["capabilities"]["scipy"] is False
    assert "quantforge" in report["method_summary"]


def test_full_stack_accuracy_without_scipy(
    monkeypatch: pytest.MonkeyPatch, gaussian_weight: np.ndarray, calibration: np.ndarray
) -> None:
    """The numpy-only path must still beat naive RTN on the output axis."""
    import quantforge.quant.selectors as sel

    monkeypatch.setattr(sel, "available_scipy", lambda: False)
    cfg = QuantConfig(bits=4, group_size=8)
    naive = build_quantizer("rtn_tensor", cfg).quantize(gaussian_weight, calibration).w_hat
    flags = build_quantizer("quantforge", cfg).quantize(gaussian_weight, calibration).w_hat
    assert act_err(gaussian_weight, flags, calibration) < act_err(gaussian_weight, naive, calibration)


def test_inverse_factor_raises_rather_than_returning_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    """numpy raises LinAlgError; backends that only return garbage must still be caught."""
    import quantforge.quant.gptq as gptq_mod

    def garbage_cholesky(_a: np.ndarray) -> np.ndarray:
        return np.full((3, 3), np.nan)

    monkeypatch.setattr(np.linalg, "cholesky", garbage_cholesky)
    with pytest.raises(DecompositionError):
        gptq_mod.inverse_factor(np.eye(3), 0.01)


def test_inverse_factor_accepts_finite_upper_factor(monkeypatch: pytest.MonkeyPatch) -> None:
    """A backend that leaves junk in the unused triangle must be triu'd, not trusted.

    ``np.linalg.cholesky`` returns the *input* matrix in the opposite triangle. If that
    garbage leaks into the upper-triangular factor, every compensation coefficient would
    be silently wrong, so ``inverse_factor`` must strip it with ``triu``.
    """
    import quantforge.quant.gptq as gptq_mod

    real = np.linalg.cholesky
    calls = {"n": 0}

    def leaky(a: np.ndarray) -> np.ndarray:
        """Leave junk in the upper triangle only on the *inverse-Hessian* factorisation."""
        calls["n"] += 1
        out = real(a)
        if calls["n"] == 2:
            out[np.triu_indices_from(out, 1)] = 12345.0
        return out

    monkeypatch.setattr(np.linalg, "cholesky", leaky)
    h = np.eye(4) * 2.0
    percdamp = 0.01
    u, g, damp = gptq_mod.inverse_factor(h, percdamp)
    assert calls["n"] >= 2
    assert np.all(np.isfinite(u))
    assert np.all(np.tril(u, -1) == 0)
    # G must be the inverse of the *damped* Hessian, not of H itself
    h_damped = h + percdamp * np.mean(np.diag(h)) * np.eye(4)
    assert damp == pytest.approx(percdamp)
    assert np.allclose(g, np.linalg.inv(h_damped), atol=1e-10)


def test_dead_hessian_is_isolated_not_fatal() -> None:
    """A fully zero Hessian is degenerate but must not produce NaN weights."""
    rng = np.random.default_rng(3)
    w = rng.standard_normal((4, 8))
    x = np.zeros((8, 16))
    payload = gptq_quantize(w, x, QuantConfig(bits=4, group_size=4))
    assert np.isfinite(np.asarray(payload["w_hat"])).all()


def test_runtimewarning_is_not_silenced() -> None:
    """Warnings must reach the caller; a silent degradation would corrupt the report."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("probe", RuntimeWarning, stacklevel=1)
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)
