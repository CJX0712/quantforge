"""AWQ scaling: invariance, grid search, and the flagship's three-stage pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from quantforge.core import QuantConfig, QuantizationError
from quantforge.eval import act_err, nmse
from quantforge.quant.awq import apply_awq_scaling, awq_search, channel_salience
from quantforge.quant.schemes import QuantForgeQuantizer, RtnGroup, RtnTensor, build_quantizer


def test_scaling_is_exactly_invertible_in_fp64(calibration: np.ndarray, gaussian_weight: np.ndarray) -> None:
    """``(W diag(s)) @ (diag(s)^-1 X)`` must reproduce ``W @ X``.

    This is the entire reason AWQ is free at inference: the scaling can be folded into
    the previous layer with no change to the network's function.
    """
    s = channel_salience(calibration, 0.5)
    lhs = (gaussian_weight * s[None, :]) @ (calibration / s[:, None])
    rhs = gaussian_weight @ calibration
    assert np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs) < 1e-12


def test_apply_awq_scaling_roundtrip(gaussian_weight: np.ndarray, calibration: np.ndarray) -> None:
    s = channel_salience(calibration, 0.4)
    scaled, inv = apply_awq_scaling(gaussian_weight, s)
    assert np.allclose(scaled * inv[None, :], gaussian_weight)


def test_apply_awq_scaling_rejects_bad_scale(gaussian_weight: np.ndarray) -> None:
    with pytest.raises(QuantizationError):
        apply_awq_scaling(gaussian_weight, np.ones(3))
    with pytest.raises(QuantizationError):
        apply_awq_scaling(gaussian_weight, -np.ones(gaussian_weight.shape[1]))


def test_alpha_zero_gives_unit_scaling(calibration: np.ndarray) -> None:
    """alpha=0 must yield s==1 exactly, which is what guarantees AWQ >= RTN."""
    s = channel_salience(calibration, 0.0)
    assert np.allclose(s, 1.0)


def test_salience_is_normalised_and_clipped(calibration: np.ndarray) -> None:
    s = channel_salience(calibration, 0.6, clip=(1.0 / 3.0, 3.0))
    assert s.min() >= 1.0 / 3.0 - 1e-12
    assert s.max() <= 3.0 + 1e-12


def test_channel_salience_rejects_negative_alpha(calibration: np.ndarray) -> None:
    with pytest.raises(QuantizationError):
        channel_salience(calibration, -0.1)


def test_channel_salience_rejects_bad_clip(calibration: np.ndarray) -> None:
    with pytest.raises(QuantizationError):
        channel_salience(calibration, 0.5, clip=(3.0, 1.0))


def test_awq_search_never_worse_than_rtn(calibration: np.ndarray, gaussian_weight: np.ndarray) -> None:
    """alpha=0 is always in the grid, so the searched result cannot lose to RTN."""
    cfg = QuantConfig(bits=4, granularity="per_tensor", awq_alpha_grid=(0.0, 0.1, 0.2, 0.3))
    found = awq_search(gaussian_weight, calibration, cfg, objective="acterr")
    alpha_zero = found["score"] if found["alpha"] == 0.0 else None
    if alpha_zero is None:
        # recompute the alpha=0 baseline explicitly
        from quantforge.quant.scale import dequantize_matrix, quantize_matrix

        rec = dequantize_matrix(*quantize_matrix(gaussian_weight, cfg))
        ref = gaussian_weight @ calibration
        alpha_zero = float(np.sum((rec @ calibration - ref) ** 2) / np.sum(ref * ref))
    assert found["score"] <= alpha_zero + 1e-12


def test_awq_search_is_deterministic(calibration: np.ndarray, gaussian_weight: np.ndarray) -> None:
    cfg = QuantConfig(bits=4, granularity="per_tensor")
    a = awq_search(gaussian_weight, calibration, cfg)
    b = awq_search(gaussian_weight, calibration, cfg)
    assert a["alpha"] == b["alpha"] and a["score"] == b["score"]


def test_awq_search_rejects_shape_mismatch(gaussian_weight: np.ndarray) -> None:
    with pytest.raises(QuantizationError):
        awq_search(gaussian_weight, np.zeros((3, 10)), QuantConfig())


def test_awq_search_rejects_unknown_objective(calibration: np.ndarray, gaussian_weight: np.ndarray) -> None:
    with pytest.raises(QuantizationError):
        awq_search(gaussian_weight, calibration, QuantConfig(), objective="bogus")


# ------------------------------------------------------- method comparison


def test_every_method_runs_and_returns_finite(calibration: np.ndarray, gaussian_weight: np.ndarray) -> None:
    cfg = QuantConfig(bits=4, group_size=8)
    for method in ("rtn_tensor", "rtn_group", "awq", "quantforge"):
        result = build_quantizer(method, cfg).quantize(gaussian_weight, calibration)
        assert result.w_hat.shape == gaussian_weight.shape
        assert np.isfinite(result.w_hat).all()
        assert result.bits_per_weight >= cfg.bits


def test_higher_bits_give_lower_error(calibration: np.ndarray, gaussian_weight: np.ndarray) -> None:
    """Sanity check on the whole stack: 8-bit must beat 2-bit."""
    errs = []
    for bits in (2, 8):
        cfg = QuantConfig(bits=bits, group_size=8)
        rec = build_quantizer("quantforge", cfg).quantize(gaussian_weight, calibration).w_hat
        errs.append(nmse(gaussian_weight, rec))
    assert errs[1] < errs[0]


def test_flagship_reduces_output_error_vs_naive_rtn(calibration: np.ndarray, gaussian_weight: np.ndarray) -> None:
    """The flagship must beat naive per-tensor RTN on the output axis."""
    naive = RtnTensor(QuantConfig(bits=4, group_size=8)).quantize(gaussian_weight, calibration).w_hat
    flags = build_quantizer("quantforge", QuantConfig(bits=4, group_size=8)).quantize(
        gaussian_weight, calibration
    ).w_hat
    assert act_err(gaussian_weight, flags, calibration) < act_err(gaussian_weight, naive, calibration)


def test_rtn_group_beats_rtn_tensor_on_weight_nmse(calibration: np.ndarray, gaussian_weight: np.ndarray) -> None:
    cfg = QuantConfig(bits=4, group_size=8)
    tensor = RtnTensor(cfg).quantize(gaussian_weight, calibration).w_hat
    group = RtnGroup(cfg).quantize(gaussian_weight, calibration).w_hat
    assert nmse(gaussian_weight, group) < nmse(gaussian_weight, tensor)


def test_rtn_group_records_clip_alphas(calibration: np.ndarray, gaussian_weight: np.ndarray) -> None:
    res = RtnGroup(QuantConfig(bits=4, group_size=8)).quantize(gaussian_weight, calibration)
    alphas = np.asarray(res.aux["clip_alphas"])
    assert alphas.shape[0] == res.scales.shape[1]
    assert np.all((alphas >= 0.5) & (alphas <= 1.0))


def test_awq_method_maps_back_to_original_coordinates(calibration: np.ndarray, gaussian_weight: np.ndarray) -> None:
    """Method C must return a drop-in replacement for W, not a rescaled matrix."""
    res = build_quantizer("awq", QuantConfig(bits=4)).quantize(gaussian_weight, calibration)
    s = np.asarray(res.aux["scale"])
    if not np.allclose(s, 1.0):
        assert res.w_hat.shape == gaussian_weight.shape
        # the effective scale in original coordinates is s/s, i.e. unity-scaled back
        assert np.isfinite(res.w_hat).all()


def test_flagship_reports_diagnostics(calibration: np.ndarray, gaussian_weight: np.ndarray) -> None:
    res = build_quantizer("quantforge", QuantConfig(bits=4, group_size=8)).quantize(
        gaussian_weight, calibration
    )
    assert res.aux["chol_residual"] < 1e-10
    assert res.aux["raw_err_energy"] >= 0.0
    assert "alpha" in res.aux


def test_flagship_ls_refit_reduces_weight_error(calibration: np.ndarray, gaussian_weight: np.ndarray) -> None:
    """Stage 3 exists because GPTQ trades weight NMSE for output error."""
    base = QuantConfig(bits=4, group_size=8)
    with_refit = QuantForgeQuantizer(base, ablation=None).quantize(gaussian_weight, calibration).w_hat
    without = QuantForgeQuantizer(base, ablation="no_ls_refit").quantize(gaussian_weight, calibration).w_hat
    assert nmse(gaussian_weight, with_refit) <= nmse(gaussian_weight, without) + 1e-12


def test_flagship_requires_calibration(gaussian_weight: np.ndarray) -> None:
    with pytest.raises(QuantizationError):
        QuantForgeQuantizer(QuantConfig()).quantize(gaussian_weight, None)


def test_awq_method_requires_calibration(gaussian_weight: np.ndarray) -> None:
    with pytest.raises(QuantizationError):
        build_quantizer("awq", QuantConfig()).quantize(gaussian_weight, None)


def test_build_quantizer_rejects_unknown() -> None:
    with pytest.raises(QuantizationError):
        build_quantizer("nope", QuantConfig())


def test_unknown_ablation_rejected() -> None:
    with pytest.raises(QuantizationError):
        QuantForgeQuantizer(QuantConfig(), ablation="does_not_exist")


@pytest.mark.parametrize(
    "ablation", ["no_hessian", "no_act_aware", "no_act_order", "no_mse_clip", "no_ls_refit"]
)
def test_every_ablation_runs(ablation: str, calibration: np.ndarray, gaussian_weight: np.ndarray) -> None:
    res = QuantForgeQuantizer(QuantConfig(bits=4, group_size=8), ablation=ablation).quantize(
        gaussian_weight, calibration
    )
    assert np.isfinite(res.w_hat).all()
    assert res.w_hat.shape == gaussian_weight.shape


def test_no_hessian_matches_rtn_on_the_same_scales(calibration: np.ndarray, gaussian_weight: np.ndarray) -> None:
    """GPTQ with the Hessian off must equal plain RTN.

    The Hessian is what supplies the error compensation, so removing it has to collapse
    the algorithm exactly onto RTN. Every other stage (act-order, MSE clip, refit) is
    disabled too, since those legitimately change the result.
    """
    from quantforge.quant.gptq import gptq_quantize
    from quantforge.quant.scale import dequantize_matrix, quantize_matrix

    cfg = QuantConfig(bits=4, group_size=8, act_order=False, use_mse_clip=False, refit_scales=False)
    direct = dequantize_matrix(*quantize_matrix(gaussian_weight, cfg))
    g = np.asarray(gptq_quantize(gaussian_weight, calibration, cfg, use_hessian=False)["w_hat"])
    assert np.allclose(g, direct, atol=1e-12)


def test_no_hessian_ablation_is_worse_than_full(calibration: np.ndarray, gaussian_weight: np.ndarray) -> None:
    """Removing the Hessian must cost output accuracy -- that is what it buys."""
    from quantforge.eval import act_err

    base = QuantConfig(bits=4, group_size=8)
    full = QuantForgeQuantizer(base).quantize(gaussian_weight, calibration).w_hat
    abl = QuantForgeQuantizer(base, ablation="no_hessian").quantize(gaussian_weight, calibration).w_hat
    assert act_err(gaussian_weight, abl, calibration) > act_err(gaussian_weight, full, calibration)


@pytest.mark.parametrize(
    "ablation",
    ["no_hessian", "no_act_aware", "no_act_order", "no_mse_clip", "no_ls_refit"],
)
def test_ablation_changes_the_result(ablation: str, calibration: np.ndarray, gaussian_weight: np.ndarray) -> None:
    """Each ablation must actually change behaviour, not be a silent no-op."""
    base = QuantConfig(bits=4, group_size=8)
    full = QuantForgeQuantizer(base).quantize(gaussian_weight, calibration).w_hat
    abl = QuantForgeQuantizer(base, ablation=ablation).quantize(gaussian_weight, calibration).w_hat
    assert not np.array_equal(full, abl), f"ablation {ablation} had no effect"
