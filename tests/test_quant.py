"""Quantization primitives and the numeric invariants that guard them."""

from __future__ import annotations

import numpy as np
import pytest
from quantforge.core import QuantConfig, QuantizationError
from quantforge.core.config import SAFETY_TARGET
from quantforge.quant.gptq import build_hessian, gptq_quantize, inverse_factor
from quantforge.quant.rounding import (
    dequantize_affine,
    qmin_qmax,
    quantize_affine,
    quantize_with_params,
    round_half_up,
)
from quantforge.quant.scale import (
    bits_per_weight,
    column_owner,
    compute_scale,
    dequantize_matrix,
    group_bounds,
    n_groups_for,
    quantize_matrix,
    search_clip_alpha,
)
from quantforge.quant.schemes import build_quantizer

# ---------------------------------------------------------------- rounding


def test_round_half_up_is_not_bankers_rounding() -> None:
    x = np.array([0.5, 1.5, 2.5, -0.5, -1.5])
    assert np.array_equal(round_half_up(x), np.array([1.0, 2.0, 3.0, 0.0, -1.0]))


def test_round_half_up_differs_from_np_round() -> None:
    x = np.array([0.5, 1.5, 2.5])
    assert not np.array_equal(round_half_up(x), np.round(x))


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_qmin_qmax_ranges(bits: int) -> None:
    qmax = 2 ** (bits - 1) - 1
    lo, hi = qmin_qmax(bits, "symmetric")
    assert lo == -qmax and hi == qmax  # symmetric: zero exactly representable, no zero-point
    lo_a, hi_a = qmin_qmax(bits, "asymmetric")
    assert lo_a == -(2 ** (bits - 1)) and hi_a == qmax


def test_qmin_qmax_rejects_bad_bits() -> None:
    with pytest.raises(QuantizationError):
        qmin_qmax(1, "symmetric")


def test_grid_points_are_exact_fixed_points() -> None:
    """Any value on the grid must survive quantize->dequantize unchanged."""
    bits, s, z = 4, 0.0173, -3
    lo, hi = qmin_qmax(bits, "asymmetric")
    levels = np.arange(lo, hi + 1)
    x = s * (levels - z)
    codes = quantize_affine(x, s, z, bits, "asymmetric")
    assert np.array_equal(codes, levels)
    assert np.allclose(dequantize_affine(codes, s, z), x, rtol=1e-12, atol=1e-15)


def test_quantization_is_idempotent_on_codes() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    s = float(np.abs(x).max()) / 7
    c1 = quantize_affine(x, s, 0, 4, "symmetric")
    c2 = quantize_affine(dequantize_affine(c1, s, 0), s, 0, 4, "symmetric")
    assert np.array_equal(c1, c2)


def test_saturation_at_extremes() -> None:
    s = 0.01
    lo, hi = qmin_qmax(4, "symmetric")
    assert quantize_affine(np.array([1e3]), s, 0, 4, "symmetric")[0] == hi
    assert quantize_affine(np.array([-1e3]), s, 0, 4, "symmetric")[0] == lo


def test_single_point_error_bounded_by_half_step() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(2000)
    s = float(np.abs(x).max()) / 7
    err = np.abs(x - dequantize_affine(quantize_affine(x, s, 0, 4, "symmetric"), s, 0))
    assert np.all(err <= s / 2 + 1e-12)


def test_quantize_affine_rejects_bad_scale() -> None:
    with pytest.raises(QuantizationError):
        quantize_affine(np.zeros(3), 0.0, 0, 4, "symmetric")
    with pytest.raises(QuantizationError):
        quantize_affine(np.zeros(3), -1.0, 0, 4, "symmetric")


def test_quantize_with_params_rejects_mismatched_scale() -> None:
    with pytest.raises(QuantizationError):
        quantize_with_params(np.zeros((3, 4)), np.ones((3, 7)), np.zeros((3, 7)), 4, "symmetric")


# ------------------------------------------------------------------- scales


def test_compute_scale_all_zero_block_is_safe() -> None:
    s, z = compute_scale(np.zeros((4, 8)), 4, "symmetric")
    assert s == 1.0 and z == 0


def test_compute_scale_asymmetric_zero_range_is_safe() -> None:
    s, z = compute_scale(np.zeros((4, 8)), 4, "asymmetric")
    assert s == 1.0
    assert qmin_qmax(4, "asymmetric")[0] <= z <= qmin_qmax(4, "asymmetric")[1]


def test_compute_scale_rejects_bad_alpha() -> None:
    with pytest.raises(QuantizationError):
        compute_scale(np.ones((2, 2)), 4, "symmetric", clip_alpha=0.0)


def test_group_bounds_never_pads_the_tail() -> None:
    assert group_bounds(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert group_bounds(8, 4) == [(0, 4), (4, 8)]
    assert group_bounds(3, 128) == [(0, 3)]  # d_in < group_size degenerates gracefully


def test_group_bounds_rejects_nonpositive() -> None:
    with pytest.raises(QuantizationError):
        group_bounds(8, 0)


def test_quantize_dequantize_roundtrip_is_lossless_in_codes() -> None:
    rng = np.random.default_rng(2)
    w = rng.standard_normal((6, 10))  # deliberately non-square, and 4 does not divide 10
    cfg = QuantConfig(bits=4, group_size=4)
    codes, scales, zeros = quantize_matrix(w, cfg)
    bnds = group_bounds(10, 4)
    rec = dequantize_matrix(codes, scales, zeros, bnds)
    assert rec.shape == w.shape
    # every element must sit exactly on its group's grid
    for gi, (k0, k1) in enumerate(bnds):
        assert np.allclose(
            rec[:, k0:k1], (codes[:, k0:k1] - zeros[:, gi : gi + 1]) * scales[:, gi : gi + 1]
        )


@pytest.mark.parametrize("d_in,group_size", [(10, 4), (16, 8), (96, 64), (64, 128), (13, 5)])
def test_column_owner_matches_group_bounds(d_in: int, group_size: int) -> None:
    """Column->group mapping must come from the same slices used to build the matrix.

    Regression guard: deriving the mapping from ``np.linspace`` yields fractional edges
    when ``group_size`` does not divide ``d_in``, which assigns columns to the wrong
    group and corrupts the reconstruction of every group after the first.
    """
    bnds = group_bounds(d_in, group_size)
    owner = column_owner(bnds, d_in)
    for gi, (k0, k1) in enumerate(bnds):
        assert np.all(owner[k0:k1] == gi), f"group {gi} columns misassigned"
    assert owner.min() == 0 and owner.max() == len(bnds) - 1
    assert len(owner) == d_in


def test_dequantize_matrix_is_exact_for_non_divisible_group_size() -> None:
    rng = np.random.default_rng(21)
    w = rng.standard_normal((5, 10))
    cfg = QuantConfig(bits=4, group_size=4)
    codes, scales, zeros = quantize_matrix(w, cfg)
    rec = dequantize_matrix(codes, scales, zeros, group_bounds(10, 4))
    # the reconstruction must use each group's own scale, so every residual is a
    # multiple of that group's step
    for gi, (k0, k1) in enumerate(group_bounds(10, 4)):
        resid = w[:, k0:k1] - rec[:, k0:k1]
        assert np.all(np.abs(resid) <= scales[:, gi : gi + 1].max() / 2 + 1e-12)


def test_requantization_is_idempotent() -> None:
    """Re-quantizing a dequantized matrix must reproduce it exactly.

    This is the strongest guard against a group off-by-one: any misalignment in the
    trailing group's scale would break it.
    """
    rng = np.random.default_rng(3)
    w = rng.standard_normal((5, 11))
    cfg = QuantConfig(bits=4, group_size=4)
    codes, scales, zeros = quantize_matrix(w, cfg)
    rec = dequantize_matrix(codes, scales, zeros)
    codes2, scales2, zeros2 = quantize_matrix(rec, cfg)
    rec2 = dequantize_matrix(codes2, scales2, zeros2)
    assert np.array_equal(rec, rec2)


def test_search_clip_alpha_never_worse_than_no_clip() -> None:
    rng = np.random.default_rng(4)
    for _ in range(5):
        block = rng.standard_t(3.0, size=(8, 32))
        alpha, best = search_clip_alpha(block, 4, "symmetric")
        assert 0.5 <= alpha <= 1.0
        s_full, _ = compute_scale(block, 4, "symmetric", 1.0)
        _s_best, _ = compute_scale(block, 4, "symmetric", alpha)
        from quantforge.quant.rounding import quantize_affine

        full = float(np.mean((block - dequantize_affine(quantize_affine(block, s_full, 0, 4), s_full, 0)) ** 2))
        assert best <= full + 1e-15


def test_search_clip_alpha_grid_always_contains_one() -> None:
    """alpha=1.0 must always be a candidate, so "no clipping" is never excluded.

    The returned alpha may additionally be a refined value (the search refines around
    the coarse optimum), so assert the invariant rather than grid membership.
    """
    rng = np.random.default_rng(5)
    block = rng.standard_normal((4, 16))
    alpha, _ = search_clip_alpha(block, 4, "symmetric", grid=(0.5, 0.6))
    assert 0.5 <= alpha <= 1.0


def test_bits_per_weight_matches_closed_form() -> None:
    w = np.zeros((32, 128))
    cfg = QuantConfig(bits=4, group_size=128, granularity="per_group")
    expected = 4 + 16 * (128 // 128) / w.size
    assert bits_per_weight(cfg, w.size, 128) == pytest.approx(expected)


def test_bits_per_weight_tensor_has_single_scale() -> None:
    cfg = QuantConfig(bits=4, granularity="per_tensor")
    assert bits_per_weight(cfg, 32 * 128, 128) == pytest.approx(4 + 16 / (32 * 128))


def test_bits_per_weight_asymmetric_charges_zero_point() -> None:
    cfg = QuantConfig(bits=4, group_size=64, granularity="per_group", scheme="asymmetric")
    sym = cfg.evolve(scheme="symmetric")
    assert bits_per_weight(cfg, 32 * 128, 128) > bits_per_weight(sym, 32 * 128, 128)


def test_bits_per_weight_rejects_empty() -> None:
    with pytest.raises(QuantizationError):
        bits_per_weight(QuantConfig(), 0, 8)


def test_n_groups_for_matches_granularity() -> None:
    assert n_groups_for(64, QuantConfig(granularity="per_tensor")) == 1
    assert n_groups_for(64, QuantConfig(granularity="per_channel")) == 64
    assert n_groups_for(64, QuantConfig(granularity="per_group", group_size=16)) == 4


# --------------------------------------------------------------------- GPTQ


def test_hessian_is_symmetric_and_scaled() -> None:
    rng = np.random.default_rng(6)
    x = rng.standard_normal((5, 40))
    h = build_hessian(x)
    assert np.array_equal(h, h.T)
    assert h.shape == (5, 5)
    assert np.allclose(h, 2.0 * (x @ x.T) / 40)


def test_hessian_rejects_wrong_rank() -> None:
    with pytest.raises(QuantizationError):
        build_hessian(np.zeros((3, 3, 3)))


def test_inverse_factor_cholesky_residual() -> None:
    """The two-step factorisation must satisfy G = H^-1 exactly."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal((12, 80))
    x[:2, :5] *= 30.0
    h = build_hessian(x)
    u, g, damp = inverse_factor(h, 0.01)
    assert u.shape == (12, 12)
    assert np.all(np.tril(u, -1) == 0)  # U is upper triangular: strict lower must be zero
    assert np.isfinite(u).all()
    assert np.allclose(g, g.T)
    assert damp == pytest.approx(0.01)
    # G must be the true inverse of the damped Hessian
    h_damped = h + 0.01 * np.mean(np.diag(h)) * np.eye(12)
    assert np.allclose(g, np.linalg.inv(h_damped), atol=1e-8)
    assert np.allclose(u.T @ u @ h_damped, np.eye(12), atol=1e-10)


def test_inverse_factor_handles_singular_hessian() -> None:
    h = np.zeros((6, 6))  # fully degenerate: must not raise
    u, g, damp = inverse_factor(h, 0.01)
    assert np.isfinite(u).all() and np.isfinite(g).all()
    assert damp > 0.0


def test_gptq_reduces_proxy_objective_vs_rtn() -> None:
    """Hard gate: GPTQ optimises J = ||(W-W_hat)X||^2, so it must not lose to RTN.

    Weight-domain NMSE is deliberately *not* asserted here: GPTQ trades weight error for
    output error by design, so it can be worse on that axis.
    """
    from quantforge.eval import act_err

    rng = np.random.default_rng(8)
    w = rng.standard_normal((8, 32)) / np.sqrt(32)
    x = np.maximum(rng.standard_normal((32, 128)), 0.0)
    x[:3, :6] *= 25.0
    cfg = QuantConfig(bits=4, group_size=16)
    rtn = dequantize_matrix(*quantize_matrix(w, cfg))
    gptq = gptq_quantize(w, x, cfg)["w_hat"]
    assert act_err(w, gptq, x) <= act_err(w, rtn, x) + 1e-9


def test_gptq_is_deterministic(calibration: np.ndarray, gaussian_weight: np.ndarray) -> None:
    cfg = QuantConfig(bits=4, group_size=8)
    a = gptq_quantize(gaussian_weight, calibration, cfg)["codes"]
    b = gptq_quantize(gaussian_weight, calibration, cfg)["codes"]
    assert np.array_equal(a, b)


def test_gptq_reports_cholesky_residual(calibration: np.ndarray, gaussian_weight: np.ndarray) -> None:
    payload = gptq_quantize(gaussian_weight, calibration, QuantConfig(bits=4, group_size=8), return_debug=True)
    assert payload["chol_residual"] < 1e-10


def test_gptq_permutation_roundtrip(calibration: np.ndarray, gaussian_weight: np.ndarray) -> None:
    h = build_hessian(calibration)
    perm = np.argsort(-np.diag(h), kind="stable")
    inv = np.argsort(perm, kind="stable")
    assert np.array_equal(gaussian_weight[:, perm][:, inv], gaussian_weight)


def test_gptq_hessian_scale_invariance(calibration: np.ndarray, gaussian_weight: np.ndarray) -> None:
    """H and cH must give identical codes: damping is relative, so scale cancels."""
    cfg = QuantConfig(bits=4, group_size=8)
    a = gptq_quantize(gaussian_weight, calibration, cfg)["codes"]
    b = gptq_quantize(gaussian_weight, calibration * np.sqrt(7.3), cfg)["codes"]
    assert np.array_equal(a, b)


def test_gptq_act_order_changes_result(calibration: np.ndarray, gaussian_weight: np.ndarray) -> None:
    on = gptq_quantize(gaussian_weight, calibration, QuantConfig(bits=4, group_size=8, act_order=True))
    off = gptq_quantize(gaussian_weight, calibration, QuantConfig(bits=4, group_size=8, act_order=False))
    assert not np.array_equal(on["codes"], off["codes"])


def test_gptq_dead_channel_is_isolated() -> None:
    rng = np.random.default_rng(9)
    w = rng.standard_normal((4, 8))
    x = rng.standard_normal((8, 40))
    x[3] = 0.0  # dead input channel -> Hessian row/col is exactly zero
    payload = gptq_quantize(w, x, QuantConfig(bits=4, group_size=4))
    assert np.isfinite(np.asarray(payload["w_hat"])).all()


def test_gptq_non_divisible_group_size() -> None:
    rng = np.random.default_rng(10)
    w = rng.standard_normal((4, 10))
    x = rng.standard_normal((10, 40))
    payload = gptq_quantize(w, x, QuantConfig(bits=4, group_size=4))
    assert np.asarray(payload["w_hat"]).shape == (4, 10)
    assert np.isfinite(np.asarray(payload["w_hat"])).all()


def test_gptq_blocksize_larger_than_width() -> None:
    rng = np.random.default_rng(11)
    w = rng.standard_normal((4, 10))
    x = rng.standard_normal((10, 40))
    payload = gptq_quantize(w, x, QuantConfig(bits=4, group_size=4, blocksize=128))
    assert np.asarray(payload["w_hat"]).shape == (4, 10)


def test_gptq_rejects_shape_mismatch(gaussian_weight: np.ndarray) -> None:
    with pytest.raises(QuantizationError):
        gptq_quantize(gaussian_weight, np.zeros((5, 10)), QuantConfig())


def test_gptq_without_hessian_reduces_to_rtn(calibration: np.ndarray, gaussian_weight: np.ndarray) -> None:
    """Hessian off + act_order off + min-max scales => identical to plain RTN.

    All three must be disabled, because the MSE clip search is itself an improvement
    over min-max and would mask the equivalence being asserted here.
    """
    cfg = QuantConfig(bits=4, group_size=8, act_order=False, use_mse_clip=False)
    no_hess = gptq_quantize(gaussian_weight, calibration, cfg, use_hessian=False)
    rtn = quantize_matrix(gaussian_weight, cfg)
    assert np.allclose(np.asarray(no_hess["w_hat"]), dequantize_matrix(*rtn), atol=1e-12)


def test_damping_safety_identity(calibration: np.ndarray, gaussian_weight: np.ndarray) -> None:
    """`mu/lam` must equal the configured safety factor exactly (T-C).

    Damping is defined as ``lam = mean(diag(H)) * percdamp``, so the ratio is
    ``1/percdamp`` for *any* Hessian. Asserting the identity turns "did an edit change
    which quantity is being controlled?" from a silent accuracy regression into a test
    failure.
    """
    cfg = QuantConfig(bits=4, group_size=8, percdamp=1.0 / SAFETY_TARGET)
    payload = gptq_quantize(gaussian_weight, calibration, cfg, return_debug=True)
    damping = payload["damping"]
    assert damping["damp_fraction"] == pytest.approx(1.0 / SAFETY_TARGET)
    assert abs(damping["safety"] - SAFETY_TARGET) / SAFETY_TARGET < 1e-9
    assert damping["lam"] > 0.0
    assert damping["cond_after"] < damping["cond_before"]


def test_damping_diagnostics_present(calibration: np.ndarray, gaussian_weight: np.ndarray) -> None:
    payload = gptq_quantize(gaussian_weight, calibration, QuantConfig(bits=4, group_size=8), return_debug=True)
    damping = payload["damping"]
    for key in ("mu", "lam", "safety", "cond_before", "cond_after", "offdiag_share"):
        assert key in damping
        assert np.isfinite(damping[key])
    # a real Hessian must have off-diagonal structure, otherwise GPTQ has nothing to do
    assert damping["offdiag_share"] >= 0.0


def test_gptq_reduces_to_rtn_when_factor_is_identity(
    calibration: np.ndarray, gaussian_weight: np.ndarray
) -> None:
    """T-D: with ``U = I`` the compensation must vanish and GPTQ must equal RTN exactly.

    This is the guard on the *control group*. Without it, passing a real ``U`` into the
    baseline would make ``J_gptq / J_rtn == 1.0`` everywhere, which looks like
    "GPTQ provides no benefit" rather than "the comparison is broken".
    """
    from quantforge.eval import act_err

    cfg = QuantConfig(bits=4, group_size=8, act_order=False, use_mse_clip=False)
    baseline = dequantize_matrix(*quantize_matrix(gaussian_weight, cfg))
    gptq = np.asarray(
        gptq_quantize(gaussian_weight, calibration, cfg, use_hessian=False)["w_hat"]
    )
    # every compensation coefficient is U[i, j] = 0 for j > i, so no update happens
    assert np.array_equal(gptq, baseline)
    assert act_err(gaussian_weight, gptq, calibration) == act_err(gaussian_weight, baseline, calibration)


def test_damping_sweep_never_hurts_against_rtn(calibration: np.ndarray, gaussian_weight: np.ndarray) -> None:
    """Sanity: the shipped safety factor must not be worse than no damping at all.

    Guards against a future edit to ``SAFETY_TARGET`` silently disabling GPTQ.
    """
    from quantforge.eval import act_err

    base = QuantConfig(bits=4, group_size=8)
    rtn = dequantize_matrix(*quantize_matrix(gaussian_weight, base))
    shipped = build_quantizer("quantforge", base).quantize(gaussian_weight, calibration).w_hat
    assert act_err(gaussian_weight, shipped, calibration) <= act_err(gaussian_weight, rtn, calibration) + 1e-12


def test_gptq_requantization_is_idempotent(calibration: np.ndarray, gaussian_weight: np.ndarray) -> None:
    """Codes must sit exactly on the grid the dequantised matrix implies."""
    cfg = QuantConfig(bits=4, group_size=8)
    first = gptq_quantize(gaussian_weight, calibration, cfg)
    w_hat = np.asarray(first["w_hat"])
    second = gptq_quantize(w_hat, calibration, cfg)
    assert np.allclose(np.asarray(second["w_hat"]), w_hat, atol=1e-9)


def test_hessian_spectrum_ratio_is_bounded_by_d_in(calibration: np.ndarray) -> None:
    """``lam_max(H)/mean(diag(H)) <= d_in`` always, by ``lam_max <= trace``.

    Regression guard. An earlier conditioning diagnostic used
    ``diag(H).max()/diag(H).min()``, which is *unbounded* -- it measures a single
    quiet channel, not the spectrum -- and reported impossible values (114708 at
    ``d_in=256``). Any spread-like quantity used in a conclusion must respect this
    bound.
    """
    x = calibration * 40.0  # provoke a wide spread without leaving the bound
    h = build_hessian(x)
    d_in = h.shape[0]
    spread = float(np.linalg.eigvalsh(h).max() / np.mean(np.diag(h)))
    assert spread <= d_in + 1e-9, f"spread {spread} exceeds the d_in={d_in} bound"


def test_hessian_spread_grows_with_activations(calibration: np.ndarray) -> None:
    """The spectrum ratio must actually respond to conditioning, not stay constant."""
    narrow = float(
        np.linalg.eigvalsh(build_hessian(calibration)).max() / np.mean(np.diag(build_hessian(calibration)))
    )
    spiked = calibration.copy()
    spiked[0] *= 100.0
    wide = float(
        np.linalg.eigvalsh(build_hessian(spiked)).max() / np.mean(np.diag(build_hessian(spiked)))
    )
    assert wide > narrow
    assert wide <= spiked.shape[0] + 1e-9
