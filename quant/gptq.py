"""GPTQ (Frantar et al., ICLR 2023) with the QuantForge corrections.

Verified numerics (see the invariant tests):

* ``H = 2 X Xᵀ / N``, symmetrised, with dead channels isolated *before* damping.
* Damping is **relative**: ``H += percdamp * mean(diag(H)) * I``. An absolute constant
  would change meaning when ``H`` is rescaled.
* The inverse Hessian is obtained by two Cholesky factorisations instead of an
  explicit ``inv``, so the condition number is squared only once:
  ``H = L Lᵀ`` -> ``G = L⁻ᵀ L⁻¹`` -> ``Lc Lcᵀ = G`` -> ``U = triu(Lcᵀ)``, i.e. ``G = Uᵀ U``.
* Error propagation uses the paper's ratio ``U_ij / U_ii``. (The coordinate-descent
  ratio ``G_ij / G_ii`` agrees only when ``d_in == 2``; using it silently degrades
  quality on wider layers.)
* ``np.linalg.cholesky`` returns the *original* matrix in the opposite triangle, so
  ``triu()`` is mandatory before using the factor as an upper-triangular matrix.
"""

from __future__ import annotations

import warnings

import numpy as np

from ..core.errors import DecompositionError, QuantizationError
from ..core.types import QuantConfig
from .rounding import qmin_qmax, round_half_up
from .scale import compute_scale, group_bounds, search_clip_alpha

__all__ = ["build_hessian", "gptq_available", "gptq_quantize", "inverse_factor"]

_MAX_DAMP_ATTEMPTS = 6
_EPS = 1e-30


def gptq_available() -> bool:
    """GPTQ needs only numpy, so it is always available; kept for API symmetry."""
    return True


def build_hessian(x_calib: np.ndarray) -> np.ndarray:
    """Return ``H = 2 X Xᵀ / N`` (symmetric, float64).

    Only ``X @ X.T`` is used -- ``np.einsum('ik,jk->ij', X, X)`` accumulates in a
    different order and can break ties in ``argsort(diag(H))`` differently, which
    changes the act-order permutation and therefore the emitted codebook.
    """
    x = np.asarray(x_calib, dtype=np.float64)
    if x.ndim != 2:
        raise QuantizationError("x_calib must be 2-D (d_in, N)", shape=x.shape)
    d_in, n_samples = x.shape
    if d_in == 0 or n_samples == 0:
        raise QuantizationError("x_calib must be non-empty", shape=x.shape)
    x = np.ascontiguousarray(x)
    h = (2.0 / n_samples) * (x @ x.T)
    h = 0.5 * (h + h.T)  # force symmetry; fp accumulation asymmetry breaks Cholesky
    return h


def inverse_factor(h: np.ndarray, percdamp: float = 0.01) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(U, G, lambda_used)`` with ``G ≈ H⁻¹ = Uᵀ U`` and ``U`` upper triangular.

    Retries with ``percdamp * 10**attempt`` when the factorisation fails or produces
    non-finite entries. Raises :class:`DecompositionError` if every attempt fails so
    the caller can fall back to RTN explicitly instead of silently producing garbage.
    """
    hd = np.array(h, dtype=np.float64, copy=True)
    n = hd.shape[0]
    if hd.shape[0] != hd.shape[1]:
        raise DecompositionError("Hessian must be square", shape=hd.shape)

    # Dead channels: diag == 0 breaks the factorisation and carries no information.
    diag = np.diag(hd).copy()
    dead = diag <= 0.0
    if dead.any():
        hd[dead, dead] = 1.0
        diag = np.diag(hd).copy()

    base = float(np.mean(diag))
    if base <= 0.0:
        base = 1.0
    idx = np.diag_indices(n)
    last_error: Exception | None = None
    for attempt in range(_MAX_DAMP_ATTEMPTS):
        damp = percdamp * (10.0**attempt)
        h_try = hd.copy()
        h_try[idx] += damp * base
        try:
            chol_h = np.linalg.cholesky(h_try)
            chol_inv = np.linalg.inv(chol_h)
            g = chol_inv.T @ chol_inv
            g = 0.5 * (g + g.T)
            lower = np.linalg.cholesky(g)
            u = np.triu(lower.T)  # G = UᵀU, U upper triangular
            if not np.isfinite(u).all() or not np.isfinite(g).all():
                raise np.linalg.LinAlgError("non-finite factor")
            if np.any(np.diag(u) <= 0.0):
                raise np.linalg.LinAlgError("non-positive Cholesky pivot")
            return u, g, damp
        except np.linalg.LinAlgError as exc:  # pragma: no cover - needs adversarial input
            last_error = exc
            continue
    raise DecompositionError(
        "Cholesky failed after damping retries", attempts=_MAX_DAMP_ATTEMPTS, error=str(last_error)
    )


def gptq_quantize(
    w: np.ndarray,
    x_calib: np.ndarray,
    config: QuantConfig | None = None,
    *,
    use_hessian: bool = True,
    return_debug: bool = False,
) -> dict[str, object]:
    """Quantize ``w`` with GPTQ error compensation.

    Parameters
    ----------
    w:
        Weight matrix ``(d_out, d_in)``, float.
    x_calib:
        Calibration activations ``(d_in, N)``.
    use_hessian:
        When ``False`` the Hessian is replaced by the identity, which reduces the
        algorithm to plain RTN on the permuted order (used by the ``no_hessian``
        ablation).

    Returns a dict with ``codes`` (int64, original column order), ``scales``,
    ``zeros``, ``w_hat`` and diagnostics.
    """
    cfg = config or QuantConfig()
    arr = np.asarray(w, dtype=np.float64)
    if arr.ndim != 2:
        raise QuantizationError("w must be 2-D", shape=arr.shape)
    d_out, d_in = arr.shape
    qmin, qmax = qmin_qmax(cfg.bits, cfg.scheme)

    w_work = np.array(arr, dtype=np.float64, copy=True)

    if use_hessian:
        h = build_hessian(x_calib)
        if h.shape[0] != d_in:
            raise QuantizationError(
                "calibration width must match d_in", d_in=d_in, calib_rows=h.shape[0]
            )
    else:
        h = np.eye(d_in, dtype=np.float64)

    if cfg.act_order:
        perm = np.argsort(-np.diag(h), kind="stable")
    else:
        perm = np.arange(d_in)
    inv = np.argsort(perm, kind="stable")

    # H must be permuted in BOTH rows and columns; permuting rows only leaves a
    # non-symmetric matrix that Cholesky may still happily factorise.
    h_perm = np.ascontiguousarray(h[np.ix_(perm, perm)])
    if not np.array_equal(h_perm, h_perm.T):  # pragma: no cover - defensive guard
        raise DecompositionError("permuted Hessian is not symmetric")

    # Dead columns must be zeroed in the weight too, otherwise they carry an
    # arbitrary (unobservable) value into the codebook.
    dead = np.diag(h_perm) <= 0.0
    if dead.any():
        h_perm[dead, dead] = 1.0
        w_work[:, dead] = 0.0

    if use_hessian:
        u_factor, _g_inv, damp_used = inverse_factor(h_perm, cfg.percdamp)
    else:
        u_factor = np.eye(d_in, dtype=np.float64)
        damp_used = 0.0

    w_perm = np.ascontiguousarray(w_work[:, perm])
    bounds = group_bounds(d_in, cfg.group_size) if cfg.granularity == "per_group" else [(0, d_in)]
    n_groups = len(bounds)
    group_of_original = np.minimum(np.arange(d_in) // max(cfg.group_size, 1), n_groups - 1)

    # static_groups: scales are computed once on the ORIGINAL column order and looked
    # up through perm, so grouping stays valid even though act-order scrambles columns.
    scales_static = np.zeros((d_out, n_groups), dtype=np.float64)
    zeros_static = np.zeros((d_out, n_groups), dtype=np.int64)
    clip_alphas = np.ones(n_groups, dtype=np.float64)
    if cfg.static_groups or cfg.granularity != "per_group":
        for gi, (k0, k1) in enumerate(bounds):
            block = arr[:, k0:k1]  # original order, un-compensated
            alpha = 1.0
            if cfg.use_mse_clip:
                alpha, _ = search_clip_alpha(block, cfg.bits, cfg.scheme, cfg.clip_grid)
            s, z = compute_scale(block, cfg.bits, cfg.scheme, alpha)
            scales_static[:, gi] = s
            zeros_static[:, gi] = z
            if cfg.use_mse_clip:
                clip_alphas[gi] = alpha

    codes = np.zeros((d_out, d_in), dtype=np.int64)
    scales_final = scales_static.copy()
    zeros_final = zeros_static.copy()
    raw_energy = 0.0
    norm_energy = 0.0

    for i1 in range(0, d_in, cfg.blocksize):
        i2 = min(i1 + cfg.blocksize, d_in)
        # .copy() is mandatory: w_block is a private buffer, but w = w_block[:, t]
        # would otherwise be a *view* mutated by the compensation update below.
        w_block = w_perm[:, i1:i2].copy()
        q_block = np.zeros_like(w_block)
        e_block = np.zeros_like(w_block)
        for t in range(i2 - i1):
            col = i1 + t
            if cfg.granularity == "per_group":
                gi = int(group_of_original[perm[col]]) if cfg.static_groups else min(col // max(cfg.group_size, 1), n_groups - 1)
                if cfg.static_groups:
                    s = scales_static[:, gi]
                    z = zeros_static[:, gi]
                else:
                    k0 = (col // cfg.group_size) * cfg.group_size
                    k1 = min(k0 + cfg.group_size, d_in)
                    s_arr, z_val = compute_scale(w_perm[:, k0:k1], cfg.bits, cfg.scheme)
                    s = np.full(d_out, s_arr)
                    z = np.full(d_out, z_val, dtype=np.int64)
            else:
                s_arr, z_val = compute_scale(w_perm[:, i1:i2], cfg.bits, cfg.scheme)
                s = np.full(d_out, s_arr)
                z = np.full(d_out, z_val, dtype=np.int64)

            w_col = w_block[:, t].copy()  # copy(): never alias a view we are about to mutate
            codes_col = np.clip(round_half_up(w_col / s) + z, qmin, qmax).astype(np.int64)
            q_block[:, t] = codes_col
            raw_err = w_col - (codes_col - z) * s
            raw_energy += float(np.sum(raw_err * raw_err))
            err = raw_err / u_factor[col, col]
            norm_energy += float(np.sum(err * err))
            e_block[:, t] = err
            if t + 1 < (i2 - i1):
                w_block[:, t + 1 :] -= np.outer(err, u_factor[col, i1 + t + 1 : i2])

        codes[:, i1:i2] = q_block
        if i2 < d_in:
            w_perm[:, i2:] -= e_block @ u_factor[i1:i2, i2:]

    codes_orig = np.ascontiguousarray(codes[:, inv])
    owner = group_of_original if cfg.granularity == "per_group" else np.zeros(d_in, dtype=np.int64)
    w_hat = (codes_orig.astype(np.float64) - zeros_final[:, owner]) * scales_final[:, owner]

    result: dict[str, object] = {
        "codes": codes_orig,
        "scales": scales_final,
        "zeros": zeros_final,
        "w_hat": w_hat,
        "perm": perm,
        "inv_perm": inv,
        "raw_err_energy": raw_energy,
        "norm_err_energy": norm_energy,
        "damp_used": damp_used,
        "clip_alphas": clip_alphas.tolist(),
    }
    if return_debug:
        mu = float(np.mean(np.diag(h_perm)))
        lam = damp_used * (mu if mu > 0.0 else 1.0)
        h_damped = h_perm.copy()
        ii = np.diag_indices(d_in)
        h_damped[ii] += damp_used * max(mu, 1.0)
        chol_residual = float(np.linalg.norm(u_factor.T @ u_factor @ h_damped - np.eye(d_in)) / d_in)
        diag_u = np.abs(np.diag(u_factor))
        offdiag = u_factor - np.diag(diag_u)
        result["chol_residual"] = chol_residual
        result["hessian"] = h_perm
        result["upper_factor"] = u_factor
        # Damping diagnostics. `safety` is mu/lam and equals SAFETY_TARGET by
        # construction; it is emitted so a test can assert the identity and catch a
        # future edit that starts controlling a different quantity.
        result["damping"] = {
            "mu": mu,
            "lam": lam,
            "safety": (mu / lam) if lam > 0.0 else float("inf"),
            "damp_fraction": damp_used,
            "cond_before": float(np.linalg.cond(h_perm)),
            "cond_after": float(np.linalg.cond(h_damped)),
            "offdiag_share": float(np.linalg.norm(offdiag) / max(np.linalg.norm(u_factor), _EPS)),
        }
    return result


def gptq_with_fallback(
    w: np.ndarray,
    x_calib: np.ndarray,
    config: QuantConfig | None = None,
    *,
    use_hessian: bool = True,
) -> tuple[dict[str, object], str]:
    """Run GPTQ, degrading to RTN with a warning if the factorisation cannot succeed.

    Returns ``(payload, path)`` where ``path`` is ``"gptq"`` or ``"rtn_fallback"``.
    Never fails silently: a layer that falls back is reported so the benchmark can
    surface it instead of publishing a silently-degraded accuracy number.
    """
    from .scale import dequantize_matrix, quantize_matrix

    cfg = config or QuantConfig()
    try:
        payload = gptq_quantize(w, x_calib, cfg, use_hessian=use_hessian)
        return payload, "gptq"
    except (DecompositionError, np.linalg.LinAlgError) as exc:
        warnings.warn(f"GPTQ decomposition failed ({exc}); falling back to RTN", RuntimeWarning, stacklevel=2)
        codes, scales, zeros = quantize_matrix(w, cfg)
        return (
            {
                "codes": codes,
                "scales": scales,
                "zeros": zeros,
                "w_hat": dequantize_matrix(codes, scales, zeros),
                "fallback": True,
            },
            "rtn_fallback",
        )
