"""The four quantization methods under comparison, plus ablation variants.

============  ====================================================================
key           definition
============  ====================================================================
``rtn_tensor``  A. round-to-nearest, per-tensor min-max scale (the naive baseline)
``rtn_group``   B. round-to-nearest, per-group min-max + MSE-optimal clipping
``awq``         C. AWQ per-channel scaling searched against output MSE, then RTN
``quantforge``  D. flagship: AWQ scaling -> GPTQ error compensation
============  ====================================================================

Ablations (``quantforge`` with one component removed):

* ``no_hessian``   -- GPTQ's Hessian replaced by the identity (reduces to RTN)
* ``no_act_aware`` -- AWQ scaling skipped
* ``no_act_order`` -- act-order permutation disabled
* ``no_mse_clip``  -- min-max scales instead of the MSE-optimal clip search
"""

from __future__ import annotations

import numpy as np

from ..core.errors import QuantizationError
from ..core.interfaces import Quantizer
from ..core.types import QuantConfig, QuantResult
from .awq import awq_search
from .gptq import gptq_quantize
from .scale import (
    bits_per_weight,
    compute_scale,
    dequantize_matrix,
    group_bounds,
    quantize_matrix,
    search_clip_alpha,
)

__all__ = ["BASELINES", "FLAGSHIP", "METHODS", "AwqQuantizer", "QuantForgeQuantizer", "RtnGroup", "RtnTensor", "build_quantizer"]

METHODS = ("rtn_tensor", "rtn_group", "awq", "quantforge")
BASELINES = ("rtn_tensor", "rtn_group")
FLAGSHIP = "quantforge"


def _result(
    w: np.ndarray,
    codes: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    config: QuantConfig,
    aux: dict[str, object] | None = None,
) -> QuantResult:
    arr = np.asarray(w, dtype=np.float64)
    return QuantResult(
        w_hat=dequantize_matrix(codes, scales, zeros),
        codes=codes,
        scales=scales,
        zeros=zeros,
        bits_per_weight=bits_per_weight(config, arr.size, arr.shape[1]),
        aux=aux or {},
    )


def _ls_refit_scales(
    w: np.ndarray, codes: np.ndarray, scales: np.ndarray, config: QuantConfig
) -> tuple[np.ndarray, bool]:
    """Closed-form least-squares refit of every scale for fixed integer codes.

    With codes ``c`` held fixed, ``Ŵ = s·c`` and minimising ``‖W - s·c‖²`` over ``s``
    gives ``s* = Σ(W·c) / Σ(c²)``. A zero group (all codes zero) keeps its old scale.
    Returns ``(new_scales, changed)``.
    """
    arr = np.asarray(w, dtype=np.float64)
    c = np.asarray(codes, dtype=np.float64)
    d_in = arr.shape[1]
    bounds = group_bounds(d_in, config.group_size) if config.granularity == "per_group" else [(0, d_in)]
    new = np.array(scales, dtype=np.float64, copy=True)
    changed = False
    for gi, (k0, k1) in enumerate(bounds):
        c_blk = c[:, k0:k1]
        num = np.sum(arr[:, k0:k1] * c_blk, axis=1)
        den = np.sum(c_blk * c_blk, axis=1)
        ok = den > 0.0
        if not np.any(ok):
            continue
        fitted = np.where(ok, num / np.where(ok, den, 1.0), scales[:, gi])
        # keep the scale positive so the symmetric codebook stays centred on zero
        fitted = np.where(np.abs(fitted) < 1e-12, scales[:, gi], np.abs(fitted))
        if not np.allclose(fitted, scales[:, gi], rtol=1e-12, atol=0.0):
            changed = True
        new[:, gi] = fitted
    return new, changed


class RtnTensor(Quantizer):
    """Method A: per-tensor min-max round-to-nearest."""

    method = "rtn_tensor"

    def __init__(self, config: QuantConfig | None = None) -> None:
        self.config = config or QuantConfig(granularity="per_tensor")

    def quantize(self, w: np.ndarray, x_calib: np.ndarray | None = None) -> QuantResult:
        cfg = self.config.evolve(granularity="per_tensor")
        codes, scales, zeros = quantize_matrix(w, cfg)
        return _result(w, codes, scales, zeros, cfg, {"scheme": "per_tensor_minmax"})


class RtnGroup(Quantizer):
    """Method B: per-group min-max with an MSE-optimal clip ratio per group."""

    method = "rtn_group"

    def __init__(self, config: QuantConfig | None = None) -> None:
        self.config = config or QuantConfig(granularity="per_group")

    def quantize(self, w: np.ndarray, x_calib: np.ndarray | None = None) -> QuantResult:
        cfg = self.config
        arr = np.asarray(w, dtype=np.float64)
        if arr.ndim != 2:
            raise QuantizationError("w must be 2-D", shape=arr.shape)
        d_out, d_in = arr.shape
        bounds = group_bounds(d_in, cfg.group_size) if cfg.granularity == "per_group" else [(0, d_in)]
        codes = np.zeros((d_out, d_in), dtype=np.int64)
        scales = np.zeros((d_out, len(bounds)), dtype=np.float64)
        zeros = np.zeros((d_out, len(bounds)), dtype=np.int64)
        alphas = np.zeros(len(bounds), dtype=np.float64)
        for gi, (k0, k1) in enumerate(bounds):
            block = arr[:, k0:k1]
            alpha, _ = search_clip_alpha(block, cfg.bits, cfg.scheme, cfg.clip_grid)
            alphas[gi] = alpha
            s, z = compute_scale(block, cfg.bits, cfg.scheme, alpha)
            scales[:, gi] = s
            zeros[:, gi] = z
            from .rounding import quantize_with_params

            codes[:, k0:k1] = quantize_with_params(block, s, z, cfg.bits, cfg.scheme)
        return _result(arr, codes, scales, zeros, cfg, {"clip_alphas": alphas.tolist()})


class AwqQuantizer(Quantizer):
    """Method C: AWQ scaling then RTN with the *same* scale strategy as method A.

    Keeping the quantizer identical to A isolates exactly one variable: the
    activation-aware rescaling.
    """

    method = "awq"

    def __init__(self, config: QuantConfig | None = None) -> None:
        self.config = config or QuantConfig(granularity="per_tensor")

    def quantize(self, w: np.ndarray, x_calib: np.ndarray | None = None) -> QuantResult:
        if x_calib is None:
            raise QuantizationError("AWQ requires calibration activations")
        cfg = self.config.evolve(granularity="per_tensor")
        found = awq_search(w, x_calib, cfg, objective="acterr")
        s = np.asarray(found["scale"], dtype=np.float64)
        codes, scales, zeros = quantize_matrix(found["w_hat"], cfg)  # type: ignore[arg-type]
        # awq_search returns W' = W diag(s); map back to the original coordinates so the
        # caller always receives a drop-in replacement for W.
        w_hat = dequantize_matrix(codes, scales, zeros) / s[None, :]
        return QuantResult(
            w_hat=w_hat,
            codes=codes,
            scales=scales,
            zeros=zeros,
            bits_per_weight=bits_per_weight(cfg, np.asarray(w).size, np.asarray(w).shape[1]),
            aux={"alpha": found["alpha"], "acterr_score": found["score"], "scale": s},
        )


class QuantForgeQuantizer(Quantizer):
    """Method D (flagship): AWQ scaling -> GPTQ error compensation -> LS scale refit.

    Three stages, in this order:

    1. **AWQ scaling.** Per-input-channel scales ``s`` found by grid search. The forward
       pass is invariant (``W diag(s) · diag(s)⁻¹ X = W X``), so the scaling costs
       nothing at inference once folded into the previous layer. The exponent is
       searched against weight NMSE rather than output error, because GPTQ re-derives
       the output error from the Hessian immediately afterwards and optimising it twice
       would double-count.
    2. **GPTQ.** Second-order error compensation using ``H = 2XXᵀ``.
    3. **LS scale refit.** GPTQ minimises the *output* error and deliberately trades
       weight-domain accuracy for it, so its min-max scales are far from optimal for
       ``‖W - Ŵ‖²``. Refitting each group in closed form, ``s* = Σ W·c / Σ c²`` over the
       group's integer codes ``c``, recovers weight NMSE without disturbing the (already
       better) code assignment.
    """

    method = "quantforge"

    def __init__(self, config: QuantConfig | None = None, ablation: str | None = None) -> None:
        self.config = config or QuantConfig()
        self.ablation = ablation
        if ablation not in (
            None,
            "no_hessian",
            "no_act_aware",
            "no_act_order",
            "no_mse_clip",
            "no_ls_refit",
        ):
            raise QuantizationError("unknown ablation", ablation=ablation)

    def quantize(self, w: np.ndarray, x_calib: np.ndarray | None = None) -> QuantResult:
        cfg = self.config
        arr = np.asarray(w, dtype=np.float64)
        aux: dict[str, object] = {"ablation": self.ablation}

        cfg_eff = cfg
        if self.ablation == "no_act_order":
            cfg_eff = cfg.evolve(act_order=False)
        if self.ablation == "no_mse_clip":
            cfg_eff = cfg.evolve(use_mse_clip=False)
        if self.ablation == "no_ls_refit":
            cfg_eff = cfg_eff.evolve(refit_scales=False)

        # stage 1: activation-aware scaling
        w_eff = arr
        s = np.ones(arr.shape[1], dtype=np.float64)
        if self.ablation == "no_act_aware":
            aux["alpha"] = 0.0
        elif x_calib is None:
            raise QuantizationError("QuantForge requires calibration activations")
        else:
            found = awq_search(arr, x_calib, cfg_eff, objective="nmse")
            s = np.asarray(found["scale"], dtype=np.float64)
            w_eff = arr * s[None, :]
            aux["alpha"] = found["alpha"]
            aux["nmse_score"] = found["score"]

        # stage 2: GPTQ in the scaled coordinate system
        payload = gptq_quantize(
            w_eff,
            x_calib if x_calib is not None else np.eye(arr.shape[1]),
            cfg_eff,
            use_hessian=self.ablation != "no_hessian",
            return_debug=True,
        )
        codes = np.asarray(payload["codes"])
        scales = np.asarray(payload["scales"])
        zeros = np.asarray(payload["zeros"])

        # stage 3: closed-form least-squares scale refit on the fixed codes
        if cfg_eff.refit_scales and not np.any(zeros):
            scales, refit = _ls_refit_scales(w_eff, codes, scales, cfg_eff)
            aux["ls_refit"] = refit
        w_hat = dequantize_matrix(codes, scales, zeros)
        if not np.allclose(s, 1.0):
            # undo the AWQ scaling: we must return a drop-in replacement for W
            w_hat = w_hat / s[None, :]

        aux.update(
            {
                "damp_used": payload.get("damp_used"),
                "chol_residual": payload.get("chol_residual"),
                "raw_err_energy": payload.get("raw_err_energy"),
                "norm_err_energy": payload.get("norm_err_energy"),
                "damping": payload.get("damping"),
            }
        )
        return QuantResult(
            w_hat=w_hat,
            codes=codes,
            scales=scales,
            zeros=zeros,
            bits_per_weight=bits_per_weight(cfg_eff, arr.size, arr.shape[1]),
            aux=aux,
        )


def build_quantizer(method: str, config: QuantConfig | None = None, ablation: str | None = None) -> Quantizer:
    """Factory for the four methods and their ablations."""
    if method == "rtn_tensor":
        return RtnTensor(config)
    if method == "rtn_group":
        return RtnGroup(config)
    if method == "awq":
        return AwqQuantizer(config)
    if method == "quantforge":
        return QuantForgeQuantizer(config, ablation=ablation)
    raise QuantizationError("unknown method", method=method, known=list(METHODS))
