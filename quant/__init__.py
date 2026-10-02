"""Quantization layer: rounding, scales, GPTQ, AWQ, schemes, backend probes."""

from __future__ import annotations

from .awq import apply_awq_scaling, awq_available, awq_search, channel_salience
from .gptq import build_hessian, gptq_available, gptq_quantize, gptq_with_fallback, inverse_factor
from .rounding import (
    dequantize_affine,
    qmin_qmax,
    quantize_affine,
    quantize_with_params,
    round_half_up,
)
from .scale import (
    bits_per_weight,
    compute_scale,
    dequantize_matrix,
    group_bounds,
    n_groups_for,
    quantize_matrix,
    search_clip_alpha,
)
from .schemes import (
    BASELINES,
    FLAGSHIP,
    METHODS,
    AwqQuantizer,
    QuantForgeQuantizer,
    RtnGroup,
    RtnTensor,
    build_quantizer,
)
from .selectors import (
    available_numpy,
    available_scipy,
    available_sklearn,
    capability_report,
    select_backend,
)

__all__ = [
    "BASELINES",
    "FLAGSHIP",
    "METHODS",
    "AwqQuantizer",
    "QuantForgeQuantizer",
    "RtnGroup",
    "RtnTensor",
    "apply_awq_scaling",
    "available_numpy",
    "available_scipy",
    "available_sklearn",
    "awq_available",
    "awq_search",
    "bits_per_weight",
    "build_hessian",
    "build_quantizer",
    "capability_report",
    "channel_salience",
    "compute_scale",
    "dequantize_affine",
    "dequantize_matrix",
    "gptq_available",
    "gptq_quantize",
    "gptq_with_fallback",
    "group_bounds",
    "inverse_factor",
    "n_groups_for",
    "qmin_qmax",
    "quantize_affine",
    "quantize_matrix",
    "quantize_with_params",
    "round_half_up",
    "search_clip_alpha",
    "select_backend",
]
