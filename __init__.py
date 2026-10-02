"""QuantForge -- post-training quantization for weight compression.

Public surface::

    from quantforge import QuantConfig, run_pipeline, SyntheticMLP, build_quantizer

Layering is strictly acyclic::

    cli -> pipeline -> {data, quant, eval} -> core
"""

from __future__ import annotations

import os as _os

# Pin BLAS threads before numpy is imported anywhere: multi-threaded GEMM changes
# summation order between runs, which breaks the bit-identical determinism guarantee.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    _os.environ.setdefault(_var, "1")

from .core import (  # noqa: E402
    Benchmark,
    ConfigError,
    DataError,
    DecompositionError,
    MetricError,
    PipelineError,
    QuantConfig,
    QuantForgeError,
    QuantizationError,
    QuantResult,
    RunConfig,
    RunRecord,
    ShapeError,
    ValidationError,
    __version__,
    load_config,
    set_all,
)
from .data import SyntheticMLP, make_mlp  # noqa: E402
from .eval import act_err, nmse, sqnr_db  # noqa: E402
from .pipeline import run_pipeline  # noqa: E402
from .quant import build_quantizer, capability_report  # noqa: E402

__all__ = [
    "Benchmark",
    "ConfigError",
    "DataError",
    "DecompositionError",
    "MetricError",
    "PipelineError",
    "QuantConfig",
    "QuantForgeError",
    "QuantResult",
    "QuantizationError",
    "RunConfig",
    "RunRecord",
    "ShapeError",
    "SyntheticMLP",
    "ValidationError",
    "__version__",
    "act_err",
    "build_quantizer",
    "capability_report",
    "load_config",
    "make_mlp",
    "nmse",
    "run_pipeline",
    "set_all",
    "sqnr_db",
]
