"""Core layer: types, errors, config, interfaces, seeding. No sibling imports."""

from __future__ import annotations

from .config import ENV_PREFIX, RunConfig, load_config
from .errors import (
    ERROR_CODES,
    ConfigError,
    DataError,
    DecompositionError,
    MetricError,
    PipelineError,
    QuantForgeError,
    QuantizationError,
    ShapeError,
    ValidationError,
    error_code_of,
)
from .interfaces import DataSource, Evaluator, Metric, Quantizer
from .seed import calibration_subset, current_seed, get_rng, pin_blas_threads, set_all
from .types import (
    Benchmark,
    LayerBundle,
    QuantConfig,
    QuantResult,
    RunRecord,
    as_2d_float64,
    as_float64,
)

__version__ = "0.1.0"

__all__ = [
    "ENV_PREFIX",
    "ERROR_CODES",
    "Benchmark",
    "ConfigError",
    "DataError",
    "DataSource",
    "DecompositionError",
    "Evaluator",
    "LayerBundle",
    "Metric",
    "MetricError",
    "PipelineError",
    "QuantConfig",
    "QuantForgeError",
    "QuantResult",
    "QuantizationError",
    "Quantizer",
    "RunConfig",
    "RunRecord",
    "ShapeError",
    "ValidationError",
    "__version__",
    "as_2d_float64",
    "as_float64",
    "calibration_subset",
    "current_seed",
    "error_code_of",
    "get_rng",
    "load_config",
    "pin_blas_threads",
    "set_all",
]
