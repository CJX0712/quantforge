"""Single source of randomness for the whole project.

Only one RNG exists: :func:`set_all` seeds ``random``, ``numpy`` global and
creates a project-wide :func:`get_rng` generator. Quantizer cores themselves are
deterministic and consume no randomness.
"""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["BLAS_ENV", "calibration_subset", "current_seed", "get_rng", "set_all"]

#: BLAS env vars that must be pinned before numpy import for bit-reproducibility.
BLAS_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_STATE: dict[str, object] = {"seed": 0, "rng": np.random.default_rng(0)}


def set_all(seed: int) -> np.random.Generator:
    """Seed every global RNG and return the project generator."""
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    _STATE["seed"] = seed
    _STATE["rng"] = np.random.default_rng(seed)
    return _STATE["rng"]  # type: ignore[return-value]


def get_rng() -> np.random.Generator:
    """Return the project generator, seeding with 0 on first use."""
    rng = _STATE.get("rng")
    if rng is None:
        return set_all(0)
    return rng  # type: ignore[return-value]


def current_seed() -> int:
    return int(_STATE["seed"])  # type: ignore[arg-type]


def pin_blas_threads(n: int = 1) -> dict[str, str]:
    """Force single-threaded BLAS so GEMM reductions are bit-stable.

    Multi-threaded OpenBLAS changes summation order run-to-run, which breaks the
    bit-identical determinism guarantee. Safe to call after numpy import because
    the vars are re-read by child BLAS runtimes; CI also sets them early.
    """
    value = str(n)
    applied = {}
    for key in BLAS_ENV:
        os.environ[key] = value
        applied[key] = value
    return applied


def calibration_subset(n_total: int, n_want: int, seed: int) -> np.ndarray:
    """Deterministic column indices for calibration sampling (sorted, unique)."""
    if not 0 < n_want <= n_total:
        raise ValueError(f"require 0 < n_want <= n_total, got n_want={n_want}, n_total={n_total}")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=n_want, replace=False))
