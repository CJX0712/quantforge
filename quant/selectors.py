"""Backend probes and runtime capability detection.

The flagship path uses numpy only. ``scipy`` is optional and used, when present, for a
faster/safer inverse; when absent the code falls back to numpy and the degradation is
reported through :func:`capability_report` rather than hidden.
"""

from __future__ import annotations

import importlib
import platform
import sys
from collections.abc import Callable

import numpy as np

__all__ = ["available_numpy", "available_scipy", "available_sklearn", "capability_report", "select_backend"]


def _probe(module: str) -> bool:
    try:
        importlib.import_module(module)
    except Exception:  # pragma: no cover - depends on the environment
        return False
    return True


def available_numpy() -> bool:
    return _probe("numpy")


def available_scipy() -> bool:
    return _probe("scipy")


def available_sklearn() -> bool:
    return _probe("sklearn")


def capability_report() -> dict[str, object]:
    """JSON-serialisable description of the runtime, recorded in benchmark metadata."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": available_scipy(),
        "sklearn": available_sklearn(),
        "blas_threads_env": {
            k: __import__("os").environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def select_backend(prefer_scipy: bool = True) -> tuple[str, Callable[[], None]]:
    """Return ``(backend_name, noop)`` describing which linear-algebra path is active.

    ``"scipy"`` when scipy is importable and requested, otherwise ``"numpy"``. The
    numpy path is fully functional, so this only affects speed and conditioning
    robustness, never correctness.
    """
    if prefer_scipy and available_scipy():
        return "scipy", lambda: None
    return "numpy", lambda: None
