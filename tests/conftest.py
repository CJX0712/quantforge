"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest
from quantforge.core import QuantConfig, set_all
from quantforge.core.seed import pin_blas_threads

# Pin BLAS before anything else so every test sees identical GEMM reduction order.
pin_blas_threads(1)


@pytest.fixture(autouse=True)
def _deterministic_rng() -> None:
    set_all(1234)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(7)


@pytest.fixture
def gaussian_weight(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((8, 16))


@pytest.fixture
def calibration(rng: np.random.Generator) -> np.ndarray:
    x = np.maximum(rng.standard_normal((16, 96)), 0.0)
    x[:3, :8] *= 15.0  # a few outlier channels on a few samples
    return x


@pytest.fixture
def qcfg() -> QuantConfig:
    return QuantConfig(bits=4, group_size=8)
