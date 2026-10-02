"""Core layer: errors, config, seeding, typed contracts."""

from __future__ import annotations

import numpy as np
import pytest
from quantforge.core import (
    ConfigError,
    QuantConfig,
    RunConfig,
    ValidationError,
    load_config,
    set_all,
)
from quantforge.core.errors import ERROR_CODES, DecompositionError, error_code_of
from quantforge.core.seed import BLAS_ENV, calibration_subset, current_seed, get_rng, pin_blas_threads


def test_error_codes_are_unique_and_prefixed() -> None:
    codes = list(ERROR_CODES)
    assert len(codes) == len(set(codes))
    for code in codes:
        assert code.startswith("E") and len(code) == 4


def test_error_message_embeds_code_and_context() -> None:
    err = DecompositionError("boom", attempts=6)
    assert err.code == "E310"
    assert "E310" in str(err) and "attempts=6" in str(err)


def test_error_code_of_foreign_exception() -> None:
    assert error_code_of(ValueError("x")) == "E999"
    assert error_code_of(ConfigError("x")) == "E100"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bits": 1},
        {"bits": 9},
        {"scheme": "weird"},
        {"granularity": "weird"},
        {"group_size": 0},
        {"blocksize": -1},
        {"percdamp": 0.0},
        {"percdamp": 1.0},
        {"awq_alpha_grid": ()},
        {"awq_alpha_grid": (1.5,)},
    ],
)
def test_quant_config_rejects_invalid(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        QuantConfig(**kwargs)


def test_quant_config_evolve_is_pure() -> None:
    base = QuantConfig()
    other = base.evolve(bits=8)
    assert base.bits == 4 and other.bits == 8


def test_run_config_validation() -> None:
    assert RunConfig().validate().seed == 42
    with pytest.raises(ValidationError):
        RunConfig(seed=-1).validate()
    with pytest.raises(ValidationError):
        RunConfig(bits=9).validate()
    with pytest.raises(ValidationError):
        RunConfig(calib_samples=999, n_calib=4).validate()
    with pytest.raises(ValidationError):
        RunConfig(dataset_seeds=(7, 7)).validate()


def test_load_config_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENV_QF_BITS", "8")
    monkeypatch.setenv("ENV_QF_MODELS", "m0,m2")
    cfg = load_config()
    assert cfg.bits == 8 and cfg.models == ("m0", "m2")


def test_load_config_rejects_unknown_key() -> None:
    with pytest.raises(ConfigError):
        load_config({"not_a_field": 1})


def test_set_all_is_reproducible() -> None:
    a = set_all(99).standard_normal(5)
    b = set_all(99).standard_normal(5)
    assert np.array_equal(a, b)
    assert current_seed() == 99


def test_set_all_rejects_non_int() -> None:
    with pytest.raises(TypeError):
        set_all("7")  # type: ignore[arg-type]


def test_get_rng_advances_so_calls_differ() -> None:
    """get_rng returns the live generator, so successive draws must differ."""
    set_all(5)
    a = get_rng().random(4)
    b = get_rng().random(4)
    assert not np.array_equal(a, b)


def test_get_rng_is_the_same_object_as_set_all_returns() -> None:
    rng = set_all(5)
    assert get_rng() is rng


def test_pin_blas_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in BLAS_ENV:
        monkeypatch.delenv(key, raising=False)
    applied = pin_blas_threads(1)
    assert set(applied) == set(BLAS_ENV)
    assert all(v == "1" for v in applied.values())


def test_calibration_subset_sorted_and_seeded() -> None:
    a = calibration_subset(100, 10, 3)
    b = calibration_subset(100, 10, 3)
    c = calibration_subset(100, 10, 4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.all(np.diff(a) > 0) and a.max() < 100


def test_calibration_subset_validates() -> None:
    with pytest.raises(ValueError):
        calibration_subset(10, 0, 1)
    with pytest.raises(ValueError):
        calibration_subset(10, 11, 1)


def test_layer_bundle_bounds_check() -> None:
    from quantforge.core import LayerBundle

    b = LayerBundle(weights=(np.zeros((2, 2)),), acts=(np.zeros((2, 2)),), x_calib=np.zeros((2, 2)))
    assert b.n_layers == 1
    b.layer(0)
    with pytest.raises(IndexError):
        b.layer(5)
