"""Run configuration with schema validation and ``ENV_QF_*`` overrides.

Precedence: explicit kwargs > ``ENV_QF_*`` environment variables > dataclass defaults.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from .errors import ConfigError, ValidationError
from .types import QuantConfig

__all__ = ["ENV_PREFIX", "SCHEMA", "RunConfig", "load_config"]

ENV_PREFIX = "ENV_QF_"

#: Damping safety factor. Damping is ``lam = mean(diag(H)) / SAFETY_TARGET``, so this is
#: exactly the ratio ``mu/lam`` by construction: it does not depend on ``d_in`` nor on
#: the Hessian's spectrum, which makes it the one damping quantity worth exposing.
#:
#: Chosen by measurement, not by derivation. Sweeping outlier gain from 1x to 500x spans
#: Hessian spreads from 5 to 2766; the flagship's weight-NMSE edge over RTN is:
#:
#:   SAFETY   1.5     2      3      5      20     100
#:   spread~5  +13.7  +11.6  +9.3   +5.9   -1.0   -2.5
#:   spread~116 +15.6 +13.3  +6.8   -6.3  -71.6  -214.5
#:   spread~2766 +21.8 +21.4 +20.3  +17.5 -17.2  -173.8
#:
#: SAFETY <= 3 is positive in *every* conditioning regime; 20 collapses once the Hessian
#: is ill-conditioned. 3.0 is the largest round value that keeps a margin, so
#: ``percdamp = 1/3``. The identity ``mu/lam == SAFETY_TARGET`` is asserted in the test
#: suite so a later edit cannot silently start controlling a different quantity.
SAFETY_TARGET = 3.0

#: JSON-ish schema used by :meth:`RunConfig.validate`. Coercion is explicit.
SCHEMA: dict[str, tuple[type, ...]] = {
    "seed": (int,),
    "bits": (int,),
    "group_size": (int,),
    "n_calib": (int,),
    "dataset_seeds": (list, tuple),
    "models": (list, tuple),
    "methods": (list, tuple),
    "bits_grid": (list, tuple),
    "group_grid": (list, tuple),
    "outdir": (str,),
    "ablations": (list, tuple),
    "calib_samples": (int,),
}


def _coerce(name: str, raw: str) -> Any:
    """Parse an environment string into the declared Python type."""
    want = SCHEMA[name]
    try:
        if want == (int,):
            return int(raw)
        if want in ((list,), (tuple,)):
            parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
            if want == (list,):
                return parts
            return tuple(parts)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"cannot parse {ENV_PREFIX}{name}={raw!r}", field=name) from exc
    return raw


@dataclass(slots=True)
class RunConfig:
    """Top-level experiment configuration for the benchmark pipeline."""

    seed: int = 42
    bits: int = 4
    group_size: int = 128
    n_calib: int = 512
    calib_samples: int = 256
    dataset_seeds: tuple[int, ...] = (7, 13, 23)
    models: tuple[str, ...] = ("m0", "m1", "m2", "m3")
    methods: tuple[str, ...] = ("rtn_tensor", "rtn_group", "awq", "quantforge")
    bits_grid: tuple[int, ...] = (2, 3, 4, 8)
    group_grid: tuple[int, ...] = (128, 64)
    outdir: str = "results"
    ablations: tuple[str, ...] = (
        "no_hessian",
        "no_act_aware",
        "no_act_order",
        "no_mse_clip",
    )

    def validate(self) -> RunConfig:
        """Validate ranges and cross-field consistency; returns self for chaining."""
        if self.seed < 0:
            raise ValidationError("seed must be non-negative", seed=self.seed)
        if not 2 <= self.bits <= 8:
            raise ValidationError("bits must be in [2, 8]", bits=self.bits)
        if self.group_size <= 0 or self.n_calib <= 0 or self.calib_samples <= 0:
            raise ValidationError(
                "group_size/n_calib/calib_samples must be positive",
                group_size=self.group_size,
                n_calib=self.n_calib,
                calib_samples=self.calib_samples,
            )
        if self.calib_samples > self.n_calib:
            raise ValidationError(
                "calib_samples must not exceed n_calib",
                calib_samples=self.calib_samples,
                n_calib=self.n_calib,
            )
        if not self.dataset_seeds:
            raise ValidationError("dataset_seeds must be non-empty")
        if len(set(self.dataset_seeds)) != len(self.dataset_seeds):
            raise ValidationError("dataset_seeds must be unique", dataset_seeds=self.dataset_seeds)
        if not self.models or not self.methods:
            raise ValidationError("models and methods must be non-empty")
        for b in self.bits_grid:
            if not 2 <= b <= 8:
                raise ValidationError("bits_grid entries must lie in [2, 8]", bit=b)
        for g in self.group_grid:
            if g <= 0:
                raise ValidationError("group_grid entries must be positive", group_size=g)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def quant_config(self, bits: int | None = None, group_size: int | None = None) -> QuantConfig:
        """Build the per-call :class:`QuantConfig` implied by this run config."""
        return QuantConfig(
            bits=self.bits if bits is None else bits,
            group_size=self.group_size if group_size is None else group_size,
        )


def load_config(overrides: Mapping[str, Any] | None = None, *, use_env: bool = True) -> RunConfig:
    """Build a validated :class:`RunConfig` from defaults, ``ENV_QF_*`` and overrides."""
    values: dict[str, Any] = {f.name: getattr(RunConfig, f.name, None) for f in fields(RunConfig)}
    defaults = RunConfig()
    values = defaults.to_dict()

    if use_env:
        for name in SCHEMA:
            raw = os.environ.get(ENV_PREFIX + name.upper())
            if raw is not None:
                values[name] = _coerce(name, raw)

    unknown = sorted(set(overrides or {}) - set(SCHEMA))
    if unknown:
        raise ConfigError("unknown config keys", keys=unknown)
    values.update(overrides or {})

    # Normalise sequence-typed fields: they may arrive as comma-separated strings from
    # the environment, and the int-valued ones must be coerced to int elements.
    for key in ("dataset_seeds", "bits_grid", "group_grid"):
        raw = values.get(key)
        if isinstance(raw, str):
            raw = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
        if isinstance(raw, (list, tuple)):
            caster = int if key != "models" else str
            values[key] = tuple(caster(v) for v in raw)
    for key in ("models", "methods", "ablations"):
        raw = values.get(key)
        if isinstance(raw, str):
            values[key] = tuple(p.strip() for p in raw.replace(";", ",").split(",") if p.strip())

    return RunConfig(**values).validate()
