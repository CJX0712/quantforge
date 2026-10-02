"""Data layer: reproducible synthetic workloads."""

from __future__ import annotations

from .synthetic import (
    MODEL_SHAPES,
    MODEL_SPECS,
    SyntheticConfig,
    SyntheticMLP,
    make_layer,
    make_mlp,
)

__all__ = [
    "MODEL_SHAPES",
    "MODEL_SPECS",
    "SyntheticConfig",
    "SyntheticMLP",
    "make_layer",
    "make_mlp",
]
