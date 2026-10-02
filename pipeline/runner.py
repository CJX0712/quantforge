"""End-to-end experiment runner: builds workloads, runs the grid, writes benchmark.json."""

from __future__ import annotations

import json
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..core.config import RunConfig, load_config
from ..core.errors import PipelineError
from ..core.seed import pin_blas_threads, set_all
from ..core.types import QuantConfig, RunRecord
from ..data.synthetic import SyntheticMLP
from ..eval.benchmark import aggregate, gate_report, run_cell, summarise
from ..quant.selectors import capability_report

__all__ = ["DEFAULT_OUTDIR", "run_pipeline", "write_benchmark"]

DEFAULT_OUTDIR = "results"


def _group_sizes(config: RunConfig, bits: int) -> list[int]:
    """Per-tensor control run plus the per-group grid, without duplicates."""
    sizes = list(dict.fromkeys([*config.group_grid]))
    return sizes if sizes else [config.group_size]


def run_pipeline(
    config: RunConfig | None = None,
    *,
    outdir: str | Path | None = None,
    methods: Sequence[str] | None = None,
    bits_grid: Sequence[int] | None = None,
    models: Sequence[str] | None = None,
    dataset_seeds: Sequence[int] | None = None,
    with_ablations: bool = True,
    progress: bool = False,
) -> dict[str, Any]:
    """Run the whole grid and return the benchmark report as a plain dict.

    Determinism: BLAS is pinned to one thread (multi-threaded GEMM changes summation
    order and breaks bit-identical reruns) and every array is float64.
    """
    cfg = config or load_config()
    pin_blas_threads(1)
    set_all(cfg.seed)
    methods = list(cfg.methods if methods is None else methods)
    bits_list = list(cfg.bits_grid if bits_grid is None else bits_grid)
    model_list = list(cfg.models if models is None else models)
    seed_list = list(cfg.dataset_seeds if dataset_seeds is None else dataset_seeds)
    # `is None` rather than truthiness: an explicitly empty list must be rejected below
    # instead of silently falling back to the config defaults.
    if not methods:
        raise PipelineError("no methods selected", methods=methods)

    records: list[RunRecord] = []
    started = time.perf_counter()
    n_cells = 0

    for ds in seed_list:
        for model in model_list:
            source = SyntheticMLP(model=model, n_calib=cfg.n_calib)
            bundle = source.build(ds)
            for bits in bits_list:
                for group_size in _group_sizes(cfg, bits):
                    for method in methods:
                        # per_tensor is granularity-independent, so run that control once
                        # at the headline group size instead of once per grid point.
                        run_sizes = [cfg.group_size] if method == "rtn_tensor" else [group_size]
                        for size in run_sizes:
                            rec = run_cell(
                                bundle,
                                layer=0,
                                method=method,
                                config=QuantConfig(bits=bits, group_size=size),
                                dataset_seed=ds,
                                model=model,
                            )
                            records.append(rec)
                            n_cells += 1
                            if progress:
                                print(
                                    f"[{len(records):3d}] seed={ds} {model} b={bits} g={size} "
                                    f"{method:12s} nmse={rec.nmse:.6e} act={rec.act_err:.6e}",
                                    flush=True,
                                )

    report = summarise(records)
    ablations: dict[str, dict[str, float]] = {}
    if with_ablations:
        # Ablations run at the *headline* bit width (cfg.bits), not bits_list[0]: when a
        # grid like (2,3,4,8) is swept, the first entry is 2-bit, whose saturation would
        # dominate the comparison and say nothing about the headline configuration.
        abl_bits = cfg.bits if cfg.bits in bits_list else bits_list[0]
        for ablation in cfg.ablations:
            abl_records: list[RunRecord] = []
            for ds in seed_list:
                for model in model_list:
                    source = SyntheticMLP(model=model, n_calib=cfg.n_calib)
                    b = source.build(ds)
                    abl_records.append(
                        run_cell(
                            b,
                            layer=0,
                            method="quantforge",
                            config=QuantConfig(bits=abl_bits, group_size=cfg.group_size),
                            dataset_seed=ds,
                            model=model,
                            ablation=ablation,
                        )
                    )
            summary = aggregate(abl_records)
            key = f"quantforge:{ablation}"
            if key in summary:
                ablations[ablation] = {**summary[key], "bits": float(abl_bits)}
                if progress:
                    print(
                        f"  ablation {ablation:14s} b={abl_bits} nmse={summary[key]['nmse_mean']:.6e}",
                        flush=True,
                    )

    elapsed = time.perf_counter() - started
    flags_summary = report["method_summary"][report["flagship"]]  # type: ignore[index]

    # Per-bit breakdown: the aggregate across a (2,3,4,8) sweep is dominated by 2-bit
    # saturation, which says nothing about the headline configuration.
    per_bit: dict[str, Any] = {}
    for bits in bits_list:
        subset = [r for r in records if r.bits == bits]
        if not subset:
            continue
        sub_summary = aggregate(subset)
        sub_gate = None
        if report["flagship"] in sub_summary:
            sub_gate = gate_report(sub_summary)
        per_bit[str(bits)] = {
            "method_summary": sub_summary,
            "n_cells": len(subset),
            **({"gate": sub_gate} if sub_gate else {}),
        }

    headline_bits = cfg.bits if cfg.bits in bits_list else bits_list[0]
    headline = per_bit.get(str(headline_bits), {})
    full: dict[str, Any] = {
        "schema": "quantforge.benchmark/1",
        "config": cfg.to_dict(),
        "methods": methods,
        "bits_grid": bits_list,
        "models": model_list,
        "dataset_seeds": seed_list,
        "records": [r.as_dict() for r in records],
        "n_cells": n_cells,
        "method_summary": report["method_summary"],
        "per_bit": per_bit,
        "headline_bits": headline_bits,
        "headline": headline,
        "baseline": report["baseline"],
        "flagship": report["flagship"],
        "relative_nmse_drop_pct": report["relative_nmse_drop_pct"],
        "significant": report["significant"],
        "gate": report["gate"],
        "flagship_nmse_mean": flags_summary["nmse_mean"],
        "flagship_nmse_std": flags_summary["nmse_std"],
        "baseline_nmse_mean": report["method_summary"][report["baseline"]]["nmse_mean"],  # type: ignore[index]
        "baseline_nmse_std": report["method_summary"][report["baseline"]]["nmse_std"],  # type: ignore[index]
        "ablations": ablations,
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "capabilities": capability_report(),
        },
        "elapsed_sec": round(elapsed, 3),
    }
    if outdir is not None:
        full["path"] = str(write_benchmark(full, outdir))
    return full


def write_benchmark(report: dict[str, Any], outdir: str | Path) -> Path:
    """Serialise the report to ``outdir/benchmark.json`` with stable key ordering.

    Only ``elapsed_sec`` and ``wall_sec`` are wall-clock dependent; every accuracy
    number is a pure function of the seed, so two runs must agree bit for bit.
    """
    target_dir = Path(outdir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "benchmark.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return path


def load_benchmark(path: str | Path) -> dict[str, Any]:
    """Read a previously written report back."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stable_core(report: dict[str, Any]) -> dict[str, Any]:
    """Extract the portion used for the determinism check.

    Drops ``elapsed_sec`` (wall clock), ``env`` (host-specific python version and
    platform string) and each record's ``wall_sec``. Sequence-valued config fields are
    normalised to lists because JSON round-trips tuples as lists, so a replayed run
    would otherwise look different purely from serialisation. Everything describing
    *results* -- accuracy numbers, summaries, gate, ablations -- is kept and must match
    bit for bit between runs.
    """
    drop = {"elapsed_sec", "env", "path"}
    core = {k: v for k, v in report.items() if k not in drop}
    if "records" in core:
        core["records"] = [{k: v for k, v in r.items() if k != "wall_sec"} for r in core["records"]]
    config = core.get("config")
    if isinstance(config, dict):
        core["config"] = {
            k: (sorted(v) if isinstance(v, (list, tuple)) else v) for k, v in config.items()
        }
    return core


def records_without_timing(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Records with per-cell wall time removed."""
    out = []
    for rec in report.get("records", []):
        out.append({k: v for k, v in rec.items() if k != "wall_sec"})
    return out
