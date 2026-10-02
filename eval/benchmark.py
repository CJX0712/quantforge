"""Benchmark orchestration: run the grid, aggregate, judge significance."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence

import numpy as np

from ..core.errors import PipelineError
from ..core.interfaces import Evaluator
from ..core.seed import set_all
from ..core.types import LayerBundle, QuantConfig, RunRecord
from ..data.synthetic import SyntheticMLP
from ..quant.schemes import BASELINES, FLAGSHIP, build_quantizer
from .metrics import act_err, compression_ratio, nmse, sqnr_db

__all__ = ["LayerEvaluator", "aggregate", "run_cell", "significance", "summarise"]


class LayerEvaluator(Evaluator):
    """Evaluates a quantized layer against the fp32 reference."""

    def evaluate(self, bundle: LayerBundle, layer: int, w_hat: np.ndarray) -> dict[str, float]:
        w, x = bundle.layer(layer)
        out = {"nmse": nmse(w, w_hat, x), "act_err": act_err(w, w_hat, x), "sqnr_db": sqnr_db(w, w_hat, x)}
        return out


def run_cell(
    bundle: LayerBundle,
    layer: int,
    method: str,
    config: QuantConfig,
    *,
    dataset_seed: int,
    model: str,
    ablation: str | None = None,
    evaluator: Evaluator | None = None,
) -> RunRecord:
    """Quantize one layer with one method and score it."""
    w, x = bundle.layer(layer)
    quantizer = build_quantizer(method, config, ablation=ablation)
    started = time.perf_counter()
    result = quantizer.quantize(w, x)
    elapsed = time.perf_counter() - started
    ev = evaluator or LayerEvaluator()
    scores = ev.evaluate(bundle, layer, result.w_hat)
    return RunRecord(
        dataset_seed=dataset_seed,
        model=model,
        layer=layer,
        bits=config.bits,
        group_size=config.group_size,
        method=method if ablation is None else f"{method}:{ablation}",
        nmse=scores["nmse"],
        act_err=scores["act_err"],
        sqnr_db=scores["sqnr_db"],
        bits_per_weight=result.bits_per_weight,
        compression_ratio=compression_ratio(result.bits_per_weight),
        wall_sec=elapsed,
    )


def aggregate(records: Iterable[RunRecord], key: str = "method") -> dict[str, dict[str, float]]:
    """Group records by ``key`` and compute ``mean``/``std``/``n`` per metric."""
    buckets: dict[str, list[RunRecord]] = {}
    for rec in records:
        buckets.setdefault(getattr(rec, key), []).append(rec)
    out: dict[str, dict[str, float]] = {}
    for name, group in buckets.items():
        nmse_vals = np.array([r.nmse for r in group], dtype=np.float64)
        act_vals = np.array([r.act_err for r in group], dtype=np.float64)
        sqnr_vals = np.array([r.sqnr_db for r in group], dtype=np.float64)
        out[name] = {
            "nmse_mean": float(nmse_vals.mean()),
            "nmse_std": float(nmse_vals.std(ddof=0)),
            "act_err_mean": float(act_vals.mean()),
            "act_err_std": float(act_vals.std(ddof=0)),
            "sqnr_db_mean": float(sqnr_vals.mean()),
            "sqnr_db_std": float(sqnr_vals.std(ddof=0)),
            "n": float(len(group)),
        }
    return out


def significance(candidate: dict[str, float], baseline: dict[str, float], margin: float = 0.5) -> bool:
    """True when the candidate beats the baseline by more than ``margin`` pooled sigmas.

    ``mean_A - mean_m > margin * (std_A + std_m)`` -- a deliberately conservative gate, so
    a win reported here is unlikely to be sampling noise.
    """
    delta = baseline["nmse_mean"] - candidate["nmse_mean"]
    pooled = baseline["nmse_std"] + candidate["nmse_std"]
    return bool(delta > margin * pooled)


def gate_report(summary: dict[str, dict[str, float]], threshold_pct: float = 20.0) -> dict[str, object]:
    """Evaluate the DoD threshold on *both* reported metrics.

    GPTQ minimises the proxy output error ``‖(W-Ŵ)X‖²`` while the headline NMSE is a
    weight-domain quantity ``‖W-Ŵ‖²/‖W‖²``. These provably diverge -- the algorithm
    deliberately moves weight error into directions the activations do not excite -- so
    a single-number gate would be misleading. Both are reported; neither is hidden.

    Also reports the same comparison against the *naive* per-tensor RTN baseline, which
    isolates the contribution at equal granularity.
    """
    flags = summary[FLAGSHIP]
    base_key = strongest_baseline(summary)
    base = summary[base_key]
    naive = summary.get("rtn_tensor")
    out: dict[str, object] = {
        "threshold_pct": threshold_pct,
        "baseline": base_key,
        "nmse_vs_baseline_pct": 100.0 * (base["nmse_mean"] - flags["nmse_mean"]) / base["nmse_mean"],
        "act_err_vs_baseline_pct": 100.0
        * (base["act_err_mean"] - flags["act_err_mean"])
        / base["act_err_mean"],
        "nmse_significant": significance(flags, base),
        "act_err_significant": _act_significant(flags, base),
        "passed": False,
    }
    if naive is not None:
        out["nmse_vs_rtn_tensor_pct"] = 100.0 * (
            naive["nmse_mean"] - flags["nmse_mean"]
        ) / naive["nmse_mean"]
    out["passed"] = bool(
        out["nmse_vs_baseline_pct"] >= threshold_pct and out["act_err_vs_baseline_pct"] >= threshold_pct
    )
    return out


def _act_significant(candidate: dict[str, float], baseline: dict[str, float], margin: float = 0.5) -> bool:
    delta = baseline["act_err_mean"] - candidate["act_err_mean"]
    pooled = baseline["act_err_std"] + candidate["act_err_std"]
    return bool(delta > margin * pooled)


def strongest_baseline(summary: dict[str, dict[str, float]], baselines: Sequence[str] = BASELINES) -> str:
    """Pick the baseline with the lowest mean NMSE (ties broken by name for determinism)."""
    present = [b for b in baselines if b in summary]
    if not present:
        raise PipelineError("no baseline present in summary", available=sorted(summary))
    return min(present, key=lambda b: (summary[b]["nmse_mean"], b))


def summarise(records: list[RunRecord]) -> dict[str, object]:
    """Build the full report: per-method summary, baseline choice, flagship win."""
    summary = aggregate(records)
    base_key = strongest_baseline(summary)
    flags = summary.get(FLAGSHIP)
    if flags is None:
        raise PipelineError("flagship results missing from summary", methods=sorted(summary))
    base = summary[base_key]
    denom = base["nmse_mean"]
    drop = 100.0 * (denom - flags["nmse_mean"]) / denom if denom > 0 else 0.0
    return {
        "method_summary": summary,
        "baseline": base_key,
        "flagship": FLAGSHIP,
        "relative_nmse_drop_pct": float(drop),
        "significant": significance(flags, base),
        "gate": gate_report(summary),
    }


def build_sources(models: Iterable[str], n_calib: int) -> list[SyntheticMLP]:
    return [SyntheticMLP(model=m, n_calib=n_calib) for m in models]


def reseed(seed: int) -> None:
    set_all(seed)
