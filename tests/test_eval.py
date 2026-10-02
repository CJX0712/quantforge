"""Metric definitions and benchmark aggregation."""

from __future__ import annotations

import numpy as np
import pytest
from quantforge.core import MetricError, QuantConfig, RunRecord
from quantforge.data import SyntheticMLP
from quantforge.eval import NMSE, SQNR, ActErr, act_err, all_metrics, compression_ratio, nmse, sqnr_db
from quantforge.eval.benchmark import (
    aggregate,
    gate_report,
    run_cell,
    significance,
    strongest_baseline,
    summarise,
)


def test_nmse_is_zero_for_identical_weights(rng: np.random.Generator) -> None:
    w = rng.standard_normal((4, 8))
    assert nmse(w, w) == pytest.approx(0.0, abs=1e-30)


def test_nmse_matches_definition(rng: np.random.Generator) -> None:
    w = rng.standard_normal((4, 8))
    wh = w + 0.1
    expected = float(np.sum((w - wh) ** 2) / np.sum(w**2))
    assert nmse(w, wh) == pytest.approx(expected)


def test_sqnr_is_minus_ten_log10_nmse(rng: np.random.Generator) -> None:
    w = rng.standard_normal((4, 8))
    wh = w + 0.1
    assert sqnr_db(w, wh) == pytest.approx(-10 * np.log10(nmse(w, wh)))


def test_act_err_matches_definition(rng: np.random.Generator) -> None:
    w = rng.standard_normal((4, 8))
    x = rng.standard_normal((8, 32))
    wh = w + 0.05
    ref = w @ x
    expected = float(np.sum((wh @ x - ref) ** 2) / np.sum(ref**2))
    assert act_err(w, wh, x) == pytest.approx(expected)


def test_zero_error_gives_infinite_sqnr(rng: np.random.Generator) -> None:
    w = rng.standard_normal((4, 8))
    assert sqnr_db(w, w) == float("inf")


def test_all_zero_reference_is_infinite_not_crash() -> None:
    z = np.zeros((4, 8))
    assert nmse(z, z) == float("inf")
    assert sqnr_db(z, z) == float("-inf")


def test_act_err_requires_matching_calibration(rng: np.random.Generator) -> None:
    w = rng.standard_normal((4, 8))
    with pytest.raises(MetricError):
        act_err(w, w, np.zeros((3, 10)))


def test_metrics_reject_shape_mismatch(rng: np.random.Generator) -> None:
    with pytest.raises(MetricError):
        nmse(np.zeros((4, 8)), np.zeros((4, 9)))


def test_act_err_metric_class_requires_calibration(rng: np.random.Generator) -> None:
    w = rng.standard_normal((4, 8))
    with pytest.raises(MetricError):
        ActErr()(w, w, None)


def test_metric_classes_agree_with_functions(rng: np.random.Generator) -> None:
    w = rng.standard_normal((4, 8))
    x = rng.standard_normal((8, 16))
    wh = w + 0.02
    assert NMSE()(w, wh) == pytest.approx(nmse(w, wh))
    assert ActErr()(w, wh, x) == pytest.approx(act_err(w, wh, x))
    assert SQNR()(w, wh) == pytest.approx(sqnr_db(w, wh))


def test_metric_direction_flags() -> None:
    assert NMSE.lower_is_better and ActErr.lower_is_better
    assert not SQNR.lower_is_better


def test_compression_ratio() -> None:
    assert compression_ratio(4.0) == pytest.approx(8.0)
    with pytest.raises(MetricError):
        compression_ratio(0.0)


def test_all_metrics_bundle(rng: np.random.Generator) -> None:
    w = rng.standard_normal((4, 8))
    x = rng.standard_normal((8, 16))
    out = all_metrics(w, w + 0.1, x, bits_per_weight=4.125)
    assert set(out) == {"nmse", "sqnr_db", "act_err", "bits_per_weight", "compression_ratio"}


# ------------------------------------------------------------- aggregation


def _record(method: str, nmse_value: float, std: float = 0.0) -> RunRecord:
    return RunRecord(
        dataset_seed=7,
        model="m0",
        layer=0,
        bits=4,
        group_size=32,
        method=method,
        nmse=nmse_value,
        act_err=nmse_value / 2,
        sqnr_db=-10 * np.log10(nmse_value),
        bits_per_weight=4.0,
        compression_ratio=8.0,
        wall_sec=std,
    )


def test_aggregate_groups_by_method() -> None:
    summary = aggregate([_record("a", 0.1), _record("a", 0.3), _record("b", 0.2)])
    assert summary["a"]["n"] == 2.0
    assert summary["a"]["nmse_mean"] == pytest.approx(0.2)
    assert summary["a"]["nmse_std"] == pytest.approx(0.1)


def test_significance_requires_margin() -> None:
    tight = {"nmse_mean": 0.10, "nmse_std": 0.001}
    loose = {"nmse_mean": 0.11, "nmse_std": 0.001}
    assert significance(tight, loose) is True
    identical = {"nmse_mean": 0.10, "nmse_std": 0.0}
    assert significance(identical, identical) is False


def test_strongest_baseline_picks_lowest_error() -> None:
    summary = {"rtn_tensor": {"nmse_mean": 0.3, "nmse_std": 0.0}, "rtn_group": {"nmse_mean": 0.1, "nmse_std": 0.0}}
    assert strongest_baseline(summary) == "rtn_group"


def test_summarise_reports_both_axes() -> None:
    records = [_record("rtn_tensor", 0.4), _record("rtn_group", 0.2), _record("quantforge", 0.1)]
    report = summarise(records)
    assert report["baseline"] == "rtn_group"
    assert report["relative_nmse_drop_pct"] == pytest.approx(50.0)
    assert report["significant"] is True
    gate = report["gate"]
    assert gate["nmse_vs_baseline_pct"] == pytest.approx(50.0)
    assert gate["act_err_vs_baseline_pct"] == pytest.approx(50.0)
    assert gate["passed"] is True


def test_gate_fails_when_baseline_not_beaten() -> None:
    records = [_record("rtn_tensor", 0.1), _record("rtn_group", 0.2), _record("quantforge", 0.5)]
    report = summarise(records)
    assert report["gate"]["passed"] is False
    assert report["gate"]["nmse_vs_baseline_pct"] < 0


def test_gate_report_requires_threshold() -> None:
    summary = aggregate([_record("rtn_group", 0.2), _record("quantforge", 0.19)])
    gate = gate_report(summary, threshold_pct=5.0)
    assert gate["threshold_pct"] == 5.0


def test_run_cell_produces_valid_record() -> None:
    bundle = SyntheticMLP("m0", n_calib=64).build(7)
    rec = run_cell(
        bundle,
        layer=0,
        method="quantforge",
        config=QuantConfig(bits=4, group_size=32),
        dataset_seed=7,
        model="m0",
    )
    assert rec.nmse > 0 and np.isfinite(rec.nmse)
    assert rec.act_err > 0 and np.isfinite(rec.act_err)
    assert rec.wall_sec >= 0.0


def test_run_cell_labels_ablations() -> None:
    bundle = SyntheticMLP("m0", n_calib=64).build(7)
    rec = run_cell(
        bundle,
        layer=0,
        method="quantforge",
        config=QuantConfig(bits=4, group_size=32),
        dataset_seed=7,
        model="m0",
        ablation="no_hessian",
    )
    assert rec.method == "quantforge:no_hessian"
