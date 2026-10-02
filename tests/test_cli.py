"""CLI surface, pipeline wiring and the determinism guarantee."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from quantforge.cli import build_parser, main
from quantforge.core import PipelineError, load_config
from quantforge.pipeline.runner import (
    load_benchmark,
    records_without_timing,
    run_pipeline,
    stable_core,
    write_benchmark,
)

_ROOT = Path(__file__).resolve().parents[1]
#: directory that *contains* the ``quantforge`` package -- what a child needs on sys.path
_IMPORT_ROOT = _ROOT.parent


def _subprocess_env() -> dict[str, str]:
    """Environment that lets a child process import the package from the repo root."""
    import os

    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{_IMPORT_ROOT}{os.pathsep}{existing}" if existing else str(_IMPORT_ROOT)
    # keep BLAS single-threaded in the child so its results match ours bit for bit
    env.setdefault("OMP_NUM_THREADS", "1")
    return env


def _tiny_config(outdir: str):
    return load_config(
        {
            "seed": 42,
            "bits": 4,
            "n_calib": 64,
            "calib_samples": 64,
            "models": ("m0",),
            "dataset_seeds": (7,),
            "methods": ("rtn_tensor", "rtn_group", "awq", "quantforge"),
            "bits_grid": (4,),
            "group_grid": (32,),
            "outdir": outdir,
        }
    )


def test_parser_exposes_all_subcommands() -> None:
    parser = build_parser()
    for cmd in ("demo", "verify", "info", "methods"):
        assert cmd in parser.format_help()


def test_info_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "numpy" in payload and "python" in payload


def test_methods_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["methods"]) == 0
    out = capsys.readouterr().out
    for name in ("rtn_tensor", "rtn_group", "awq", "quantforge", "no_hessian"):
        assert name in out


def test_demo_writes_benchmark(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    outdir = tmp_path / "res"
    assert main(["demo", "--outdir", str(outdir), "--n-calib", "64", "--no-ablations"]) == 0
    path = outdir / "benchmark.json"
    assert path.exists()
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["schema"] == "quantforge.benchmark/1"
    assert report["n_cells"] > 0
    assert "method_summary" in report and "gate" in report
    assert set(report["method_summary"]) == {"rtn_tensor", "rtn_group", "awq", "quantforge"}
    assert "QuantForge benchmark" in capsys.readouterr().out


def test_demo_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["demo", "--outdir", str(tmp_path / "r"), "--n-calib", "64", "--no-ablations", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "quantforge.benchmark/1"


def test_verify_detects_missing_benchmark(tmp_path: Path) -> None:
    assert main(["verify", "--outdir", str(tmp_path / "empty")]) == 2


def test_verify_passes_on_reproducible_run(tmp_path: Path) -> None:
    outdir = tmp_path / "res"
    assert main(["demo", "--outdir", str(outdir), "--n-calib", "64", "--no-ablations"]) == 0
    assert main(["verify", "--outdir", str(outdir), "--n-calib", "64"]) == 0


def test_pipeline_is_bit_identical_across_runs(tmp_path: Path) -> None:
    cfg = _tiny_config(str(tmp_path))
    a = run_pipeline(cfg, with_ablations=False)
    b = run_pipeline(cfg, with_ablations=False)
    assert records_without_timing(a) == records_without_timing(b)
    assert json.dumps(stable_core(a), sort_keys=True) == json.dumps(stable_core(b), sort_keys=True)


def test_stable_core_drops_only_nondeterministic_fields() -> None:
    report = {
        "a": 1,
        "elapsed_sec": 3.0,
        "env": {"x": 1},
        "path": "p",
        "records": [{"nmse": 0.1, "wall_sec": 0.5}],
    }
    core = stable_core(report)
    assert core == {"a": 1, "records": [{"nmse": 0.1}]}


def test_write_and_load_benchmark_roundtrip(tmp_path: Path) -> None:
    report = {"schema": "quantforge.benchmark/1", "records": [], "elapsed_sec": 1.0}
    path = write_benchmark(report, tmp_path)
    assert json.loads(path.read_text(encoding="utf-8")) == report
    assert load_benchmark(path) == report


def test_benchmark_json_is_sorted_for_diffability(tmp_path: Path) -> None:
    report = {"zebra": 1, "alpha": 2}
    text = write_benchmark(report, tmp_path).read_text(encoding="utf-8")
    assert text.index('"alpha"') < text.index('"zebra"')


def test_pipeline_rejects_empty_methods(tmp_path: Path) -> None:
    """An empty method list is rejected at config validation, before any work happens."""
    from quantforge.core import ValidationError

    with pytest.raises(ValidationError):
        load_config({"methods": (), "n_calib": 64, "calib_samples": 64})
    cfg = _tiny_config(str(tmp_path))
    with pytest.raises(PipelineError):
        run_pipeline(cfg, methods=[], with_ablations=False)


def test_ablations_are_reported(tmp_path: Path) -> None:
    cfg = _tiny_config(str(tmp_path))
    report = run_pipeline(cfg, with_ablations=True)
    assert set(report["ablations"]) == {
        "no_hessian",
        "no_act_aware",
        "no_act_order",
        "no_mse_clip",
    }
    for stats in report["ablations"].values():
        assert stats["nmse_mean"] > 0 and stats["n"] == 1.0


def test_records_carry_required_fields(tmp_path: Path) -> None:
    cfg = _tiny_config(str(tmp_path))
    report = run_pipeline(cfg, with_ablations=False)
    for rec in report["records"]:
        for key in (
            "dataset_seed",
            "model",
            "bits",
            "group_size",
            "method",
            "nmse",
            "act_err",
            "sqnr_db",
            "bits_per_weight",
            "compression_ratio",
        ):
            assert key in rec


def test_records_are_json_serialisable_without_numpy_types(tmp_path: Path) -> None:
    cfg = _tiny_config(str(tmp_path))
    report = run_pipeline(cfg, with_ablations=False)
    json.dumps(report["records"])  # must not raise on np.float64


@pytest.mark.parametrize("test_id", ["module", "demo"])
def test_subprocess_env_is_importable(test_id: str) -> None:
    """The helper must actually make `import quantforge` work in a child process."""
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-c", "import quantforge; print(quantforge.__version__)"],
        cwd=str(_IMPORT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=_subprocess_env(),
    )
    assert proc.returncode == 0, proc.stderr


def test_module_entrypoint_runs() -> None:
    """`python -m quantforge --help` must work from the repository root."""
    proc = subprocess.run(
        [sys.executable, "-m", "quantforge", "--help"],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=_subprocess_env(),
    )
    assert proc.returncode == 0, proc.stderr
    assert "demo" in proc.stdout


def test_demo_entrypoint_quick_run(tmp_path: Path) -> None:
    """The shipped example must run end to end."""
    proc = subprocess.run(
        [sys.executable, str(_ROOT / "examples" / "run_demo.py"), "--quick", "--outdir", str(tmp_path / "d")],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env=_subprocess_env(),
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "determinism: records identical=True" in proc.stdout
