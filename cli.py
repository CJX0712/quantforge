"""Command line interface: ``python -m quantforge`` / ``quantforge`` console script.

Subcommands
-----------
``demo``      run the benchmark and write ``benchmark.json``
``verify``    re-run and assert the determinism guarantee
``info``      print the runtime capability report
``methods``   list the available methods and ablations
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .core.config import SCHEMA, load_config
from .core.errors import QuantForgeError, error_code_of
from .core.seed import pin_blas_threads, set_all
from .pipeline.runner import (
    load_benchmark,
    records_without_timing,
    run_pipeline,
    stable_core,
)

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantforge",
        description="QuantForge -- post-training quantization research toolkit",
    )
    parser.add_argument("--version", action="version", version=f"quantforge {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the benchmark grid and write benchmark.json")
    demo.add_argument("--outdir", default="results", help="output directory (default: results)")
    demo.add_argument("--bits", type=int, default=4, help="headline bit width")
    demo.add_argument("--seed", type=int, default=42, help="master RNG seed")
    demo.add_argument("--n-calib", type=int, default=512, help="calibration samples per layer")
    demo.add_argument(
        "--models", default="m0,m1,m2,m3", help="comma-separated model list (default: all)"
    )
    demo.add_argument(
        "--dataset-seeds", default="7,13,23", help="comma-separated dataset seeds (default: 7,13,23)"
    )
    demo.add_argument(
        "--methods", default="rtn_tensor,rtn_group,awq,quantforge", help="comma-separated methods"
    )
    demo.add_argument("--bits-grid", default="4", help="comma-separated bit widths to sweep")
    demo.add_argument("--group-grid", default="32", help="comma-separated group sizes")
    demo.add_argument("--no-ablations", action="store_true", help="skip the ablation study")
    demo.add_argument("--progress", action="store_true", help="print每 cell as it completes")
    demo.add_argument("--json", action="store_true", help="print the full report to stdout")

    verify = sub.add_parser("verify", help="re-run and check bit-identical determinism")
    verify.add_argument("--outdir", default="results", help="directory holding benchmark.json")
    verify.add_argument("--bits", type=int, default=4)
    verify.add_argument("--seed", type=int, default=42)
    verify.add_argument("--n-calib", type=int, default=512)
    verify.add_argument("--bits-grid", default="4")
    verify.add_argument("--group-grid", default="32")

    sub.add_parser("info", help="print runtime capabilities")
    sub.add_parser("methods", help="list methods and ablations")
    return parser


def _split_ints(text: str) -> list[int]:
    return [int(p) for p in str(text).replace(";", ",").split(",") if p.strip()]


def _split_strs(text: str) -> list[str]:
    return [p.strip() for p in str(text).replace(";", ",").split(",") if p.strip()]


def _cmd_demo(args: argparse.Namespace) -> int:
    n_calib = args.n_calib
    # calib_samples defaults to 256 but must never exceed the calibration set size
    overrides: dict[str, object] = {
        "seed": args.seed,
        "bits": args.bits,
        "n_calib": n_calib,
        "calib_samples": min(n_calib, 256),
        "models": tuple(_split_strs(args.models)),
        "dataset_seeds": tuple(_split_ints(args.dataset_seeds)),
        "methods": tuple(_split_strs(args.methods)),
        "bits_grid": tuple(_split_ints(args.bits_grid)),
        "group_grid": tuple(_split_ints(args.group_grid)),
        "outdir": args.outdir,
    }
    cfg = load_config(overrides)
    report = run_pipeline(
        cfg,
        outdir=args.outdir,
        progress=bool(args.progress),
        with_ablations=not args.no_ablations,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        _print_summary(report)
    return 0


def _print_summary(report: dict) -> None:
    summary = report["method_summary"]
    base_key = report["baseline"]
    print("\nQuantForge benchmark")
    print("=" * 78)
    print(f"{'method':<24}{'NMSE mean±std':>26}{'ActErr mean±std':>26}")
    print("-" * 78)
    for name in sorted(summary):
        s = summary[name]
        mark = " *" if name in (base_key, report["flagship"]) else "  "
        print(
            f"{mark}{name:<22}"
            f"{s['nmse_mean']:.6e}±{s['nmse_std']:.1e}"
            f"{s['act_err_mean']:.6e}±{s['act_err_std']:.1e}"
        )
    print("-" * 78)
    print(f"  * strongest baseline = {base_key}; flagship = {report['flagship']}")
    gate = report["gate"]
    print(
        f"  NMSE   vs strongest baseline: {gate['nmse_vs_baseline_pct']:+.2f}%"
        f"   ActErr vs strongest baseline: {gate['act_err_vs_baseline_pct']:+.2f}%"
    )
    if "nmse_vs_rtn_tensor_pct" in gate:
        print(f"  NMSE   vs naive RTN per-tensor: {gate['nmse_vs_rtn_tensor_pct']:+.2f}%")
    print(f"  DoD gate (>= {gate['threshold_pct']:.0f}% on both axes): "
          f"{'PASS' if gate['passed'] else 'FAIL'}")
    if report.get("ablations"):
        print("\n  ablations (mean NMSE / mean ActErr)")
        base_n = summary[report["flagship"]]["nmse_mean"]
        for name, s in report["ablations"].items():
            print(
                f"    {name:<14}{s['nmse_mean']:.6e}  {s['act_err_mean']:.6e}"
                f"   (NMSE {100.0 * (s['nmse_mean'] - base_n) / base_n:+.1f}% vs flagship)"
            )
    print(f"\n  cells={report['n_cells']}  elapsed={report['elapsed_sec']:.1f}s")
    print(f"  report written to {report.get('path', '<memory>')}")


def _cmd_verify(args: argparse.Namespace) -> int:
    path = Path(args.outdir) / "benchmark.json"
    if not path.exists():
        print(f"no benchmark at {path}; run `quantforge demo` first", file=sys.stderr)
        return 2
    first = load_benchmark(path)
    # Replay the *recorded* configuration exactly; reconstructing it from a subset of
    # fields would silently compare two different experiments.
    prev = dict(first.get("config", {}))
    prev.setdefault("seed", args.seed)
    prev.setdefault("bits", args.bits)
    prev.setdefault("n_calib", args.n_calib)
    prev["models"] = tuple(first["models"])
    prev["dataset_seeds"] = tuple(first["dataset_seeds"])
    prev["methods"] = tuple(first["methods"])
    prev["bits_grid"] = tuple(first["bits_grid"])
    prev["group_grid"] = tuple(first.get("config", {}).get("group_grid", (128,)))
    prev["calib_samples"] = min(int(prev.get("calib_samples", 256)), int(prev["n_calib"]))
    unknown = sorted(set(prev) - set(SCHEMA))
    if unknown:
        print(f"cannot replay: unknown config keys {unknown}", file=sys.stderr)
        return 2
    cfg = load_config(prev)
    # Replay the same scope as the recorded run: a full report contains the ablation
    # study, and comparing it against an ablation-free rerun would report a false diff.
    second = run_pipeline(cfg, with_ablations=bool(first.get("ablations")))
    same_records = records_without_timing(first) == records_without_timing(second)
    same_core = json.dumps(stable_core(first), sort_keys=True) == json.dumps(
        stable_core(second), sort_keys=True
    )
    print("determinism check")
    print(f"  records identical (ignoring wall_sec): {same_records}")
    print(f"  full report identical (ignoring timing): {same_core}")
    if not same_records:
        a = {(r["model"], r["dataset_seed"], r["method"]): r["nmse"] for r in records_without_timing(first)}
        b = {(r["model"], r["dataset_seed"], r["method"]): r["nmse"] for r in records_without_timing(second)}
        for key in sorted(set(a) | set(b)):
            if a.get(key) != b.get(key):
                print(f"    mismatch {key}: {a.get(key)} vs {b.get(key)}")
    return 0 if same_records else 1


def _cmd_info(_: argparse.Namespace) -> int:
    from .quant.selectors import capability_report

    pin_blas_threads(1)
    set_all(0)
    print(json.dumps(capability_report(), indent=2, sort_keys=True))
    return 0


def _cmd_methods(_: argparse.Namespace) -> int:
    from .quant.schemes import BASELINES, FLAGSHIP, METHODS

    print("methods:")
    for m in METHODS:
        tag = []
        if m in BASELINES:
            tag.append("baseline")
        if m == FLAGSHIP:
            tag.append("flagship")
        print(f"  {m:<14}{' '.join(tag)}")
    print("\nablations (applied to the flagship):")
    for a in ("no_hessian", "no_act_aware", "no_act_order", "no_mse_clip", "no_ls_refit"):
        print(f"  {a}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code; never raises for expected errors."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            return _cmd_demo(args)
        if args.command == "verify":
            return _cmd_verify(args)
        if args.command == "info":
            return _cmd_info(args)
        if args.command == "methods":
            return _cmd_methods(args)
    except QuantForgeError as exc:
        print(f"error {error_code_of(exc)}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
