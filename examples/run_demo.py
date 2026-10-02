"""End-to-end demo: run the benchmark, print a table, verify determinism.

    python examples/run_demo.py                # full grid
    python examples/run_demo.py --quick         # 2 models x 1 seed, no ablations
    python examples/run_demo.py --no-verify     # skip the second (determinism) run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
# repository root = the directory that *contains* the ``quantforge`` package
for _candidate in (_HERE.parents[1], _HERE.parents[2]):
    if (_candidate / "quantforge" / "__init__.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from quantforge.core.config import load_config  # noqa: E402
from quantforge.core.seed import pin_blas_threads, set_all  # noqa: E402
from quantforge.pipeline.runner import (  # noqa: E402
    load_benchmark,
    records_without_timing,
    run_pipeline,
    stable_core,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="QuantForge demo")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--quick", action="store_true", help="tiny grid for a fast smoke run")
    ap.add_argument("--no-verify", action="store_true", help="skip the determinism re-run")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    pin_blas_threads(1)
    set_all(42)

    n_calib = 256
    overrides = {"seed": 42, "bits": 4, "n_calib": n_calib, "calib_samples": n_calib}
    overrides["models"] = ("m0", "m1") if args.quick else ("m0", "m1", "m2", "m3")
    overrides["dataset_seeds"] = (7,) if args.quick else (7, 13, 23)
    overrides["bits_grid"] = (4,) if args.quick else (2, 3, 4, 8)
    overrides["group_grid"] = (32,) if args.quick else (128, 64)
    cfg = load_config(overrides)

    print(f"QuantForge demo  (quick={args.quick}, outdir={args.outdir})")
    report = run_pipeline(cfg, outdir=args.outdir, progress=args.progress, with_ablations=not args.quick)
    _print_table(report)

    rc = 0
    if not args.no_verify:
        # Compare like with like: the first run may include ablations, so the second run
        # repeats the same configuration rather than a reduced one.
        path = Path(args.outdir) / "benchmark.json"
        first = load_benchmark(path)
        second = run_pipeline(cfg, with_ablations=bool(report.get("ablations")))
        same = records_without_timing(first) == records_without_timing(second)
        full = json.dumps(stable_core(first), sort_keys=True) == json.dumps(stable_core(second), sort_keys=True)
        print(f"\ndeterminism: records identical={same}  full report identical={full}")
        if not same:
            rc = 1
        if not full:
            a, b = stable_core(first), stable_core(second)
            for key in sorted(set(a) | set(b)):
                if a.get(key) != b.get(key):
                    print(f"    differing key: {key}")
            rc = 1
    return rc


def _print_table(report: dict) -> None:
    summary = report["method_summary"]
    base = report["baseline"]
    flags = report["flagship"]
    print("\n" + "=" * 96)
    print(f"{'method':<26}{'NMSE (mean±std)':>26}{'ActErr (mean±std)':>26}{'SQNR dB':>12}")
    print("-" * 96)
    for name in sorted(summary):
        s = summary[name]
        mark = " *" if name in (base, flags) else "  "
        print(
            f"{mark}{name:<24}{s['nmse_mean']:.6e} ± {s['nmse_std']:.1e}"
            f"   {s['act_err_mean']:.6e} ± {s['act_err_std']:.1e}{s['sqnr_db_mean']:>12.2f}"
        )
    print("-" * 96)
    g = report["gate"]
    print(f"strongest baseline: {base}    flagship: {flags}")
    print(
        f"NMSE   vs baseline {g['nmse_vs_baseline_pct']:+7.2f}%   "
        f"ActErr vs baseline {g['act_err_vs_baseline_pct']:+7.2f}%"
    )
    if "nmse_vs_rtn_tensor_pct" in g:
        print(f"NMSE   vs naive RTN per-tensor {g['nmse_vs_rtn_tensor_pct']:+7.2f}%")
    print(f"DoD gate (>= {g['threshold_pct']:.0f}% on both axes): {'PASS' if g['passed'] else 'FAIL'}")
    if report.get("ablations"):
        print("\nablations (flagship NMSE reference):")
        f = summary[flags]["nmse_mean"]
        for name, s in report["ablations"].items():
            print(
                f"  {name:<14} NMSE {s['nmse_mean']:.6e} ({100*(s['nmse_mean']-f)/f:+7.1f}%)"
                f"   ActErr {s['act_err_mean']:.6e}"
            )
    print(f"\ncells={report['n_cells']}  elapsed={report['elapsed_sec']:.1f}s  -> {report.get('path')}")
    print("=" * 96)


if __name__ == "__main__":
    raise SystemExit(main())
