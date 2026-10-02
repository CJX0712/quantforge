# QuantForge

[![CI](https://github.com/CJX0712/quantforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/quantforge/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/CJX0712/quantforge?label=Release)](https://github.com/CJX0712/quantforge/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Quality](https://img.shields.io/badge/tests-186%20passed-brightgreen.svg)](#testing)
[![Coverage](https://img.shields.io/badge/coverage-78.8%25-brightgreen.svg)](#testing)

**Post-training quantization (PTQ) for weight compression** — RTN, AWQ and a GPTQ-based
flagship, implemented from the papers in NumPy with no deep-learning framework, plus a
seeded benchmark harness that compares them under a fixed protocol.

Every number in `benchmark.json` is produced by the run that wrote it, and two runs on
the same machine are **bit-identical**.

---

## Why this exists

Quantization papers report accuracy on their own benchmarks with their own baselines,
which makes cross-paper comparison unreliable. QuantForge reimplements RTN, AWQ and
GPTQ from scratch and runs them through **one** harness, **one** metric definition and
**one** seed, so the comparison is apples-to-apples — and reports the result it actually
gets, including where its own flagship loses.

## Install

```bash
git clone https://github.com/CJX0712/quantforge.git
cd quantforge
pip install -e ".[dev]"
```

Requires Python 3.11+ (CI covers 3.12 and 3.13). The only runtime dependency is NumPy;
`scipy` is optional and the toolkit degrades cleanly without it.

## Quick start

```bash
python examples/run_demo.py --quick     # ~5s smoke run, includes the determinism check
python -m quantforge demo               # full grid -> results/benchmark.json
python -m quantforge verify             # assert the rerun is bit-identical
python -m quantforge info               # runtime capabilities
```

## The four methods

| key | method | what it adds |
|---|---|---|
| `rtn_tensor` | **A.** RTN, per-tensor min-max | the naive baseline |
| `rtn_group` | **B.** RTN, per-group + MSE-optimal clip | finer scales, optimal clipping |
| `awq` | **C.** AWQ + RTN | activation-aware per-channel scaling |
| `quantforge` | **D.** AWQ → GPTQ → LS scale refit | second-order error compensation |

Method D runs in three stages:

1. **AWQ scaling** — per-input-channel scales `s`, grid-searched. Exactly invertible
   (`W·diag(s) · diag(s)⁻¹·X = W·X`), so it folds into the previous layer for free.
2. **GPTQ** — error compensation using the Hessian `H = 2XXᵀ`, with act-order
   permutation and a two-step Cholesky factorisation.
3. **LS scale refit** — with the codes fixed, `s* = Σ(W·c)/Σ(c²)` in closed form.

Stage 3 is not cosmetic: GPTQ minimises the *output* error and deliberately trades
weight-domain accuracy for it, so its min-max scales are far from optimal for
`‖W − Ŵ‖²`. Refitting recovers weight NMSE without disturbing the code assignment.

## Results (4-bit, 96 cells)

Full grid: 4 models x 3 seeds x 4 bit widths x group sizes {128, 64} = **384 cells**,
**7.9 s**, ~15 ms/cell. Per-bit tables, ablations and failure analysis are in
[`docs/benchmark_report.md`](docs/benchmark_report.md).

| method | NMSE (mean+/-std) v | ActErr (mean+/-std) v | SQNR (dB) ^ |
|---|---|---|---|
| `rtn_tensor` (A) | 2.2527e-01 +/- 1.0e-01 | 2.2736e-01 +/- 1.2e-01 | 7.05 |
| `rtn_group` (B) | 8.9143e-02 +/- 2.5e-02 | 9.0514e-02 +/- 3.0e-02 | 10.73 |
| `awq` (C) | 2.2660e-01 +/- 1.0e-01 | 2.0622e-01 +/- 1.1e-01 | 7.02 |
| **`quantforge` (D)** | **8.4477e-02** +/- 2.3e-02 | **5.9918e-02** +/- 1.5e-02 | **10.95** |

| comparison | NMSE | ActErr |
|---|---|---|
| D vs strongest baseline (`rtn_group`) | **+5.23%** | **+33.80%** |
| D vs naive RTN per-tensor (equal granularity) | **+62.50%** | -- |

Output-axis margin is significant at 4 bits (+33.80%); the weight-axis margin (+5.23%)
is not statistically significant by the 0.5-sigma-pooled test.

**Per-bit, D vs the strongest baseline at that width:**

| bits | NMSE | ActErr | note |
|---|---|---|---|
| 2 | +3.76% | +9.72% | 3 usable levels; everything collapses (SQNR ~2 dB) |
| 3 | +10.29% | +34.38% | largest output margin of any width |
| 4 | +5.23% | +33.80% | headline |
| 8 | **-167.86%** | -109.10% | **per-tensor wins** -- see below |

### Read this before quoting a number

**The flagship does not win on weight NMSE, and cannot.** GPTQ minimises
`||(W-W')X||^2`; `nmse` is `||W-W'||^2/||W||^2`. These provably diverge -- GPTQ moves weight error
into directions the activations do not excite, which *is* the mechanism. The
`no_hessian` ablation (AWQ + LS refit, no GPTQ) attains the **best weight NMSE of any
variant** (7.697e-02, -8.9% vs the flagship) while its ActErr is 20.7% worse. Judge the
flagship on `act_err`, or use `no_hessian` when weight fidelity matters. The
invariant the suite actually asserts is the one GPTQ is *defined* by:
`||(W-W'_gptq)X||^2 <= ||(W-W'_rtn)X||^2`, which holds in every cell.

**At 8 bits the flagship loses, and that is the correct result.** A single per-tensor
scale already reaches NMSE 7.5e-04 (32 dB); adding 127 per-group fp16 scales and 15 group
boundaries costs more than the finer resolution gains, so every per-group method is
~2.7x worse than per-tensor. Do not use group-wise quantization at 8 bits.

**The DoD gate (>=20% on both axes) is reported as FAIL**, not massaged into a pass. The
output axis exceeds it; the weight axis reaches 5.23% because grading GPTQ on a
weight-domain metric contradicts its objective. `benchmark.json` carries a `gate`
object with both axes so the choice is explicit. Section 5 of the benchmark report
states exactly what would be required to pass it, and why none of those routes were
taken silently.

Two further honest negatives, both measured: **act-order is a net negative here**
(removing it improves both axes, -2.6% NMSE / -0.8% ActErr) because the synthetic
Hessian has a mild `diag(H)` spread, and **MSE-optimal clipping is the real
workhorse** (removing it costs +130% NMSE, dwarfing every algorithmic contribution).

All results are on **synthetic** workloads, not trained language models. The activation
outliers are realistic in shape but not in value (weight kurtosis ~28 vs ~5-10 for real
LLM weights), so absolute values will not transfer to a real checkpoint.

## Library use

```python
import numpy as np
from quantforge import QuantConfig, SyntheticMLP, build_quantizer
from quantforge.eval import nmse, act_err, sqnr_db

bundle = SyntheticMLP("m2", n_calib=512).build(seed=7)
w, x_calib = bundle.layer(0)

result = build_quantizer("quantforge", QuantConfig(bits=4, group_size=128)).quantize(w, x_calib)

print(nmse(w, result.w_hat), act_err(w, result.w_hat, x_calib), sqnr_db(w, result.w_hat))
print(result.codes.shape, result.scales.shape, result.bits_per_weight)
```

Every quantizer returns integer codes, per-group fp16 scales and a dequantized matrix
in the **caller's coordinates** — a drop-in replacement for `w`.

## Documentation

| Document | Contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Layering, algorithms, implementation traps, error codes |
| [`docs/model_card.md`](docs/model_card.md) | Metrics, bit-width formulas, protocol, ablations, **limitations** |
| [`docs/benchmark_report.md`](docs/benchmark_report.md) | Per-bit tables, ablations, failure-case analysis |

## Testing

```
186 tests, 78.8% coverage (statement + branch), ruff clean
```

The suite is invariant-driven rather than assertion-driven. The guards that matter:

| invariant | catches |
|---|---|
| `‖UᵀU·H_damped − I‖ / d_in < 1e-10` | wrong factorisation order |
| `gptq(Ŵ) == Ŵ` (requantization idempotency) | group off-by-one |
| `Q[:, inv][:, perm] == Q` | missing act-order restoration |
| `gptq(X·√c) == gptq(X)` bit-for-bit | absolute instead of relative damping |
| `‖(W−Ŵ_gptq)X‖² ≤ ‖(W−Ŵ_rtn)X‖²` | GPTQ not actually optimising its objective |
| `column_owner` vs `group_bounds` | fractional `linspace` group edges |

```bash
make test     # pytest
make cov      # with coverage
make lint     # ruff
make demo     # full benchmark
make verify   # determinism check
```

## Implementation notes

Five bugs found and fixed during development that a naive reading of the papers would
have shipped:

1. `np.linalg.cholesky` returns the **input matrix in the opposite triangle**. Using it
   as an upper-triangular factor without `triu()` injects the Hessian into every
   compensation coefficient.
2. The error-propagation ratio is `U_ij / U_ii` (the paper's form). The
   coordinate-descent ratio `G_ij / G_ii` agrees only when `d_in == 2`, and degrades
   quality on wider layers.
3. `W1[:, t]` is a **view**: the compensation update mutates the column before its error
   is read. Every column read needs `.copy()`.
4. Deriving group boundaries from `np.linspace(0, d_in, n_groups+1)` gives **fractional
   edges** when the group size does not divide `d_in`, assigning columns to the wrong
   group. The owner mapping must come from the same list used to slice.
5. AWQ and GPTQ return weights in the **scaled coordinate system**. Without dividing by
   `s` the result is not a valid replacement for `W`.

## License

MIT — see [LICENSE](LICENSE).

## References

- Frantar et al., *GPTQ: Accurate Post-training Compression for LLMs*, ICLR 2023
- Frantar et al., *Optimal Brain Compression*, ICLR 2022
- Lin et al., *AWQ: Activation-aware Weight Quantization*, MLSys 2024
- Chen et al., *ZeroQuant*, NeurIPS 2022
- Bennett, *Rational Quantizers*, IRE Trans. Inf. Theory 1948
- Max, *Quantizing for Minimum Mean Square Error*, IRE Trans. Inf. Theory 1960

---

Author: 晨星 (Chenxing) · [github.com/CJX0712/quantforge](https://github.com/CJX0712/quantforge)