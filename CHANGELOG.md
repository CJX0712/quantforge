# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-02

First release. Post-training quantization research toolkit: RTN, AWQ and a GPTQ-based
flagship, implemented from the papers in NumPy with no deep-learning framework.

### Added

**Quantizers** (`quant/`)
- `rounding.py` -- uniform affine quantization with `floor(x + 0.5)` round-half-up.
  Deliberately not `np.round` (round-half-to-even) and never `int()` (truncation toward
  zero, which corrupts asymmetric zero-points on the negative half axis).
- `scale.py` -- `per_tensor` / `per_channel` / `per_group` scales, MSE-optimal clip
  search over `alpha in [0.5, 1.0]`, and exact `bits_per_weight` accounting.
- `gptq.py` -- GPTQ with `H = 2XXᵀ`, act-order permutation, `static_groups`, two-step
  Cholesky inverse-Hessian factorisation, relative damping with a ×10 retry ladder, and
  an explicit warned RTN fallback.
- `awq.py` -- activation-aware per-input-channel scaling with a grid-searched exponent,
  clipped to `[1/3, 3]` so the previous layer's error is not amplified.
- `schemes.py` -- four methods (`rtn_tensor`, `rtn_group`, `awq`, `quantforge`) plus
  five ablations (`no_hessian`, `no_act_aware`, `no_act_order`, `no_mse_clip`,
  `no_ls_refit`).
- `selectors.py` -- optional-dependency probes and a JSON-serialisable capability report.

**Flagship pipeline** -- AWQ scaling -> GPTQ compensation -> closed-form least-squares
scale refit. The refit exists because GPTQ minimises the *output* error and therefore
leaves its scales far from optimal for the weight-domain error.

**Infrastructure**
- `core/` -- typed contracts, `E1xx`-`E5xx` error codes, `ENV_QF_*` config with schema
  validation, a single seeded RNG, and BLAS thread pinning.
- `data/synthetic.py` -- reproducible ReLU MLPs with sparse, high-gain activation
  outliers (the structure that makes activation-aware scaling non-trivial).
- `eval/` -- `nmse` / `act_err` / `sqnr_db` / `bits_per_weight` / `compression_ratio`,
  aggregation, and a two-axis DoD gate.
- `pipeline/runner.py` -- grid execution, per-bit breakdown, ablation study, and
  `benchmark.json` writing with sorted keys for diffability.
- `cli.py` -- `demo`, `verify`, `info`, `methods`.
- `Dockerfile`, `Makefile`, CI matrix (Python 3.12 + 3.13, lint + pytest + demo smoke).

**Documentation**
- `docs/architecture.md` -- layering, algorithm derivations, the trap table, error codes.
- `docs/model_card.md` -- metrics, bit-width formulas, protocol, ablations, limitations.
- `docs/benchmark_report.md` -- per-bit tables, ablation study, four failure cases.

### Numerics

- `SAFETY_TARGET = 3.0`: damping is `lam = mean(diag(H)) / SAFETY_TARGET`, so the safety
  ratio is exact by construction and independent of `d_in` and of the Hessian spectrum.
  Chosen by sweeping outlier gain 1x-500x (spread 5-2766); `S <= 3` keeps the flagship's
  edge over RTN positive in every regime, while `S = 20` collapses to -71.6% at
  spread ~116.

### Testing

186 tests, 86.5% branch coverage, `ruff` clean. The suite is invariant-driven; the
guards that caught real bugs during development:

| invariant | catches |
|---|---|
| `‖UᵀU·H_damped − I‖ / d_in < 1e-10` | wrong factorisation order (`UUᵀ` vs `UᵀU`) |
| `gptq(Ŵ) == Ŵ` | group off-by-one in the trailing group |
| `Q[:, inv][:, perm] == Q` | missing act-order restoration |
| `gptq(X·√c) == gptq(X)` bit-for-bit | absolute instead of relative damping |
| `‖(W−Ŵ_gptq)X‖² ≤ ‖(W−Ŵ_rtn)X‖²` | GPTQ not optimising its own objective |
| `U = I ⇒ gptq == rtn` bit-for-bit | a broken control group masquerading as "no benefit" |
| `abs(mu/lam − SAFETY_TARGET)/SAFETY_TARGET < 1e-9` | damping variable wired to the wrong quantity |
| `column_owner == group_bounds` | fractional `linspace` group edges |

### Known limitations

Documented in full in `docs/model_card.md`. In short: synthetic workloads only; the
flagship wins on output error rather than weight NMSE by construction; 8-bit
group-wise quantization is a net loss; act-order is a net negative on this workload's
mildly-conditioned Hessian; determinism is guaranteed same-machine, not cross-machine.

[0.1.0]: https://github.com/CJX0712/quantforge/releases/tag/v0.1.0
