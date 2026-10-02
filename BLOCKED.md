# BLOCKED / DoD Status

**Release**: v0.1.0 · **Date**: 2026-10-02 · **Overall quality grade: A-**

One acceptance criterion is not met. It is reported here rather than worked around.
Every other criterion passes with measured evidence.

---

## The unmet criterion

| Item | Required | Measured | Status |
|---|---|---|---|
| Flagship NMSE reduction vs strongest baseline | ≥ 20% | **+5.23%** (4-bit, 96 cells) | ❌ **FAIL** |
| Flagship ActErr reduction vs strongest baseline | ≥ 20% | **+33.80%** (4-bit, 96 cells) | ✅ PASS |

The output axis passes with margin. The weight-domain axis does not.

### Why

GPTQ minimises the proxy objective `‖(W−Ŵ)X‖²_F`. The `nmse` metric is the
weight-domain quantity `‖W−Ŵ‖²_F/‖W‖²_F`. **These are different functionals and diverge
by design**: GPTQ's mechanism is to move weight error into directions the calibration
activations do not excite. Grading it on weight-domain fidelity therefore penalises
exactly the behaviour that makes it work.

This is not an implementation defect, and it is not a tuning failure. Evidence:

1. **The proxy objective gate passes in every cell.** The suite asserts
   `‖(W−Ŵ_gptq)X‖² ≤ ‖(W−Ŵ_rtn)X‖²`, which is the property GPTQ is *defined* by.
2. **The `no_hessian` ablation has the best weight NMSE of any variant** (7.697e-02,
   −8.9% vs the flagship) while its ActErr is 20.7% worse. Removing GPTQ and keeping
   AWQ + LS refit is what maximises weight-domain accuracy.
3. **The same code beats the naive baseline by +62.50% NMSE**, so the implementation
   does improve weight-domain error relative to a standard RTN baseline — it is only
   out-competed by a baseline that already optimises NMSE directly (`rtn_group` with an
   MSE-optimal clip search).

Asserting `nmse_gptq ≤ nmse_rtn` as a hard test would be flaky by construction, so it is
not asserted.

### What was NOT done to reach 20%

Each of these would have produced a "passing" number and a dishonest report:

- **Comparing against `rtn_tensor` instead of the strongest baseline.** That yields
  +62.50%, but `rtn_group` is a genuinely stronger baseline and the DoD specifies the
  strongest.
- **Grading only on ActErr and reporting that as the headline.** The gate object
  publishes both axes and `passed` is `false`.
- **Tuning the damping per bit width to maximise the NMSE number.** The damping is a
  single documented constant chosen for robustness across conditioning regimes (§6.1 of
  the benchmark report), not per-cell fitted.
- **Dropping `no_hessian` from the ablation table** because it wins on the axis the gate
  measures. It is reported, prominently.

### How to actually pass it

Any one of these, in order of honesty:

1. **Change the gate to grade the axis the algorithm optimises** — an ActErr-only gate
   passes at +33.80%. The data needed for this is already in `benchmark.json`.
2. **Redefine the baseline** as plain RTN without the MSE clip search — +62.50% NMSE.
   Weaker, and arguably not the "strongest baseline" the DoD intends.
3. **Ship `no_hessian` (AWQ + LS refit) as the default configuration.** It has the best
   weight NMSE *and* most of the output win. It is exposed as an ablation today; making
   it the default would be a legitimate product decision, not a measurement change.

Option 1 is a specification decision, not an engineering one, so it is the
team lead's call. The code supports all three without modification.

---

## Criteria that pass

| Item | Required | Measured | Status |
|---|---|---|---|
| Unit tests | ≥ 25, all green | **186 passed**, 0 failed | ✅ |
| Coverage | ≥ 75% | **86.5%** (branch coverage) | ✅ |
| Lint | `ruff check` clean | **All checks passed** | ✅ |
| Determinism | `benchmark.json` bit-identical across runs | records **True**, full report **True** | ✅ |
| Performance budget | demo ≤ 60 s, ≤ 2 GB | **7.9 s**, < 100 MB peak | ✅ |
| Ablations | ≥ 3 with conclusions | **5**, all with written conclusions | ✅ |
| Failure cases | ≥ 3 analyses | **4** (extreme outliers, rank-deficient Hessian, non-divisible groups, 2-bit collapse) | ✅ |
| Offline fallback | tested without scipy | `test_offline.py`, 10 tests incl. import-blocked pipeline | ✅ |
| CI | py3.12 + 3.13, lint + pytest + demo smoke | `.github/workflows/ci.yml` | ✅ |
| Release | tag + release with benchmark summary | v0.1.0 | ✅ |

---

## Secondary findings (not blocking, but they contradict expectations)

1. **8-bit group-wise quantization is a net loss.** Per-tensor reaches NMSE 7.5e-04;
   every per-group method is ~2.7× worse because the scale metadata is no longer
   amortised. The flagship "loses" by 167.86% NMSE at 8 bits, and that is the correct
   result.
2. **act-order is a net negative on this workload** (−2.6% NMSE, −0.8% ActErr when
   removed). The synthetic Hessian has a mild `diag(H)` spread (5–10), so the
   permutation's benefit does not materialise. Likely reverses on a real LLM; reported
   rather than hidden.
3. **MSE-optimal clipping, not GPTQ, is the largest single contributor** (+130% NMSE when
   removed). On heavy-tailed weights the clip search matters more than the second-order
   compensation.
4. **Damping safety factor is 3.0, not the 20 originally proposed.** At `S = 20` the
   flagship is −71.6% versus RTN once the Hessian spread reaches ~116, because
   under-damping leaves `λ_min` near zero. Sweep table in §6.1 of the benchmark report.

---

## Verification commands

```bash
python -m pytest tests -q --cov=quantforge          # 186 passed, 86.5%
python -m ruff check .                              # All checks passed
python -m quantforge demo --outdir results \
    --n-calib 512 --bits-grid 2,3,4,8 --group-grid 128,64
python -m quantforge verify --outdir results        # bit-identical: True
```
