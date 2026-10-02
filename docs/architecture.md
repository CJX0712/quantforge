# QuantForge Architecture

> Post-training quantization (PTQ) research toolkit: RTN, AWQ and a GPTQ-based
> flagship, implemented from the papers in NumPy with no deep-learning framework.

## 1. Design goals

1. **Reproducible.** Every number in `benchmark.json` is a pure function of the seed.
   Two runs on the same machine produce bit-identical results.
2. **Verifiable.** Each algorithm is guarded by numeric invariants that would fail
   loudly if the implementation drifted (Cholesky residual, requantization
   idempotency, grid-point fixed points, permutation round trips).
3. **Honest.** The metric a method actually optimises is reported next to the metric
   it is graded on, rather than quietly optimising the graded one.
4. **Dependency-light.** The quantizer core needs NumPy only. `scipy` is optional;
   when the factorisation cannot succeed the code degrades to RTN *with a warning*,
   never silently.

## 2. Layering

The call graph is strictly acyclic. `core` never imports a sibling subpackage.

```
                    cli.py
                      |
                 pipeline/            <- experiment orchestration, report writing
                   /   |   \
                  /    |    \
            data/  quant/  eval/     <- synthetic workloads | algorithms | metrics
                  \    |    /
                   \   |   /
                    core/            <- types, errors, config, interfaces, seeding
```

| Layer | Modules | Responsibility |
|---|---|---|
| `core` | `types` `errors` `config` `interfaces` `seed` | Contracts, `E1xx`-`E5xx` codes, `ENV_QF_*` config, the single RNG |
| `data` | `synthetic` | Reproducible ReLU MLPs with activation outliers |
| `quant` | `rounding` `scale` `gptq` `awq` `schemes` `selectors` | Quantizers |
| `eval` | `metrics` `benchmark` | Error metrics, aggregation, DoD gate |
| `pipeline` | `runner` | Grid execution, `benchmark.json` |

## 3. Quantization primitives

### 3.1 Rounding

All rounding is `floor(x + 0.5)` (round-half-up), never `np.round`.
`np.round` is round-half-**even**, so `0.5 -> 0` and `1.5 -> 2`: unbiased, but the
behaviour of an exact tie depends on the value's parity, which is a nuisance when
debugging a codebook. One rule, applied identically during calibration and export, is
easier to reason about. Python's `int()` is never used for rounding -- it truncates
toward zero, which silently corrupts asymmetric zero-points on the negative half axis.

Integer range, `qmax = 2^(b-1) - 1`:

| scheme | range | rationale |
|---|---|---|
| symmetric | `[-qmax, qmax]` | Weights are near zero-centred; `z=0` makes `0` exactly representable and needs no zero-point storage |
| asymmetric | `[-2^(b-1), 2^(b-1)-1]` | Post-ReLU activations have `min = 0`; a symmetric grid would waste half its levels |

### 3.2 Granularity and storage cost

Quantization runs along the **column** (input-channel) axis of a `(d_out, d_in)` matrix.

| granularity | one scale per | covers |
|---|---|---|
| `per_tensor` | whole matrix | `n_weights` |
| `per_channel` | input channel | `d_out` |
| `per_group` | `group_size` consecutive columns | `min(group_size, d_in)` for the tail |

```
bits_per_weight = bits + n_groups * (scale_bits + zero_bits) / n_weights
```

with `scale_bits = 16` (fp16) and `zero_bits = 0` for symmetric weights. At
`bits=4, g=128, d_in=128` this is `4.125`; at `g=64` it is `4.25`.

The trailing group is **never padded**. Padding with zeros would raise `max|x|` and
therefore change the whole group's scale, corrupting its codebook. The group is simply
shorter.

### 3.3 MSE-optimal clipping

A few outlier weights inflate `max|x|`, squeezing the bulk of the distribution into a
handful of levels. `search_clip_alpha` grid-searches a clipping ratio in `[0.5, 1.0]`
minimising NMSE. `alpha = 1.0` (no clipping) is always in the grid, so the search can
never do worse than min-max. The curve has kinks where values cross a clipping
boundary or hit a rounding tie, so a coarse grid is followed by a local refinement
rather than a golden section that could step over a kink.

## 4. GPTQ

Implements Frantar et al., ICLR 2023, with the corrections below.

### 4.1 Pipeline

```
H = 2 X Xᵀ / N                       # Hessian, symmetrised
H[dead, dead] = 1; W[:, dead] = 0   # dead channels isolated BEFORE damping
perm = argsort(-diag(H))            # act-order: highest activation energy first
H, W ← H[perm][:, perm], W[:, perm]
H += percdamp * mean(diag(H)) * I   # RELATIVE damping, with a x10 retry ladder
U = upper Cholesky factor of H⁻¹
for each block of 128 columns:
    for each column i in the block:
        q   = quantize(W[:, i])
        e_i = (W[:, i] - q) / U[i, i]
        W[:, i+1:] -= e_i ⊗ U[i, i+1:]
    W[:, next_block:] -= E_block @ U[block, next_block:]
return (codes, scales) un-permuted by inv(perm)
```

### 4.2 Why two Cholesky factorisations

`np.linalg.inv(H)` conditions the result by `κ(H)²`. Factoring twice squares the
condition number only once and avoids the explicit inverse:

```
H = L Lᵀ  →  G = L⁻ᵀ L⁻¹  →  Lc Lcᵀ = G  →  U = triu(Lcᵀ),  so G = Uᵀ U
```

Verified by the test `‖Uᵀ U H_damped − I‖ / d_in < 1e-10`.

`np.linalg.cholesky` returns the **input matrix in the opposite triangle**, so `triu()`
is mandatory before treating the factor as upper triangular. Skipping it injects the
Hessian into the factor and every compensation coefficient becomes garbage.

### 4.3 Corrections to the naive implementation

| Trap | Consequence | Guard |
|---|---|---|
| `W1[:, t]` is a **view** | The compensation update mutates the "original" column before its error is read | `.copy()` on every column read |
| Only `W` permuted, not `H` | Non-symmetric Hessian that Cholesky may still factorise | `assert np.array_equal(Hp, Hp.T)` |
| Absolute damping constant | Meaning changes when `H` is rescaled | Only `percdamp * mean(diag(H))` |
| `int()` for rounding | Truncation toward zero corrupts negative zero-points | `floor(x + 0.5)` only |
| Cross-block update without an `i2 < d_in` guard | Index error on the final block | Explicit guard |
| Silent `LinAlgError` catch | Garbage accuracy reported as healthy | `try/except` + `isfinite` + upper-triangle check + warn + report |

### 4.4 Static groups and act-order

`per_group` requires the columns sharing a scale to be **contiguous in storage order**.
act-order scrambles that order, so groups are fixed on the *original* column index and
looked up through the permutation: `group_of(perm[col])`. The alternative -- slicing
groups in permuted order -- produces a scale array belonging to the permuted order,
which cannot be serialised without also persisting `perm` and a kernel that respects
it. Production should not take that path.

### 4.5 Cost

`H` accumulation `O(N·d_in²)`, two Cholesky factorisations `~3·d_in³/3` (roughly 40% of
total time), column-block compensation `O(d_out·d_in²/2)`. The block size reduces
**memory traffic** by `1/blocksize`; FLOPs are unchanged. `blocksize = 128` is the
sweet spot.

## 5. AWQ

Per-input-channel scaling with `W' = W diag(s)`, `X' = diag(s)⁻¹ X`. Exactly invertible
in floating point, so the scaling folds into the previous layer (`W_prev / s[:, None]`,
`b_prev / s`) at zero inference cost.

Dequantising back to original coordinates gives `|E_kj| ≤ Δ_g(s) / (2 s_j)`: amplifying a
column buys it more grid points, at a cost shared by its whole group. The objective
being searched is

```
J(s) = (1/12) Σ_g Δ_g(s)² Σ_{j∈g} E‖x_j‖² / s_j²
```

Salience uses `mean(|x|)` (not `mean(x²)`) so a few outlier tokens cannot dominate, and
is normalised by its own mean so `s` is dimensionless. `alpha = 0` gives `s ≡ 1`, which
is why the grid always contains it: **AWQ under an identical quantiser can never lose
to RTN**. `s` is clipped to `[1/3, 3]` so the previous layer's error is not amplified
without bound.

## 6. QuantForge flagship

Three stages, in this order:

1. **AWQ scaling** -- exponent searched against weight NMSE. GPTQ re-derives the output
   error from the Hessian immediately afterwards, so searching output error here would
   optimise the same thing twice.
2. **GPTQ** -- second-order error compensation.
3. **LS scale refit** -- with codes `c` fixed, minimising `‖W − s·c‖²` over `s` has the
   closed form `s* = Σ(W·c) / Σ(c²)`.

Stage 3 exists because GPTQ minimises the **output** error and deliberately trades
weight-domain accuracy for it. Its min-max scales are therefore far from optimal for
`‖W − Ŵ‖²`, and refitting recovers weight NMSE without disturbing the code assignment.

The AWQ scaling is undone before returning (`Ŵ / s`), so every method returns a drop-in
replacement for `W` in the caller's coordinates.

## 7. Metrics

| metric | definition | direction |
|---|---|---|
| `nmse` | `‖W − Ŵ‖²_F / ‖W‖²_F` | lower |
| `act_err` | `‖ŴX − WX‖²_F / ‖WX‖²_F` | lower |
| `sqnr_db` | `10 log10(Σw² / Σ(w−ŵ)²)` = `−10 log10(nmse)` | higher |
| `bits_per_weight` | see §3.2 | lower |
| `compression_ratio` | `32 / bits_per_weight` | higher |

An all-zero reference has no defined NMSE; these return `inf` rather than raising, so a
degenerate layer shows up as a visibly bad score instead of a crash.

### 7.1 The two axes

GPTQ minimises `‖(W−Ŵ)X‖²` while `nmse` is `‖W−Ŵ‖²/‖W‖²`. **These provably diverge.**
GPTQ moves weight error into directions the activations do not excite, which is the
whole point -- so it can be *worse* on `nmse` while being clearly better on `act_err`.

The report therefore evaluates the DoD threshold on both axes and publishes both
(`gate` in `benchmark.json`), plus a comparison against the naive per-tensor baseline
at equal granularity. A single headline number would misrepresent the result. The
hard invariant actually asserted in the test suite is the one GPTQ is *defined* by:

```
‖(W − Ŵ_gptq)X‖²_F  ≤  ‖(W − Ŵ_rtn)X‖²_F
```

Asserting `nmse_gptq ≤ nmse_rtn` would be a flaky test by construction.

## 8. Determinism

- One RNG: `core.seed.set_all(seed)` seeds `random`, NumPy global and the project
  generator. Quantizer cores consume no randomness at all.
- BLAS is pinned to a single thread. Multi-threaded GEMM changes summation order
  between runs, which changes `argsort` tie-breaking, which changes the codebook.
- `stable_core()` excludes only `elapsed_sec`, per-record `wall_sec`, `env` and `path`.
  Everything describing results must match bit for bit; `quantforge verify` enforces it.

## 9. Error codes

| code | class | meaning |
|---|---|---|
| `E100` | `ConfigError` | Bad or unknown configuration |
| `E110` | `ValidationError` | Failed schema/range validation |
| `E200` | `DataError` | Invalid synthetic workload request |
| `E210` | `ShapeError` | Tensor shape mismatch |
| `E300` | `QuantizationError` | Invalid quantizer configuration or state |
| `E310` | `DecompositionError` | Cholesky failed after damping retries |
| `E400` | `MetricError` | Metric undefined for the given inputs |
| `E500` | `PipelineError` | Benchmark orchestration failure |
| `E999` | -- | Any non-QuantForge exception |

## 10. Extension points

- **New method**: subclass `core.interfaces.Quantizer`, register it in
  `quant.schemes.build_quantizer`.
- **New granularity**: add a case to `quant.scale.group_bounds` / `n_groups_for` and
  return the matching `bits_per_weight`.
- **New ablation**: accept the name in `QuantForgeQuantizer.__init__` and disable the
  corresponding stage; the test that every ablation *changes* the result keeps it honest.
- **New model**: add an entry to `data.synthetic.MODEL_SHAPES` as `(d_out, d_in)` per layer.

## References

- Frantar et al., *GPTQ: Accurate Post-training Compression for LLMs*, ICLR 2023
- Frantar et al., *Optimal Brain Compression*, ICLR 2022
- Lin et al., *AWQ: Activation-aware Weight Quantization*, MLSys 2024
- Chen et al., *ZeroQuant*, NeurIPS 2022
- Bennett, *Rational Quantizers*, IRE Trans. Inf. Theory 1948 (`6.02b + 1.76` dB)
- Max, *Quantizing for Minimum Mean Square Error*, IRE Trans. Inf. Theory 1960

---

Author: 晨星 (Chenxing)