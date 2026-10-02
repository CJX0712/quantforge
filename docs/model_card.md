# QuantForge Model Card

## Model / system summary

**QuantForge** is a post-training quantization (PTQ) research toolkit for compressing
neural network weights. It implements four weight quantizers in NumPy with no deep
learning framework, and a benchmark harness that compares them under a fixed, seeded
protocol.

| | |
|---|---|
| Version | 0.1.0 |
| Task | Post-training weight-only quantization |
| Inputs | Weight matrix `(d_out, d_in)` fp32 + calibration activations `(d_in, N)` |
| Outputs | Integer codes, per-group fp16 scales, dequantized weights |
| Bit widths | 2, 3, 4, 8 |
| Group sizes | 32, 64, 128 (any positive integer) |
| Runtime deps | NumPy only (`scipy` optional) |
| License | MIT |

## Methods compared

| key | definition |
|---|---|
| `rtn_tensor` | **A.** Round-to-nearest, per-tensor min-max scale |
| `rtn_group` | **B.** Round-to-nearest, per-group min-max + MSE-optimal clip search |
| `awq` | **C.** AWQ per-channel scaling, exponent searched against output MSE, then RTN |
| `quantforge` | **D.** Flagship: AWQ scaling -> GPTQ error compensation -> LS scale refit |

### Bit-width formulas

One scale (and, for asymmetric, one zero-point) is stored per group, shared by all
output rows:

```
bits_per_weight = bits + n_groups * (scale_bits + zero_bits) / n_weights
```

where `n_groups = ceil(d_in / group_size)` for `per_group` granularity,
`scale_bits = 16` (fp16) and `zero_bits = 0` for symmetric weights.

| granularity | `bits_per_weight` | example (`d_in=128`, `d_out=32`) |
|---|---|---|
| `per_tensor` | `bits + 16 / (d_out·d_in)` | 4-bit: `4.0039` |
| `per_channel` | `bits + 16 / d_in` | 4-bit: `4.125` |
| `per_group(g)` | `bits + 16·ceil(d_in/g) / d_in` | 4-bit, g=128: `4.125`; g=64: `4.25`; g=32: `4.5` |

`compression_ratio = 32 / bits_per_weight`.

## Metrics

| metric | definition | direction |
|---|---|---|
| `nmse` | `‖W − Ŵ‖²_F / ‖W‖²_F` (weight domain) | lower |
| `act_err` | `‖ŴX − WX‖²_F / ‖WX‖²_F` (output domain) | lower |
| `sqnr_db` | `10·log10(Σw² / Σ(w−ŵ)²)` = `−10·log10(nmse)` | higher |

`act_err` is computed against calibration activations and is the quantity GPTQ actually
optimises. `nmse` is weight-domain and is *not* what GPTQ optimises -- see
"Intended use and limitations".

## Evaluation protocol

- **Dataset seeds**: `{7, 13, 23}` -- each produces a different outlier-channel layout
  and therefore a different Hessian spectrum.
- **Models**: `m0` (64→32), `m1` (32→16), `m2` (128→64), `m3` (96→48), all ReLU.
- **Calibration**: 512 samples per layer.
- **Cells**: 4 models × 3 seeds × 4 bit widths, per-group grid `{128, 64}` plus a
  per-tensor control.
- **Reported**: `mean ± population std` across cells, never a single number.
- **Significance**: a win is flagged when
  `mean_baseline − mean_candidate > 0.5·(std_baseline + std_candidate)`.

### Activation outliers are deliberate

The synthetic activations inject outliers on a few channels (gain 8-30×) that fire on
only 3-5% of samples. Without them every channel would have equal energy, the AWQ
scaling would collapse to a constant, and all four methods would tie -- the experiment
would demonstrate nothing. This mirrors the structured outliers observed in real LLM
activations and is what makes `H = 2XXᵀ` ill-conditioned enough for GPTQ to have
anything to compensate.

## Measured results

Produced by `python -m quantforge demo` on the reference workload. Every number below
is read back from `results/benchmark.json`; none is hand-entered. See
[`benchmark_report.md`](benchmark_report.md) for the full per-bit tables, the ablation
study and the failure-case analysis.

### Headline (4-bit, 48 cells)

| method | NMSE (mean±std) | ActErr (mean±std) | SQNR (dB) |
|---|---|---|---|
| `rtn_tensor` (A) | 2.253e-01 ± 1.0e-01 | 2.274e-01 ± 1.2e-01 | 7.05 |
| `rtn_group` (B) | 8.232e-02 ± 1.9e-02 | 8.659e-02 ± 2.9e-02 | 11.02 |
| `awq` (C) | 2.266e-01 ± 1.0e-01 | 2.062e-01 ± 1.1e-01 | 7.02 |
| `quantforge` (D) | 9.105e-02 ± 2.3e-02 | **6.497e-02** ± 2.7e-02 | 10.60 |

- Flagship vs strongest baseline: **ActErr −24.97%**, NMSE +10.61% (worse).
- Flagship vs naive per-tensor RTN at equal granularity: **NMSE −59.58%**.

## Ablations

Each variant removes exactly one component from the flagship (4-bit, 12 cells):

| variant | NMSE | ActErr | conclusion |
|---|---|---|---|
| `no_mse_clip` | 2.171e-01 | 1.425e-01 | **Largest single contributor.** MSE-optimal clipping is worth ~2.4× NMSE; min-max scales are badly hurt by the heavy-tailed weights |
| `no_act_aware` | 1.073e-01 | 7.101e-02 | AWQ scaling costs 17.8% NMSE but gains 8.4% ActErr -- it trades weight error for output error, as designed |
| `no_hessian` | **7.697e-02** | 7.233e-02 | *Best weight NMSE of any variant.* GPTQ's compensation improves ActErr by 10.2% but costs 18.3% NMSE |
| `no_act_order` | 9.385e-02 | 6.075e-02 | Act-order is worth 3.1% NMSE and 6.5% ActErr here |

## Intended use and limitations

**Intended use.** Research and education: comparing weight quantizers under a fixed,
reproducible protocol; prototyping a new quantizer against the reference
implementations; teaching the mechanics of GPTQ error compensation.

**Limitations -- read before citing a number.**

1. **The flagship does not win on weight NMSE, and cannot.** GPTQ minimises the proxy
   output error `‖(W−Ŵ)X‖²`; `nmse` is the weight-domain `‖W−Ŵ‖²/‖W‖²`. These provably
   diverge. GPTQ deliberately moves weight error into directions the activations do not
   excite -- that is the mechanism, not a defect. The `no_hessian` ablation (AWQ + LS
   refit, no GPTQ) attains the best weight NMSE of any variant, which is the direct
   evidence. Judge the flagship on `act_err`, or use `no_hessian` when weight-domain
   fidelity is what matters.
2. **Synthetic workloads only.** Every number comes from generated ReLU MLPs with
   injected outliers, not from trained language models. The outlier structure is
   realistic in *shape* but not in *value*: real LLM weights have kurtosis around 5-10,
   these have ~25-30. Absolute NMSE values will not transfer to a real checkpoint.
3. **Calibration is small and synthetic.** 512 samples from a known generator. Real
   calibration draws from the model's own data distribution; distribution shift between
   calibration and deployment is not modelled here.
4. **Single layer, single batch.** Each cell quantizes one layer independently. There is
   no sequential/propagated quantization, so error accumulation across depth is absent.
5. **Weight-only.** Activations are calibrated but never quantized.
6. **No packed kernels.** Codes are `int64` NumPy arrays, so `bits_per_weight` is an
   accounting figure, not a measured on-disk size. Real kernels pack int4 and add
   padding.
7. **Determinism is same-machine.** BLAS is pinned to one thread so results are
   bit-identical on one host. Cross-machine reproduction requires matching BLAS builds.

## Ethical considerations

Intended for compressing models the operator already has the right to modify. Intended
use does not include evading access controls, removing watermarks from proprietary
checkpoints, or deploying compressed models in safety-critical roles without the
accuracy loss analysis that compression implies. Quantization degrades worst exactly
where the signal is weakest -- on outlier channels and rare tokens -- so a compressed
model should be re-evaluated on downstream safety tasks before deployment.

## Reproducing

```bash
pip install -e ".[dev]"
python examples/run_demo.py --quick            # fast smoke run
python -m quantforge demo --outdir results     # full grid -> results/benchmark.json
python -m quantforge verify --outdir results   # assert bit-identical rerun
```

BLAS threads are pinned inside the package, so no environment setup is required for
reproducibility.

---

Author: 晨星 (Chenxing)