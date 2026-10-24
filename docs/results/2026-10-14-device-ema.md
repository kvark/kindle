# Device-side slow-critic averaging removes another host round-trip

October 14, 2026. [Declaration](../experiments/2026-10-14-device-ema.md) ·
[Data](2026-10-14-device-ema.json) · [Performance plan](../backend_training_performance_plan.md).

**RGB updates improve30.260→29.169ms (3.61%); larger-batch CDP improves
95.473→94.059ms (1.48%).** This confirms another avoidable host boundary,
not the entire remaining gap. RGB is still **2.27×** retained complete
JAX12.836ms. Production remains on its earlier qualified~53ms RGB runtime.

The control is the [exact-sequence GPU RNG candidate](2026-10-14-device-rng.md).
The only change is slow-critic averaging. Meganeura blends selected dense F32
parameters on the shared queue and refreshes affected vertical/horizontal
concatenations. It checks shapes, storage, extents, aliases, duplicates and
prepared-step boundaries before writing. Both sessions track completion.
Reduced/packed master storage fails explicitly; no raw-pointer parameter escape
or CPU fallback. The learning graphs, model/batch/precision/RNG and optimizer
are unchanged. F32 contraction/underflow can change rounding.

## Matched full updates

One pair per workload,16 warmup+256 measured updates, no profiler or concurrent
build/analysis. RGB12M uses B8/T16/H15 BF16; CDP12M B32/T16/H15 mixed.

| Stage | RGB control → device | CDP control → device |
| --- | ---: | ---: |
| Behavior training/sync | 2.593→1.594ms | 6.050→4.857ms |
| Imagination/targets | 8.026→8.012ms | 31.582→31.604ms |
| World training/sync | 15.235→15.187ms | 46.223→46.159ms |
| Complete update | 30.260→29.169ms | 95.473→94.059ms |

The improvement is concentrated in the changed behavior/sync stage. These
are synthetic updates, not game throughput or time-to-score. JAX was not rerun;
its previous-metric-return limitation and different graphs/weights/RNG remain.
CDP is not compared directly with that RGB/JAX number. No SM-utilization or
additive speedup claim.

## Numerical scope and retained limitations

Four CPU tests and both release Clippy checks pass. **5,267,639 primitive
values** meet the predeclared independent F64 pointwise bound;286,366 differ
from the host F32 formula. Irregular tails, cancellation, partial concatenation
refresh, untouched parameters, source immutability/destruction, later host
updates and a queued source SGD update pass. Derived copies remain exact.
The direct GPU length cases run through1,048,579 elements; this is not exhaustive
qualification of every size supported by the dispatch geometry.

Both24-update smokes pass. Full runs retain exact initial292/346 tensors,
first pre-EMA outputs, first-step non-EMA tensors and compiled session plans.
Each workload's853,503 first EMA values passes the F64 bound; maximum error
is0.412/0.481 of that bound. Only slow-critic tensors differ after the first
step: RGB3, maximum1.49e−8; CDP9, maximum1.19e−7.

**Later trajectories differ:**82/272 RGB and1/272 CDP reports remain exact.
Final tensor relative L2 is.0724/.8690; all states/metrics are finite. These
are not learning-equivalence results. Small EMA rounding changes propagating
through the learner also do not prove the earlier near-zero LaProp hypothesis
or a GPU fault. Keep this limitation before any adoption/learning comparison.

Seven guards pass; six allocation warnings retained, no GPU failure/retry,
recovery or separate NVML polling.1,136 full synthetic updates plus one small
SGD update, zero game actions. The offline audit initially shadowed its tensor
file handles with temporary arrays; failed source/log remain, and the corrected
audit uses the same saved outputs and unchanged gates. No worker remains.

Artifacts `runs/device-ema-20261014.PuHZB9mx`; nativea1c366a5 versus RNG
control57a1a1a3, backend tests786083c2. Upstream2214412/fe5f5b5 is unchanged;
all patches, configurations, binaries, controls and warning receipts remain.

**Keep device EMA in the isolated candidate.** No installed pins/defaults,
neighboring work or Atari allocations change. Next connect device target
construction, then row-level normalization/backward and safe view/layout
fusion. The broad root-cause/performance goal remains active.
