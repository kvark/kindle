# Prior batching: whole-world gradient gate fails

October 16, 2026. **Do not time or promote aca80566.** Staging and state checks
pass, but the serial/batched gradient comparison fails its declared pointwise
gate. Keep qualified834241b6: RGB48.068ms versus inclusive JAX27.223ms,1.77×.
Production and Atari remain unchanged.
[Declaration](../experiments/2026-10-16-prior-world.md),
[data](2026-10-16-prior-world.json),
[preceding reference diagnosis](2026-10-16-prior-gradient.md).

## Result

Four processes run the existing RGB/CDP12M B32/T16 whole/staged test, changing
only training-prior batching. Each performs six raw backwards without optimizer
updates: original/changed targets, exact whole/staged comparisons, NaN-poisoned
late-input isolation and original restoration. Every within-arm test passes.
All292 RGB/346 CDP model and optimizer tensors remain unchanged; paired initial
states, posterior samples and first targets match the retained control exactly.

Across both target conditions, all56 output arrays pass. Of308 raw-gradient
array comparisons,152 are exact and113 fail the pointwise gate:

| Workload / targets | Failed arrays | Failed values | Largest relative L2 error |
| --- | ---: | ---: | ---: |
| RGB / original | 20 | 8,410 | 0.001717 |
| RGB / changed | 17 | 8,039 | 0.001796 |
| CDP / original | 38 | 493,552 | 0.006894 |
| CDP / changed | 38 | 393,878 | 0.007095 |

All relative L2 errors pass the separate0.01 limit. That does not override
the pointwise2e-5+.02|reference| failures. A second CPU reviewer rechecks all
52,325,812 output/gradient values and reproduces all903,879 failed values.
These are native-versus-native differences, distinct from the preceding
standalone F64-reference discrepancy. No thresholds are changed.

## Concrete contract difference

The disabled graph matches the retained full compiled control. Actual typed
matrix counts drop392→257 RGB and436→301 CDP, with unchanged nominal arithmetic.
Total dispatches drop3992→3010 and4425→3443; barrier groups stay1242/1248.
All23 RGB/11 CDP convolution contracts are unchanged. Counts are not timings.

The first prior matrix changes implementation when its row count grows:

- Serial: sixteen32×256×2048 products, each with eight K partitions, F32 partial
  products, an F32 `SumRows`, then explicit BF16 conversion.
- Batched: one512×256×2048 product, one K partition, the64-column/K32 staged
  cooperative implementation, and fused BF16 output conversion.

All34 first-layer paths across both workloads reconcile. **The BF16 conversion
is present in both arms**; an F32 partial output is not evidence that rounding
was omitted. Partitioning/reduction order and implementation nevertheless change.
This is a concrete candidate cause, not a measured causal explanation of the
full gradient difference. The earlier tidy standalone fixtures do not establish
agreement for the trained checkpoint's inputs and weights.

Next inspect the actual first-prior product on saved posterior inputs and
checkpoint weights, including both partition schedules and conversion boundaries,
before deciding whether new instrumentation or a contract-preserving repair is
warranted. Do not replace this failed qualification, enlarge its tolerances,
start a K/tile/precision sweep or proceed to the queued timing allocation.
The broader convolution/imagination/host-work root-cause goal remains open.

Four execution guards/no warnings,24 raw backwards, zero optimizer updates or
game actions. The numerical review fails and stays failed; independent failure
and contract audits pass without another GPU call.3,455 source seals verified;
minimum sampled estimated headroom12,332,564,480bytes. No source rebuild or
worker remains. Raw artifacts: `runs/prior-world-20261016.YVA1AFTT`.
