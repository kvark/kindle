# Final optimizer repair qualifies on Metal and the complete native learner

October21,2026. [Meganeura PR275](https://github.com/kvark/meganeura/pull/275)
at`5d916851` is **ready for review, not merged**. All six
[CI1048 jobs](https://github.com/kvark/meganeura/actions/runs/39568247325) pass,
including the unchanged finite/tiny-state and NaN/Inf regressions on Metal.
Original CI1047 remains failed. No tolerance relaxation or test skip.

The independent`1-beta` uniforms resolve the observed Metal failures without
changing checkpoint/state storage, epsilon or global fast-math policy. This
supports the preceding cancellation diagnosis; it is not a disassembly result
or a general IEEE emulation guarantee. [Prior evidence](2026-10-21-optimizer-model.md).

## Retained full-model interface

Isolated native`72005c8b536bedf02543676e2be04e7d55f18a10e7c5b6975a63506cdc642faf`
applies exactly the two-file coefficient delta to the previous6b621e5c runtime.
All retained kernels/extensions, compiler versions, six flags and RGB/CDP recipes
remain. The initial whole-file equivalence assertion correctly rejects an
upstream difference: the retained optimizer has a prepared-step safety guard.
The separately reviewed integration applies exact-context committed hunks and
preserves that guard. Original failure/source/log stay; no GPU retry.

Four24-update ordinary/observer runs pass **160,998,750 independent F64 state
comparisons** at the original bounds. Every optimized parameter, momentum and
variance is checked at updates1/24, including clocks, learning-rate multipliers
and actual post-AGC gradients. Both cached observer trajectories match exactly,
as do the previous repair's24 reports, first targets and initial/first/final
292 RGB/346 CDP tensors. All five programs/pipelines reconcile; only the exact
optimizer body changes in runtime-source comparisons. No model subnormal
variances occur in these snapshots; the component tests establish that case.
Four guards/zero warnings/96 updates,1,341 receipts. Review08:26:05UTC.

## Final-interface complete-update cost

| RGB/CDP12M/B32/T16/H15/BF16 | Prior repair6b621e5c | Final72005c8b | Change |
| --- | ---: | ---: | ---: |
| RGB |48.524ms|48.448ms|−.156%|
| CDP |41.792ms|41.826ms|+.082%|

Essentially unchanged, both within the declared5% overhead limit. All256 measured
samples per arm retained; no retime or profiler timing. All272 reports and
initial/first/final tensors match exactly on these fixtures; candidate prefixes
match their own numerical qualification. Full programs/pipelines/source deltas
pass. This is not a gameplay result, independent full-network oracle or speed win.

Four guards/zero warnings/1,088 updates,1,346 receipts; final CPU review
**08:28:34UTC**. Across this study: eight passing guards,1,184 synthetic updates,
zero game actions. No local worker remains. Artifacts:
`runs/optimizer-final-20261021.f8VAIszC`; [compact result](2026-10-21-optimizer-final.json).

## Next

Use the qualified isolated72005c8b for the next separately declared performance
study; production remains unchanged and only the user merges PR275. Refresh
complete-update attribution on driver615 before acting on historical580 counters.
The old coarse observer needs both categorical-mean and resident-target timestamp
coverage: current helper waits do not publish those timings. Preserve numerics,
cached execution and account for profiling-induced waits. Do not label pass
coverage SM utilization or infer equal machine instructions from equal sources.

Current RGB48.448ms versus retained current-driver JAX26.066ms remains about1.86×;
weights/RNG differ, so this is matched-shape throughput, not learning parity.
Convolution feeding remains an open historical structural finding. No automatic
production promotion or Atari restart. The five-game CDP goal stays unfinished.
