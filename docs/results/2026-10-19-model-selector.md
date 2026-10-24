# Actual-model selection is correct, but not a substantial speedup

The common dense selector now works on the actual cached/staged learner, not
just small fixtures. **Its complete-update benefit fails the3% RGB gate.**
Keep the untuned`d3ff4948` control; production and Atari remain stopped.
[Compact evidence](2026-10-19-model-selector.json).

| Complete12M/B32/T16/H15/BF16 update | Fresh untuned | Selected | Reduction |
|---|---:|---:|---:|
| RGB |43.010ms|42.890ms|0.28%|
| CDP |40.817ms|40.375ms|1.08%|

Controls reproduce the retained trajectories and remain within.064%/.187% of
their preceding timings. All256 measured samples per arm remain, after16 warmup
updates and one initializer. RGB is still approximately1.58× caller-inclusive
JAX27.223ms. These small differences do not establish a useful optimization or
learning benefit. CDP remains a different exploration workload from RGB/JAX.

## What this rules out

The search is no longer missing the retained typed dense family or its large
bindings. Conservative full-race bounds are111.25MiB RGB/228.625MiB CDP, including
staging, split reductions and per-candidate command-encoder allowances. They fit
the declared512MiB cap; opaque driver/pipeline storage is excluded, so these are
not peak-VRAM measurements. Actual alias/device filtering admits53 RGB/61 CDP
classes across the four programs. No class/scratch/deadline limit blocks a race;
the ordinary predicted-share cutoff leaves a few low-ranked classes unvisited.

The timed candidates change **180 RGB/206 CDP dispatches and exactly as many
selected pipeline keys**, including checked, shared-tiled and workgroup kernels.
Plans retain all other semantics, groups, constants, allocations and interfaces.
Cached commands re-record after selection. Source review confirms recording reads
the selected pipelines and current launch grids; this is not a label-only change.
Grouped products and convolutions remain outside this dense search.

Search takes3.536s RGB/5.352s CDP, outside measured updates. Summing selected
private host-timed differences weighted by occurrence gives1.106/1.478ms, while
the complete updates differ by only.121/.442ms. **Those private timings are not
additive graph savings.** Their execution context differs; this study does not
isolate the cause of that compression. RGB posterior+world is essentially
unchanged,31.844→31.840ms. Missing dense-kernel choices are therefore not the
large remaining explanation under this recipe.

## Qualification and limits

- Six guarded processes pass: two qualification workloads, then four fresh
  timing arms, each qualifying its own selections before timing. The timing
  searches can select differently; their actual plans are retained and checked.
- **527,639,604 output/raw-gradient values match exactly** across before/after
  selection. Actual synthetic learner inputs, post-initializer weights and late
  targets are used, not live-game replay or an independent full-model oracle.
- Whole/staged world pairs match exactly for original, changed and restored
  reward targets, with NaN-poisoned late targets during preparation. Inputs,
  parameters and optimizer moments stay unchanged during diagnostic passes.
- Both timing pairs match all272 metric reports and all292 RGB/346 CDP tensors
  at initial, first, post-qualification, final and restored checkpoints. Optimizer
  counters advance272 times after diagnostics; target-state roundtrips pass.
  Untuned reports reproduce the retained271-report continuation prefix.
- Four CPU tests/1,182 offline variants pass; strict Kindle test-target Clippy
  passes. Full source/plan/pipeline review checks678 timing-queue seals.
- Eight recorded allocation warnings, no API/numerical/hard-fault failure,
  no recovery/NVML polling. Total1,094 synthetic updates,84 diagnostic raw
  backwards,24 forward-only passes, zero actions. Last GPU guard ends14:14:54UTC;
  final tensor review14:19:05UTC. No worker remains.

The first independent CPU audit finishes in397s with unrestricted BLAS threads
under its one-CPU quota. It completes before an attempted stop; no repeated GPU
or CPU cohort follows. The four-arm audit uses one BLAS thread and finishes in20s
under the same CPU/memory limits and unchanged arithmetic/gates. Preserve both
logs. The failed performance gate is not relaxed or retimed.

## Next

Keep this selector integration as qualified isolated infrastructure, not a speed
promotion or a reason to expand its geometry catalog. The large remaining targets
are still convolution feeding/control work and non-matrix recurrence processing.
The existing [actual-model counters](2026-10-17-model-counters.md) and
[recurrence contract](2026-10-17-recurrence-contract.md) provide the starting
evidence. Resume the CPU proof of typed subgroup reduction tails and slice-view
consumers before a new implementation/allocation, while keeping their limited
possible benefit distinct from convolution's larger deficit. No general parity
plan or root-cause completion is claimed; no blind kernel sweep follows.

Evidence: `runs/model-selector-20261019.0xWDbwn1`, including both sealed queues,
all plans/arrays/races, `review.json`, `timing/publication.json`, `publication.json`
and `trajectories.json`. Qualification binary`e98ebeb1`, timing binary`33dc91b8`;
unchanged qualified backend plus selector, distinct timing harness/cache.
[Qualification declaration](../experiments/2026-10-19-model-selector.md) ·
[Complete timing declaration](../experiments/2026-10-19-model-selector-timing.md).
