# Recurrence matrices do not share convolution's instruction overhead

October16,2026. [Declaration](../experiments/2026-10-16-jax-matrix-counters.md)
· [Compact evidence](2026-10-16-jax-matrix-counters.json).

Across all **180 actual imagined-recurrence matrix products**, native executes
**277.979M warp instructions versus JAX300.272M:7.42% fewer**, with nearly
identical tensor-instruction counts and zero measured local-memory traffic in
both. The large native instruction overhead established for convolution does
**not** generalize to these matrices. It is not evidence that native matrices
are faster or that the remaining complete-update gap is explained.

Keep isolated58aff452: **RGB46.199/27.223ms inclusive JAX,1.70×**. No new
ordinary timing, kernel change, production promotion or Atari restart.

## Complete recurrence counters

| Measured counter | Native | Actual JAX |
| --- | ---: | ---: |
| Total warp instructions |277,978,560|300,271,680|
| Tensor-pipe instructions |21,934,080|21,964,800|
| ALU-pipe instructions |126,150,720|94,085,760|
| FMA-pipe instructions |43,366,080|18,994,560|
| LSU-pipe instructions |40,186,560|109,371,840|
| Uniform-pipe instructions |236,160|28,264,320|
| Shared-read conflicts |22,837,740|937,688|
| Shared-read wavefronts |69,416,940|68,229,848|
| Shared-write conflicts |19,709,715|17,028,196|
| Shared-write wavefronts |44,416,275|48,831,076|
| Local read/write bytes |0/0|0/0|

Native has24.36× as many shared-read conflicts, but only1.0174× the total read
wavefronts. Conflict counts alone are not a latency share or an optimization
target. The rejected vector-staging family already reduced conflicts while
increasing total instructions. Pipeline counters are not source-site attribution
or model MAC counts; retain each independent counter rather than assuming exact
shared-counter additivity.

Actual JAX SASS includes1,344,000 `LDGSTS`,1,996,800 `LDSM`,142,080 `LDGDEPBAR`
and130,560 `DEPBAR` executions across this recurrence. This shows library async
copy/matrix-load forms, not an8× matrix advantage or a reason to revive the
rejected native pipeline. The complete opcode distributions remain in the data.

## Scope and controls

The [interface audit](2026-10-16-jax-matrix-contract.md) proves12 roles repeated
15 times, matching physical layouts except explicit18→24 action padding.
Grouped JAX products include BF16 beta=1 additions; native adds bias outside the
matrix boundary. Nominal MACs differ only44.866/44.890G. Individual HLO-to-launch
ownership remains unresolved; aggregate membership is independently proven.

The unchanged full-JAX benchmark executes272 synthetic updates. Its existing
profiler API interval plus exact four-name selector/skip85/count180 captures
**every actual recurrence matrix**, not one timestep multiplied by15. All eight
retained trace updates prove the same selection. Names, grids, blocks, registers
and shared storage match all180 captured launches. All272 metric reports,
288 initial/final tensor fingerprints and full semantic HLO work match the
retained control, including940 fusion bodies/112 custom calls.
The separate publication audit verifies203 retained source/artifact seals,
recomputes both16-counter aggregates and all180 native memberships, and repeats
the exact reference comparison. It finishes15:28:23UTC without GPU calls.

Native uses the seven retained unchanged tiled contracts for150 occurrences,
plus two newly captured actual512-row action contracts for30. All180 membership
indices reconcile exactly. Native operands are synthetic checked fixtures, not
JAX's live model values: this is not joint numerical or learning parity.
561,152 independent F64/integer output checks and149,504 input checks pass,
including sign-changed/restored outputs and128-byte tails. Offline counter
evaluation agrees; no native backend rebuild or repeated old cohort.

Three GPU guards pass, zero allocation warnings, no recovery or separate NVML
polling. JAX uses18 passes for each of180 launches:3,240 profiler passes. Native
adds176 operation calls/74 counter passes/four ranges/200 offline metric values.
Zero game actions. Capture finishes15:15:27UTC in3m12.010s with5.4GiB peak host
memory; CPU review finishes15:18:11UTC. No GPU worker remains.

Retain the profiler's uncontrolled-cache/unmodified-clock and host-memory replay
warnings. Instrumented timings are not ordinary benchmark results. Sampled JAX
Vulkan budget headroom is7,737,835,520B, not physical free/peak VRAM.

## Decision and retained failure

Do not pursue another generic matrix geometry/staging sweep on an assumed
instruction deficit. Convolution feeding, unnecessary training-prior unrolling,
and graph materialization/dependencies remain separate evidenced issues.
Current ordinary RGB caller intervals are7.037ms posterior,10.208ms imagination/
targets,24.403ms world remainder and4.552ms behavior:46.199ms total over all256
retained measured updates. These are consecutive host intervals containing GPU
waits, not kernel costs. In particular, posterior+world is31.440ms; the staged
world program must not be counted twice. Existing JAX loop sums do not cover
equivalent complete stages, so they cannot be subtracted into a gap budget.

Next reconcile current complete execution boundaries against retained JAX
before selecting another change. No new GPU allocation follows this report;
the full root-cause/parity goal remains open.

The completion controller failed because systemd had already collected the
successful GPU transient service (`LoadState=not-found`). Queue, all guards and
journal confirm completion; the unchanged declared CPU review was started
separately and passed. Original source/log stay. This was an observer failure,
not a GPU fault, permission blocker or reason to replay any capture. Artifacts:
`runs/jax-matrix-counters-20261016.s1wgIVVr`.
