# Current upstream qualifies; update speed is unchanged

**Adopt Meganeura31026833 / Blade56f0565, native7311547d.** The unchanged CDP/RGB
graphs pass qualification. The one controlled 12M timing pair measures
107.97 ->108.23ms per full update (0.25% longer), not an improvement.
No backend fix is identified as the cause of earlier learning weakness.

[Declaration](../experiments/2026-10-09-meganeura-refresh.md) ·
[Compact evidence](2026-10-09-meganeura-refresh.json) ·
[Raw review](../../runs/meganeura-refresh-20261009.EZAbGTI2/review.json).
Qualification finishes October9 at16:37:15 UTC; timing finishes16:38:45.
Both services and the independent CPU audits complete successfully. No worker
or new learning allocation remains. The five-game quality goal stays open.

## What passed

- All20 GPU guards, the existing grouped RSSM F64/output/gradient tests,
  four exploration checks and raw/centered cosine value/derivative checks.
- All1,300 CDP and1,524 RGB upstream comparisons, including raw gradients,
  optimizer and EMA. These retain their original tolerances and small fixture
  scope; they do not erase the rejected stronger full-head F64 failure.
- Three production/restore pairs: 1M raw CDP Pong, 12M centered CDP Breakout,
  1M RGB Seaquest. Each uses1,024 training actions/195 updates, then1,024 frozen
  actions/zero updates. Actual initial checkpoints, counters, finite metrics,
  native identity and replay receipts pass. All346/346/292 saved tensors remain
  exact across frozen evaluation. These short runs are not competence evidence.
- Minimum sampled Vulkan budget headroom is12,397,051,904 bytes across the
  smokes, above the fixed2GiB gate. This is estimated budget minus usage,
  not physical free/peak VRAM. Three standalone allocation warnings remain,
  all in the first grouped-block test. No GPU/API/numerical failure, recovery
  or separate NVML polling; SM utilization remains unmeasured.
- Formatting, strict Clippy,115 CPU Rust tests and1,189 Python tests pass;
  66 Rust GPU tests stay ignored during CPU preparation. The first CPU build
  correctly rejected stale backend revision labels after dependency changes.
  Updating those two constants fixes it; [failed source/log and diagnosis](../../runs/meganeura-refresh-20261009.EZAbGTI2/build-failure.json)
  remain. That attempt launched no GPU work. No numerical tolerance changed.

The corrected CPU build takes2m17s, qualification2m6s, timing1m12s. The failed
build and independent audits are additional preparation costs. Smokes add3,072
training actions/585 updates and3,072 frozen actions, plus offline gradient,
reference and timing work. No previous learner/evaluation is repeated.

## One same-graph timing pair

Both release binaries restore the exact same346 tensors/config from the
completed12M Breakout1009 checkpoint, using fixture RNG701. Each does16 warmup
and256 measured ordinary full updates, including world/behavior training,
parameter synchronization and slow EMA. Fixture generation and exports are
outside the timer. No compilation, heavy analysis or dispatch timestamp
instrumentation overlaps timing.

| Measurement | f104f354 control |31026833 candidate |
| --- | ---: | ---: |
| Mean full update |107.967ms |108.232ms |
| Median |107.936ms |108.185ms |
|90th percentile |108.584ms |108.822ms |
| Imagination dispatches |1,210 |1,210 |

Every first output is exact. All272 numerical metric reports (11,968 values)
and all346 final tensors also match exactly in this fixture. Later trajectory
equality was measured, not made a new acceptance gate. This is not universal
parity or whole-game throughput evidence. The reused audit's `retain=false`
means the speed gain is below5%, not rejection of a qualified backend refresh.

Blade descriptor reuse does not yield a measurable whole-update improvement
here. Do not launch an optimization sweep or attribute learning regressions to
backend arithmetic on this evidence. The original expanded-action graph and
Auto precision policy remain; the rejected projection rewrite stays rejected.

## Next decision

The latest fixes are qualified, but the cost problem remains. The next bounded
candidate is training sequence length: current screens use B8/T16 while the
pinned DreamerV3 default uses T64. At fixed B8/N8/R32/H15, T64 would combine four
times as many replay positions and imagination starts per update and provide
longer temporal gradients. Whether it fits and lowers cost per trained position
must be measured; neither throughput nor learning improvement is assumed.

First declare one memory/correctness/timing preflight, then review it before
allocating a matched three-seed learning comparison. This changes BPTT, batch
statistics and update scheduling, so it is not an unchanged-learning speedup.
Do not change model capacity, exploration coefficient or reward gates alongside
it. No automatic extension of the completed500k learners or restoration claim
without replay/RNG/live belief. All historical backends/results stay unchanged;
there is still no matched five-game RGB compute-saving result.
