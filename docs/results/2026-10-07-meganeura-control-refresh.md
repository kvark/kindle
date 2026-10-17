# Latest Meganeura qualifies on the unchanged control graph

Adopt Meganeura`f104f354`, Blade`e349cddf`, native`83be73bf` for the next
CDP allocation. The [declared refresh](../experiments/2026-10-07-meganeura-control-refresh.md)
passes its established scoped checks and has essentially unchanged measured
cost. It **does not** adopt the [failed projection rewrite](2026-10-07-cdp-exploration-throughput.md)
or make that stronger full-head gate pass. No tolerances or precision policy change.

| Same12M synthetic update fixture | Previous c637 | Latest f104 |
| --- | ---: | ---: |
| Mean,256 measured updates |107.846ms |108.004ms |
| Median |107.801ms |107.957ms |
| p90 |108.491ms |108.647ms |
| Imagination dispatches |1,210 |1,210 |

One matched pair, each16 warmup plus256 measured updates, exact same source
checkpoint/config/346 initial tensors. All first outputs,272 world/behavior
metric reports and346 final tensors match exactly in this fixture. The observed
0.15% slowdown is not a meaningful speedup claim or universal trajectory proof.
No instrumentation, concurrent builds or other heavy work overlaps timing.
This is not game throughput; GPU utilization remains unmeasured.

All20 guards pass:18 qualification and two timing jobs.1,300 CDP plus1,524 RGB
upstream comparisons include raw gradients and independent optimizer/EMA.
Original/centered cosine, grouped RSSM, action-effects/detachment and ensemble
learning checks pass. Three train/frozen smokes cover1M raw CDP Pong,12M
centered CDP Breakout and1M RGB Seaquest:3,072 training actions/585 updates,
then3,072 frozen actions/zero updates. Exact frozen tensor counts are346/346/292.
Finite checkpoints, actual initial identities, counters, zero debt and device
headroom pass; minimum sampled Vulkan budget-minus-usage is12,552,241,152bytes,
not physical free or peak VRAM.

Thirteen known allocation warnings remain:12 during qualification and one
during timing, all `_memdescAllocInternal / NV_ERR_NO_MEMORY`. They do not stop
successful jobs under the user's permission. No separate NVML polling or GPU
recovery. The reference fixtures,256-step ensemble unit test and544 synthetic
full updates are additional offline work, not gameplay competence evidence.

Clippy/CPU checks and1,189 Python tests pass. The raw generic timing audit's
`retain=false` denotes failure to reach5% speedup; it is not a backend rejection
rule. This review adopts the upstream fixes with no speedup claim. Tiny video
attention is not newly qualified by these CDP/RGB checks; historical learning
keeps its original pins. No blanket F64-precision guarantee is made.

[Compact review](2026-10-07-meganeura-control-refresh.json),
[qualification audit](../../runs/meganeura-control-refresh-20261007.FZHHoQCK/qualification-audit.json),
[timing audit](../../runs/meganeura-control-refresh-20261007.FZHHoQCK/timing-audit.json),
[review script](../../runs/meganeura-control-refresh-20261007.FZHHoQCK/review.py).
Next: [the remaining four12M capacity cohorts](../experiments/2026-10-07-cdp-12m-four-game-capacity.md).
All five DreamerV3 quality targets and the budget question remain open.
