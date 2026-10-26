# Small-image Boxing: useful frozen learning within 30 minutes

The selected 1M/RGB32/500k/R32 recipe passes the existing Boxing gate on two fresh
learners: seed2017 scores76.83 in28m22s and confirmation seed4013 scores79.17 in
28m13s, each with24/24 frozen wins. The earlier200k/R16 screen fails. Two learners
support this cheap recipe without establishing broad seed reliability.

## Initial 200k/R16 screen

Fresh seed1009, centered CDP/action-effects1, Size1M/RGB32/Auto,
N8/B8/T16/H15/R16, full BPTT/microbatch8 and replay100k. Native max-pooled
frames receive one GPU resize; full18/sticky.25/repeat4, no action aid.
Actual trainable count834673. The fixed200k screen fails its frozen gate.

| Policy | Natural matches | Mean | Wins | Draws | Gate |
|---|---:|---:|---:|---:|---|
| Fixed200k final |24|−3.875|10|0|fail|
| Actual initial |24|0.000|10|4|fail|

Training completes200000 aggregate actions,799683 emulator frames and24970
updates in363.204s (6m03s),550.65 actions/s, first update at248 and zero debt.
The110 completed training matches end with last50 mean0.42. Final and initial
frozen arms each collect42864 actions, with171387 and171384 emulator frames
respectively; each selects24 natural matches. All346 tensors remain unchanged.

All24970 reports, complete counters/config/source checks, finite checkpoints,
full CPU replays/videos and host guards pass. Service
`kindle-cheap-boxing-200k-v2-20261025` exits successfully. The first controller
invocation failed before native initialization because its required argument was
omitted; the corrected launch repeats no training. That failure remains recorded.

Artifacts, complete action/frame/time curve, checkpoints and replay evidence:
`/x/Temp/kindle-cheap-20261025/boxing-1m-32-200k`.
Review SHA256`6e69968a7f7d2ce7a83136810dd4ff0103c15bc6d27a4611577220810f6d0500`.
Native`bfb8c31`/SHA`f2606ac28f5f293b73ccdb3f408278c6c584e50ccc73d52fd605a39b231609a5`,
runner`a99416e`, RX7900XT/RADV. All initial tensors match the bounded smoke.

This short screen is cheap to execute but has not demonstrated useful learning.
Compared with historical successful1M/RGB64/R32/500k controls, it changes
resolution, replay ratio, experience budget and runtime. It performs roughly
one fifth as many updates (24970 versus124939). The result cannot isolate
resolution or establish that the smaller model cannot learn. No warm-start
extension follows. The subsequent fresh run below tests the historical update
budget under the unchanged 30-minute training cap.

## Fresh 500k/R32 run

[Declaration](../experiments/2026-10-25-cdp-cheap-boxing-update-budget.md).
Fresh seed2017, 834,673 trainable parameters, Size1M/RGB32/Auto, centered CDP/
action-effects1, N8/B8/T16/H15/R32, full BPTT/microbatch8 and replay100k.
Full18/sticky.25/repeat4, native max-pooled frames and one GPU resize, no aid.
Ratio, experience budget and seed change together relative to the short screen.

| Fixed policy | Natural matches | Mean | Wins | Draws | Gate |
|---|---:|---:|---:|---:|---|
| 500k final |24|76.8333|24|0|pass|
| Actual initial |24|2.8750|15|4|fail|

Training completes 500,000 aggregate actions, 1,999,235 emulator frames and
124,939 updates in **1,702.147s (28m22s)**, 293.75 actions/s. First update248,
zero debt, 429 natural training matches; final online last50 mean87.38.
The fixed final/initial evaluations use seed4000002017 and the first three
natural sampled matches per lane, with no truncations or excess episodes.
Final evaluation uses25,328 actions/101,278 frames/11.097s; initial uses42,864
actions/171,384 frames/18.658s. Neither performs updates; all346 tensors remain
identical. The mean≥50/90%-wins gate passes for this frozen policy. Its confidence
intervals condition on one trained policy, not training-seed reliability.

All124,939 numeric reports, finite checkpoints, complete config/source/counter
checks, full CPU replays/videos and three host guards pass. Service
`kindle-cheap-boxing-r32-500k-2017-20261025` exits successfully. No cap increase,
checkpoint selection or warm-start extension. Native/runner identity is unchanged
from the short screen. Actual initial weights are saved for fresh seed2017;
they are deliberately different from the earlier1009 initial weights.

At the same500k/R32 budget and development seed2017, the
[retained1M/RGB64 control](https://github.com/kvark/kindle/blob/archive/phase2-before-cleanup-20261025/docs/results/2026-10-09-cdp-boxing-capacity.json)
takes3,938.393s (65m38s), with frozen71.625 versus initial2.5417. The current run
takes43.2% of that training wall time (2.31× lower cost) with frozen76.8333.
Image geometry, actual parameter count, runtime and GPU differ. This is a useful
measured recipe comparison, not an isolated resolution speedup or superiority
claim. Seed2017 is historically reused development data, not a held-out test.

Artifacts: `/x/Temp/kindle-cheap-20261025/boxing-1m-32-500k-r32-seed2017`.
`curve/` contains independently reconciled action/frame/time curves;
`historical-comparison/` contains both observed curves and separate frozen scores.
Review SHA256`109450faca39af5ea6b51766c53924d8568494e4dd785498289e67eb331faa34`.

## Fixed-recipe confirmation: seed4013

[Declaration](../experiments/2026-10-26-cdp-cheap-boxing-confirmation.md).
Only learner/evaluation seeds change, to4013/4000004013. The recipe, native/runner
bytes,500k-action budget and30-minute training cap are fixed before this seed.
It is a fresh learner with new replay/games/RNG and saved actual-initial weights.

| Fixed policy | Natural matches | Mean | Wins | Draws | Losses | Gate |
|---|---:|---:|---:|---:|---:|---|
| 500k final |24|79.1667|24|0|0|pass|
| Actual initial |24|−0.6250|11|2|11|fail|

Training completes500,000 aggregate actions,1,999,275 emulator frames and124,939
updates in **1,692.899s (28m13s)**,295.35 actions/s. First update248, zero debt;
286 natural training matches and final online last50 mean80.58. The250k recovery
checkpoint passes finite/config/counter checks before replacement by fixed500k.
No checkpoint selection, cap increase, retry or warm-start extension.

Final/initial evaluations select the first three natural sampled matches per
lane,24 each, with no excess episodes or truncations; unfinished tails remain.
Final uses40,192 actions/160,727 frames/17.541s; initial42,864 actions/171,384
frames/18.656s. Zero updates and all346 tensors unchanged in both arms.
All124,939 strict numeric reports, finite checkpoints, complete config/source/
counter checks, full CPU replays/videos and three host guards pass. Service
`kindle-cheap-boxing-r32-500k-4013-20261026` exits successfully.

Both frozen policies pass the existing mean≥50/90%-wins gate within30 minutes.
The confirmation learns more slowly on its online curve but reaches a slightly
higher fixed frozen score; those are different measures. Keep the two trained
policies separate rather than treating48 matches as48 independent learners.
Seed2017 is reused development data;4013 is one focused confirmation. This does
not establish broad seed reliability, an isolated resolution speedup or five-game
completion. No further Boxing allocation is needed for the current cheap screen.

Artifacts: `/x/Temp/kindle-cheap-20261025/boxing-1m-32-500k-r32-seed4013`.
`curve/` retains independently reconciled action/frame/time curves;
`seed-comparison/` retains both observed curves, separate frozen outcomes and
source hashes. Review SHA256
`74365191a4a70e173de18b3ef5c189030f9578af517d5ecd9ecc72376d8566c7`.
The next allocation is the declared
[Breakout1% native policy-mixture test](../experiments/2026-10-26-cdp-cheap-breakout-unimix.md).
