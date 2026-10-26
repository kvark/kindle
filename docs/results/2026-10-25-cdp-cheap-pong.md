# Intermediate Pong learns to win with a larger update budget

The4M/RGB48/R32 recipe reaches **11.0 versus actual initial−20.375, with23/24
frozen wins**, after500k actions in48m53s. The mean-score target of15 remains
unmet. All audits pass. This is a useful compact-model result on one reused
development seed, not broad reliability or a demonstrated hardware speedup.

## Short R16 screen

The fixed200k-action screen improves the frozen score but fails the Pong gate.
Fresh seed1009, centered CDP/action-effects1, Size4M/RGB48/Auto,
N8/B8/T16/H15/R16, full BPTT/microbatch8, replay100k, full18/sticky.25/repeat4.
Native max-pooled frames receive one GPU resize; no action aid. Actual trainable
count3,578,225. All346 initial tensors match the bounded smoke.

| Policy | Selected natural matches | Mean | Wins | Gate |
|---|---:|---:|---:|---|
| Fixed200k final |24|−17.125|0|fail|
| Actual initial |24|−20.375|0|fail|

Training completes200000 aggregate actions,799850 emulator frames and24970
updates in621.700s (10m22s),321.70 actions/s, first update at248 and zero debt.
There are123 completed training matches; last50 mean−17.56. The selected frozen
gain is3.25 points, with no wins. It does not meet the mean≥15/90%-wins gate.

Final frozen collection uses68880 actions/275483 emulator frames in25.69s and
collects25 natural matches. Selection remains the first three per lane; the one
excess match and all tails remain in the full replay/video. Actual initial uses
24200 actions/96771 frames in9.05s and collects24 matches. Both have zero updates,
no truncations and exact equality of all346 frozen tensors.

All24970 reports, counters/config/source checks, finite checkpoints, full CPU
action replays/videos and host guards pass. Service
`kindle-cheap-pong-200k-20261025` exits successfully. Artifacts and independently
reconciled action/frame/time curve:
`/x/Temp/kindle-cheap-20261025/pong-4m-48-200k`.
Review SHA256`1382642300c86ff51d61924997c6db618f685523fa062f422c77b5bb2461e648`.
Native`bfb8c31`/SHA`f2606ac28f5f293b73ccdb3f408278c6c584e50ccc73d52fd605a39b231609a5`,
runner`a99416e`, RX7900XT/RADV.

This is one reused development seed with changed capacity, resolution, replay
ratio and runtime relative to historical controls. A small frozen gain is useful
evidence, not reliable winning or a matched-resolution ablation. No unchanged
extension follows. The later fresh recipe test is reported below.

## R32 and500k actions

The [declared test](../experiments/2026-10-26-cdp-cheap-pong-update-budget.md) keeps
the model, image geometry, native binary, seed1009 and all346 actual initial
tensors. Replay ratio16→32 and experience budget200k→500k change together.
Centered CDP/action-effects1, Auto,3,578,225 parameters,N8/B8/T16/H15, full
BPTT/microbatch8, replay100k, full18/sticky.25/repeat4 and one GPU resize remain.
There is no action aid; actor_unimix stays0.

Training completes **500,000 aggregate actions /1,999,756 emulator frames /
124,939 updates in2,933.450s (48m53s)**,170.45 actions/s. First update248, zero
debt,157 natural training matches, final online last50 mean8.66. The predeclared
55-minute cap is met. The100k/200k/300k/400k recovery checkpoints each pass finite,
configuration and counter checks before replacement; evaluation uses fixed500k
final weights. No selected checkpoint, cap increase or extension occurs.

| Fixed policy | Selected natural matches | Mean | Wins / losses | Gate |
|---|---:|---:|---:|---|
|500k final|24|11.000|23 /1|fail: mean below15|
|Actual initial|24|−20.375|0 /24|fail|

Frozen seed4000001009, first three natural sampled matches per lane. Final
collection uses119,416 actions/477,607 frames in44.441s, collecting26 natural
matches; the two excess matches and unfinished tails remain in the full replay.
Actual initial uses24,200 actions/96,771 frames in9.078s and collects24 matches.
Both arms have zero updates or truncations, and all346 tensors remain unchanged.

All124,939 numeric reports, complete source/configuration/counters, initial/final
checkpoints, full CPU replays/videos and three host guards pass. Service
`kindle-cheap-pong-48-r32-500k-20261026` exits successfully. Independent
action/frame/time curves and the R16/R32 comparison by actions, updates and wall
time are retained. The comparison does not extrapolate the short control or
treat the differently budgeted frozen endpoints as a replay-ratio-only effect.

Artifacts: `/x/Temp/kindle-cheap-20261025/pong-4m-48-500k-r32`, including
`learning-review.json`, `curve/`, `budget-comparison/` and actual initial/final
weights. Review SHA256
`8aa8857a90d050bda830b475a2e8b392bb3c06bfce04e35687bd8f953f68c506`.
Source at launch`4918f78`, native`bfb8c31`, runner`bcef410`, same native SHA above.

This is a fresh life on reused development data. The recipe learns to win most
matches but does not meet the complete score gate. Historical successful500k
learners took about37 minutes on different GPU/precision/model settings; smaller
parameter count has not established a speedup here. Preserve the useful result
without extending it unchanged. The focused Freeway seed confirmation follows
after review; Pong's remaining score gap stays open.
