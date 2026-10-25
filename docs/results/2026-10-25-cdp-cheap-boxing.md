# Small-image Boxing: fast execution, failed learning screen

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

This recipe is cheap to execute but has not demonstrated useful learning.
Compared with historical successful1M/RGB64/R32/500k controls, it changes
resolution, replay ratio, experience budget and runtime. It performs roughly
one fifth as many updates (24970 versus124939). The result cannot isolate
resolution or establish that the smaller model cannot learn. No warm-start
extension follows. The next declared screen tests intermediate-capacity Pong.
