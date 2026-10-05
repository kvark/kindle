# Freeway: action-effects disagreement discovers rewards in all three seeds

At the same32,768-action/8,131-update budget, the new bonus collects2/1/1 real
crossings across seeds1009/2017/3019. All three previous visual-target and all
three extrinsic-only controls collected zero. This is the first positive
unassisted CDP exploration result, **not Freeway unlocked or reliable mastery**.

| Seed | Crossings | First reward at aggregate actions | Online mean (16 episodes) | Last-quarter UP / DOWN |
| --- | ---: | ---: | ---: | ---: |
|1009|2|28,504|.125|42.4% /30.5%|
|2017|1|32,416|.0625|38.5% /31.3%|
|3019|1|6,240|.0625|34.0% /38.0%|

All48 completed episodes and every unfinished tail are retained. Mean online
score .08333, paired difference versus retained visual targets .08333,
seed-bootstrap95% interval[.0625,.125]. Only three learner replicates; neither
episodes nor streams are independent seeds. Four rewards are weak evidence,
not a precise performance estimate or a claim that each actor retained a skill.

The fixed8,192-action windows show different trajectories:1009/2017 improve
late, whereas3019 reaches the far side early and then regresses toward DOWN.
That seed's last-quarter mean height is19.65 versus32.06 in its first quarter.
Do not hide this stability problem behind the aggregate discovery rate.
Full curves and all four windows are in the JSON/raw artifacts.

All three guards, accounting/finite-checkpoint audits and full trajectory CPU
replays pass. The GPU learner has no new encoder, learned parameter, game label,
action override, reward shaping, normalization or budget change. Only the bonus
now measures disagreement after each head's all-action mean is removed.
1,529 of1,536 sampled raw frames independently match RAM-derived player position;
seven ambiguous/invisible samples remain unchecked. Original pixels are
unavailable; no original-image equality claim. Labels are diagnostics only.

Mean wall273.25s and update31.70ms. Paired wall increase versus retained visual
targets25.34%, interval[23.18,29.11]. This is descriptive, not matched timing.
The whole three-seed screen plus automated coverage finishes in about15minutes.
All eight streams execute16,384 real emulator frames each: throughput is
7.98–8.00x aggregate real time, .998–1.000x per stream while learning. These are
frame-clock ratios, not GPU utilization; utilization is unmeasured. No separate
NVML polling or recovery.

## Next: retention, not a longer run yet

The declared positive branch is running: all three frozen candidates, three
retained extrinsic-only controls and three actual initial-weight controls.
Each uses the same held-out environment seeds, first3 natural episodes/stream
(24), cap200k actions/30min, sampled policy, zero updates, exact checkpoint tensor
equality and complete stream0 videos. Retained controls are evaluated on the
current runtime; their original training binary identity is preserved and the
zero-ensemble equivalence qualification applies. No training extension yet.

[Declaration](../experiments/2026-10-05-freeway-action-effects.md) ·
[Qualification](2026-10-05-freeway-action-effects-qualification.md) ·
[Compact evidence](2026-10-05-freeway-action-effects-learning.json).
Artifacts: `runs/freeway-action-effects-20261005.z9KUOhCS`.
