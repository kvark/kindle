# Bounded Breakout check with stronger prior fitting

Select one fresh bounded check of dynamics KL scale1→3, leaving posterior
representation KL scale.1 and both free-nat thresholds1 unchanged. The native
loss fits the prior to a stopped posterior; its gradients also affect shared
recurrent dynamics. This is not an isolated prior-head or memory experiment.

The retained200k log has late raw KL mean11.815355, while its clipped dynamics
loss averages11.815389. The small difference provides little support for the
free-nat floor as the main late fitting limitation; aggregate logs cannot give
the exact number or importance of clipped states. Preserve and independently
recount these statistics before launch. They do not prove prior error causes
weak play. The prior-weight experiment is a learning hypothesis, not a fix
deduced from those averages. Lowering the shared dynamics optimizer rate and
reducing posterior KL weight previously failed their gameplay comparisons.

Keep centered CDP12M64/Auto/N8/B16/T16/full BPTT/micro16/H15/R16/context1,
base/encoder4e-5,dynamics4e-4,behavior8e-5,policy/value/replay-value1,
continuation1,CDPfuture500,disagreement1,expert100/unit weights/world gradients,
all-state corrective labels,20% episode guidance/cap1920,sampled collection
and posteriors,actor-unimix.01,replay100k and full18/sticky.25/repeat4/noops0.
Use native pooled RGB and one GPU64 resize. No optional reward or auxiliary loss.
No transferred weights or experience; fresh seed1009, frozen seed4000001009.

Use unchanged native447fe50/SHA35ad0db from
`/x/Temp/kindle-breakout-corrective-20261028/runtime`. Add only the exposed
`--dynamics-scale` option to an exact copy of retained runnercf4ea238; preserve
its source/diff and the corresponding repository CLI change. The native graph
already scales this loss, so no native build or new differentiation rule is
needed. Qualify default/explicit-default/scale3 configuration and invalid,
restore/evaluation overrides in both canonical and packaged runners.

Train8,192 actions/489 updates, first update376 and residual credit.5, within180s.
Then assess1,024 greedy frozen actions within60s, with guidance/labels off,
zero updates and357 invariant tensors. This assesses execution, not gameplay.
Require357 finite initial/final tensors and exact initial equality to retained
control,47 shared pre-update events,correct weighted losses/240-row label bounds,
changed CNN/reward/continuation/actor/value,complete training pixel/action/label/
override replay and both host guards. Both actual queued commands must pass
mocked construction tests before launch; no native agent/emulator in preflight.

The cost screen requires training time×200000/8192×1.1≤30min. Report actual
cost relative to retained47.690s without a reliable speed claim. Unlike the
preceding B32/T8 cost experiment, this intervention targets learning quality
and is not required to reduce short-run time by5%. Its gate is fixed before
execution. Replay cap120s,audit180s,main600s,collector650s. Preserve failures,
stop real faults/nonfinite reports,never repeat completed GPU stages or reset
the device. No concurrent GPU job,coefficient sweep or automatic successor.

Only after successful execution/cost checks may a fresh200k comparison be
selected separately:30-minute training cap,100k/6,227 recovery,12,477 final
updates and four final/actual-initial sampled/greedy24-natural-episode arms.
Greedy remains primary. Favor the candidate only if raw greedy return and
initial-subtracted gain/min improve with a positive paired lane-bootstrap
gain/min interval. Keep the original mean864/90%-two-wall gate. Reused seed/cohort,
historical timing and changed runner limit causal/reliability claims.

Root `/x/Temp/kindle-breakout-dynamics-scale3-20261104`; service
`kindle-breakout-dynamics-scale3-check-20261104`. Preparation pending at selection;
check live state. Retain143.667 greedy/70.750 sampled. No full learner or
automatic promotion is selected; the five-game goal remains unfinished.
