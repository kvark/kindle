# Performance screen: wider reduced-precision learning

This is a new bounded performance study, not a restart of the stopped Pong
allocation. Test the faster recipe on **Freeway**, where keeping exploration
useful matters. No new capacity, reward, encoder or game-protocol change.

## Allocation

- Three paired fresh learner seeds1009/2017/3019,200,000 actions each:
  **control B8/Auto** versus **candidate B32/F16**. Both use the newly optimized
  exploration algebra and convolution splitting on Meganeurad7a7293/Blade56f0565.
  The old full-cost runtime remains a timing reference, not a new learning arm.
- Same12M centered CDP/action-effects recipe, four ensemble heads, coefficient1,
  N8/T16/H15/R32, full-batch world training, replay100000 and original learning
  rates/AGC/warmup. Native frames, one GPU resize,18 actions, sticky.25/repeat4,
  no no-ops, action hints or additional reward. No concurrent GPU learners.
- B8 starts at248 actions:49,939 updates, final debt0. B32 starts at632:
  12,461 updates, final debt.5. Total1.2M actions/187,200 updates. Replay ratio
  is unchanged, but centering statistics, prefill and optimizer/warmup/EMA
  clocks per interaction change. This tests the **batch/precision package**,
  not the independent causal effect of each part.
- Preserve both actual initial checkpoints and require identical initial
  tensors within each seed. Alternate arm order by seed. Audit each learner,
  then its final and actual-initial frozen cohorts before the next learner.
  Do not restore a historical lifetime or extend a weak run.

## Evaluation and decision

Use sampled frozen policies, first3 **natural** episodes per each of8 streams,
seed base4,000,000,000+learner seed,200k-action/30min caps. Require zero updates,
exact346 saved tensors, full CPU replay and whole stream-zero videos. Retain
all excess episodes, cutoffs and unfinished tails; select no cutoff. These are
development seeds, not a final unseen test set.

Report paired final/initial means, learner-seed bootstrap intervals, complete
online score-versus-actions/time curves and censored time to online mean20.
Numerical checks alone do not establish faster learning. The provisional
screen passes only if every candidate improves over its own initial, no
candidate scores zero when its control exceeds5, mean paired frozen loss is
at most2 crossings, and aggregate candidate training time is<=80% of control.
Keep uncertainty and all individual failures; this small screen is not a
statistical noninferiority certificate or five-game promotion. No automatic
budget extension or precision/batch sweep follows a failure.

Launch only after the numerical checks, isolated timings and real train/frozen
preflight pass. Expected about4–5h total;2h learner/30min frozen/8h service
bounds. Persistent systemd service, Restart=no, KillMode=control-group, guarded
serial GPU jobs with no CPU quota; CPU audits/replays use one CPU/2GiB/zero swap.
Require RTX5080 and>=2GiB sampled Vulkan budget headroom. Record allocation
warnings; API/numerical/hard-fault/deadline failures stop for review.

Artifacts: `runs/cdp-performance-20261011.7SpyzGUa`; declaration pins the exact
native binary and exercised scripts before launch. PR31 remains the dashboard.
