# Breakout encoder4e-5 in the retained 12M/expert100 recipe

The retained encoder2e-5 learner improves frozen sampled/greedy Breakout to
25.292/42.625 in22m27s, versus13.208/17.875 with6e-6 at nearly the same cost.
Fixed-state ball and action readouts also improve, while precise state use and
closed-loop control remain weak. The subsequent late-imitation decay hurts both
frozen modes; retain constant expert100. These results motivate a bounded check
of one further encoder-rate increase,2e-5→4e-5, within the existing five-game goal.

The earlier4M/B8/expert10 increase6e-6→4e-5 lowered sampled score4.292→1.458.
That negative result remains evidence against treating a larger rate as a general
fix. This check uses the distinct retained12M/B16/expert100 regime and compares
with2e-5; no architecture bottleneck or universal optimal rate is established.

## Selected bounded execution and cost check

Use centered CDP/12M64/Auto/N8/B16/T16/H15/R16,encoder4e-5,base/actor4e-5,
dynamics4e-4,warmup1000,actor-unimix.01,action-effects coefficient1,full18 actions,
sticky actions.25 and action repeat4. Keep constant expert100,world imitation
gradients,all action weights1,20% episode guidance/cap1920 and all-state
corrective labels. Native447fe50, the exact retained optional-rate runner and
pixel teacher stay unchanged. Use the qualified runtime at
`/x/Temp/kindle-breakout-corrective-20261028/runtime`; no build or backend work.
Seeds1009/eval4000001009 and the evaluation cohort are reused development inputs.
No previous weights,replay,games,belief or RNG enter the fresh learner.

Train8192 aggregate actions/489 updates,then restore for1024 greedy frozen
actions with zero updates. Require357 finite/control-matching initial tensors,
357 invariant frozen tensors,47 matched pre-update semantic events,exact config
and source hashes,first update376/residual credit.5,finite reports and changed
CNN/reward/actor/value parameters. Check raw expert world/actor agreement,
row/weight accounting,passive labels and overrides,complete input-pixel/action/
teacher-label replay and both AMD host guards. This short frozen check does not
establish task competence or a full learning result.

Training cap180s,frozen60s,CPU audit180s,whole service480s. Conservative cost gate:
`training_seconds * 200000 / 8192 * 1.10 <= 1800`. A future full comparison would
require a separate selection after these results,with a30-minute training cap
and fixed final/actual-initial sampled/greedy24-episode assessments against the
retained25.292/42.625. No full learner is selected by this bounded declaration.
No retry,cap increase,extension,rate sweep,automatic promotion or experience reuse.

Root`/x/Temp/kindle-breakout-encoder4e5-20261101/native-check`,trial
`breakout-12m-64-b16-r16-encoder4e5-expert100-check`,service
`kindle-breakout-encoder4e5-check-20261101`. CPU preflight passes both actual CLI
paths,13 actual audit-config cases,the357-tensor schema and173 retained control
pins in.425s,with zero native construction attempts or environment creation.
The runner is byte-identical to the retained2e-5 control. The host is idle at
preparation; recheck live state and boot/driver pins at selection and launch.

## Bounded result

The check succeeds03:05:01 UTC November1/sourcee11932e; collection03:05:09.
Training8192 actions/32,721 frames/489 updates takes46.899s; frozen1024/4096
frames/.899s haszero updates/no completed natural episodes. All194 pins,
357 matching initial/invariant frozen tensors,47 pre-update events,full replay,
finite reports and guards pass. Conservative200k projection20m59s passes30min.
No GPU repeat or failure occurs. This qualifies execution/cost only; see the
[result](../results/2026-11-01-cdp-breakout-encoder4e5.md). A separately selected
[fresh200k comparison](2026-11-01-cdp-breakout-encoder4e5-200k.md) measures gameplay;
no check experience is reused.
