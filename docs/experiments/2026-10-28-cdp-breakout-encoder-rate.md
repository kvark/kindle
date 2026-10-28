# Breakout with a higher CDP encoder learning rate

The completed corrective200k learner scores4.292 versus its actual initial1.375,
below unaided full18's7.083; all have zero two-wall successes. Its weighted replay
expert cross-entropy averages1.000 over the final50k actions. That descriptive
training loss motivates an optimization trial, without proving an encoder bottleneck
or treating loss as task competence.

Select a fresh bounded8192-action/994-update check with encoder learning rate
6e-6→4e-5. Keep4M64/Auto/N8/B8/T16/H15/R16/coefficient1/actor-unimix.01/full18,
base/actor-critic rate4e-5,dynamics4e-4,warmup1000,20%episode guidance/cap1920/
expert10/all18 weights1/expert-world gradients/all-state corrective labels and
edge-reset feedback teacher. Seeds1009/eval4000001009 and the evaluation cohort
are reused development inputs. Native447fe50 and runtime remain unchanged.
This is one intended learner-setting change, not a proven causal explanation.

The optional fresh-CDP `--encoder-learning-rate` flag exposes existing native
configuration; omission preserves the old scaling. Nonpositive/nonfinite or
unrepresentable rates/ratios and non-CDP use are rejected before outputs or
environments. Restore rejects the override. Focused CPU checks pass348 tests,
10 native-construction tests excluded,zero construction attempts,in1.047s.
Historical CLI/test bytes are archived; completed experiment declarations and
runners remain unchanged. No native build or backend qualification is needed.

After8192 actions,restore for1024 greedy frozen actions with zero updates. Require
357 matched control initial tensors,357 finite/invariant frozen tensors,exact
config/source/counters,first update248/no residual debt,finite reports and changed
CNN/reward/actor/value. Audit expert row/weight accounting and aligned raw world/
actor losses,all-state labels,passive actions,complete label/input-pixel/action
replay and the existing AMD host guard. This short frozen check does not measure
natural-episode competence. Native max-pooled frames retain one GPU resize.

Training cap120s,frozen60s,CPU audit180s,owning service420s. Stop on real GPU faults
or nonfinite reports; no driver recovery,retry,cap increase,extension or automatic
promotion. Check experience will not enter any later learner. No full learner is
selected yet. Root`/x/Temp/kindle-breakout-encoder-rate-20261028`,service
`kindle-breakout-encoder-rate-check-20261028`. Live state was idle before preparation.
