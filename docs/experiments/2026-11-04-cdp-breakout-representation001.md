# Bounded Breakout check with representation KL scale0.01

Select one fresh execution and cost check of the existing representation KL
weight. Retained replay-value1 reports average raw KL6.215/11.217/11.457/11.815
over successive50k-action windows, close to the floored means. The penalty
therefore remains active in aggregate; these means do not reconstruct per-row
gradients. Earlier [two-batch gradient measurements](../results/2026-10-29-cdp-breakout-gradient-attribution.md)
showed opposing representation-KL and imitation directions in some groups, on
an older learner. Neither finding establishes a causal limitation. Reducing
posterior regularization is an untested learning hypothesis.

Change only `loss_scales.representation` from0.1 to0.01 in the retained learning
configuration. Keep dynamics KL1, both free-nat floors1 and all other objectives.
The representation KL differentiates the posterior against a stopped prior;
the dynamics KL separately fits the prior to a stopped posterior. This scales
an existing objective, adding no parameters or new target. It may change learned
states, optimization and later experience; no isolated mechanism is claimed.

Expose `--representation-scale` in the canonical vector runner and a private
copy of qualified runnercf4ea238. The option requires positive finite F32 and
fresh training; omission preserves defaults and restore preserves checkpoint
configuration. The private runner differs by exactly ten added lines. Reuse
native447fe50/SHA35ad0db and the corrective runtime unchanged; no native build.
Preserve exact pre-edit canonical source bytes for historical declarations.

Keep CDP12M64/Auto/N8/B16/T16/fullBPTT/micro16/H15/R16/context1,
base/encoder4e-5,dynamics4e-4,behavior8e-5,policy1/critic1/replay-value1,
continuation1,CDPfuture500,disagreement1,expert100/unit weights/world gradients,
sampled collection/posteriors,actor-unimix.01,replay100k,full18/sticky.25/repeat4/
noops0,native pooled RGB with one GPU64 resize,20% guided episodes/cap1920 and
all-state corrective labels. All optional prior/terminal/multi-step objectives
remain off. Fresh seed1009; transfer no prior experience or weights.

Train8,192 actions/489 updates within180s, first update376 and residual credit.5.
Save actual initial/final checkpoints. Require357 matching initial tensors,
47 matching pre-update events, exact configuration/cadence, finite reports,
the full weighted world total with representation0.01, changed CNN/reward/
continuation/actor/value, and full pixel/action/label replay. Record raw/floored
KL and imitation loss without treating lower loss as gameplay competence.

Restore for1,024 frozen greedy actions at seed4000001009 within60s, with guidance,
labels and updates off. Require357 invariant finite tensors, source/native/ROM
identity and host guards. This short capture need not finish natural episodes
and cannot establish task performance. CPU qualification must exercise default,
requested and invalid configurations, restore rejection, actual queued commands,
loss arithmetic, replay argv, checkpoint schema and terminal collector before
native-agent or emulator construction.

Caps: replay120s,CPU audit180s,main600s,collector650s. Preserve failures and
completed stages; no repeat, cap increase or automatic recovery. Check an idle
host before launch. Compare cost with retained47.689684s; conservative200k
projection `seconds*200000/8192*1.1` must fit30min before separately selecting
a full learner. Any full comparison needs fresh experience,100k recovery and
final/actual-initial sampled/greedy24-natural-episode assessments, greedy primary
against143.667 and sampled secondary against70.750. Original864/90%-two-wall
gate remains; no sweep, extension or automatic successor.

Root`/x/Temp/kindle-breakout-representation001-20261104`, service
`kindle-breakout-representation001-check-20261104`. The five-game goal remains
unfinished.
