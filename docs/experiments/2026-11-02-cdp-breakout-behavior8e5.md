# Bounded Breakout check with a doubled actor/critic rate

Select one fresh execution and early-cost check of the shared behavior learning
rate4e-5→8e-5. Keep base/encoder4e-5 and dynamics4e-4. The existing native
setting changes actor andcritic optimizer rates together, including the actor
parameters synchronized with world training. It does not isolate critic learning
or establish that a rate limitation caused the observed control failures.

The current Breakout critics have weak sampled return calibration. The short
action-ranking comparison is largely uninformative anddemonstrates no gain over
uniform choice. Neither result justifies a specific new loss. This comparison
tests learning balance with the existing objective, no new parameters, native
build or per-update work. The earlier Qbert behavior-rate comparison supplies
no clear primary quality gain; it does not answer this Breakout question.

## Fixed bounded stages

Keep centered CDP/12M/RGB64/Auto/N8/B16/T16/full BPTT/microbatch16/H15/R16,
base4e-5/encoder4e-5/dynamics4e-4, warmup1000, actor-unimix.01, sampled collection
andcategorical posteriors, action-effects1, replay100k, expert100/unit weights/
world gradients/all-state corrective labels,20% episode guidance/cap1920.
Full18 actions, sticky.25, repeat4, native pooled RGB andone GPU resize stayfixed.
Learner1009/evaluation4000001009 are reused development seeds. Only
`behavior_learning_rate` changes fromunset (effective4e-5) to8e-5.

Reuse native447fe50/SHA35ad0db andthe original Breakout teacher from
`/x/Temp/kindle-breakout-corrective-20261028/runtime`. Copy the retained4e-4
control's runner byte-for-byte (SHA6c1b69fb); its existing fresh-only
`--behavior-learning-rate` flag already supports this setting. No production
runner or native edit, mixed overlay, experience reuse or checkpoint extension.

Train8,192 aggregate actions /489 updates, first update376/residual.5, with a
180-second cap andactual-initial/final checkpoints. Then freeze for1,024 greedy
actions, guidance off, zero updates,60-second cap. Audit180s/service480s.
Require357 initial tensors and47 pre-update semantic events to match the
retained encoder4e-5 control; config must differ only in the declared rate.
Verify489 finite reports, changed CNN/reward/actor/value tensors, aligned world/
actor imitation reports, full pixel/action/corrective-label training replay,
357 invariant frozen tensors andboth host guards. Preserve original identities,
input hashes andany failure. Short frozen outcomes do not establish competence.

CPU preparation checks both actual commands, unchanged runner/schema/native,
all227 retained-control pins, exact cadence andactual audit config rejection.
Do not construct a native agent or emulator during that preparation. Before
selection, require the previous ranking assessment's original closed evidence,
terminal main/collector, matching idle host andclean committed worktree.

Conservative200k cost is `training_seconds * 200000 / 8192 * 1.10`; require≤1800s
before considering a separate fresh full learner. A full comparison would retain
200k/12,477/30min,100k/6,227 recovery, actual-initial/final sampled/greedy24-episode
assessments andthe original Breakout task gate. Compare sampled56.292 primary
andgreedy104.667, allzero two-wall, at21m28s. Improving a metric on reused1009
would be development evidence, not reliable five-game success.

Root `/x/Temp/kindle-breakout-behavior8e5-20261102/native-check`; trial
`breakout-12m-64-b16-r16-behavior8e5-expert100-check`; service
`kindle-breakout-behavior8e5-check-20261102`. No retry, cap increase, rate sweep,
automatic full selection, extension or reuse of bounded training experience.
The five-game goal remains unfinished.


## CPU qualification and bounded selection

Preparation passes00:24:51.686 UTC November2/source55b56d2 in.657s (.690s
wrapper): two actual commands,15 audit config cases,357 schema tensors,227
control pins andexact489/6,227/12,477 cadence. Native/agent/emulator construction
counts arezero. The private runner matches control SHA6c1b69fb byte-for-byte.
Seven direct-global screens andthree observer fixtures pass; the observer
collects the original successful terminal run once and rejects changed orfailed
identities. No preparation failure orrepeated successful stage occurs.

Separately select the bounded learner after the ranking main/collector's original
success andan idle-host check. Keep the fixed caps andall audit requirements.
No full learner isselected. Launch ispending.
