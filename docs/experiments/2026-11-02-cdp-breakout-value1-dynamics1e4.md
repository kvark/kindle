# Breakout: lower dynamics rate with replay-value scale 1

Select one fresh 8,192-action execution and cost check within the unfinished
five-game goal. Qbert's fixed R20 recipe has completed its fresh confirmation;
all original services are terminal. Check live state before selection and launch.

The retained [replay-value1 control](../results/2026-11-02-cdp-breakout-replay-value1.md)
scores 70.750 sampled /143.667 greedy in21m27.910s, with zero two-wall successes.
The earlier [lower dynamics-rate comparison](../results/2026-11-01-cdp-breakout-dynamics1e4.md)
raises greedy104.667→138.750 but lowers sampled56.292→50.625 at behavior4e-5 and
replay-value.3. Its mixed result remains valid. Behavior8e-5 and replay-value1
subsequently improved both modes at dynamics4e-4. Their combination with1e-4 is
untested; benefits need not compose. This is a separately justified development
comparison, not an extension of an earlier learner or a rate sweep.

## Fixed change

Change only dynamics learning rate4e-4→1e-4 from the replay-value1 control.
Keep centered CDP12M64/Auto/N8/B16/T16/full BPTT/microbatch16/H15/R16,
base/encoder4e-5, behavior8e-5, replay-value1 with world gradients, sampled
collection and posteriors, actor unimix.01, replay100k and action-effects1.
Keep expert100/unit action weights/world gradients, all-state corrective labels,
20% episode guidance/cap1920, full18/sticky.25/repeat4/noops0 and one GPU resize
from native pooled RGB. Parameter count remains16,334,353.

Use development learner1009/evaluation4000001009 with the same eight lane offsets.
No weights, replay, beliefs, games, RNG or prior experience are reused. The
cohort is reused; no independent reliability or isolated mechanism is implied.
The dynamics group includes recurrent/predictive world components; a later
gameplay change would not isolate a particular layer or critic effect.

Reuse native447fe50/SHA35ad0db and the byte-identical Breakout teacher at
`/x/Temp/kindle-breakout-corrective-20261028/runtime`. No build or production
change. The exact control runner gains only the existing twelve-line fresh-CDP
dynamics-rate option from its qualified source; omitted defaults and frozen
restore behavior remain unchanged. Qualify the private runner and actual commands
before declaring the check.

## Bounds and evidence

- Train8,192 aggregate actions within180s:489 updates, first at376, residual.5.
  Retain actual-initial and final checkpoints.
- Restore final weights for1,024 greedy frozen actions within60s, guidance and
  labels off, zero updates. This short budget does not test competence.
- Require357 finite control-matching initial tensors,47 matching pre-update
  semantic events,357 invariant frozen tensors, exact config/cadence, finite
  aligned imitation reports and changed CNN/reward/actor/value parameters.
- Require complete training action/pixel/teacher-label replay, native/source
  identities and host guards. Replay cap120s/process150s, audit180s, main480s,
  collector600s. No concurrent GPU allocation or repeated completed stage.
- Cost gate: check training seconds×200000/8192×1.10≤1800. Passing establishes
  execution and early cost only; it does not automatically select a full learner.

No retry, extension, cap increase, rate sweep or check-experience reuse.
A separately selected fresh200k comparison would retain100k/6,227 recovery and
four final/actual-initial sampled/greedy24-natural-episode assessments against
70.750/143.667, with sampled primary and both modes reported. The original
90%-two-wall/864 task gate remains unchanged. No result or confirmation is
inferred from finite losses or fast execution.

Package `/x/Temp/kindle-breakout-value1-dynamics1e4-20261102`; check trial
`native-check/breakout-12m-64-b16-r16-value1-dynamics1e4-expert100-check`.
Main `kindle-breakout-value1-dynamics1e4-check-20261102` and original
`kindle-breakout-value1-dynamics1e4-check-collection-20261102` observer.
Pin the completed control and preceding Qbert closure, exact commands, protocol
snapshot, schema, private source diff, tests and source/native/teacher bytes.

## CPU qualification

The private runner adds exactly12 lines and deletes none, copied verbatim from
the qualified dynamics option. SHAea2bf8c6; native andteacher are unchanged.
All32 targeted dynamics/replay-value/rate CPU tests pass in.05s (.278s wrapper).
The actual-command preflight passes in.996s (1.028s wrapper): two commands,17
configuration cases,357 tensor shapes, exact489/6,227/12,477 cadence and338
unchanged control pins. No native module or emulator is constructed.

The first source generator stops before writing any runner because its expected
help-text snippet differs from the qualified source. Its.022s failure and exact
source remain retained. The separate corrected generator uses the exact snippet
and completes in.029s. No successful test or experiment stage repeats. All
completed CPU qualification is retained; do not rerun it.

## Completed bounded check

The [result](../results/2026-11-02-cdp-breakout-value1-dynamics1e4.md) passes8,192
actions/489 updates in48.489s;1,024 frozen/.861s haszero updates/no natural episodes.
All382 pins,tensors,pre-update traces,full replay andguards pass. Main/collector
succeed06:39:55.422/06:40:01.115 UTC November2/sourcef15e58e; terminal reader
passes once06:40:38.369. No stage repeats. The separately selected
[fresh200k comparison](2026-11-02-cdp-breakout-value1-dynamics1e4-200k.md) uses
no check experience. Completed qualification,stages andreaders must not repeat.
