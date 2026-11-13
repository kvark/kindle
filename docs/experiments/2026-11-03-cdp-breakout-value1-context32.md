# Longer sequences with the retained Breakout actor/critic recipe

Select one bounded execution and cost check after the completed
[four-action comparison](../results/2026-11-03-cdp-breakout-minimal-guided.md#full-200k-comparison).
That candidate saved time but lowered both frozen scores. Retain full18.
The original mean864 / 90%-two-wall gate and five-game goal are unchanged.

The earlier [B8/T32 comparison](../results/2026-11-01-cdp-breakout-context32.md)
raised greedy104.667→130.417 and greedy initial-subtracted return/min by15.75%,
but lowered sampled return, then the primary outcome. It used behavior4e-5 and
replay-value0.3. The current retained recipe uses behavior8e-5 and replay-value1,
scoring70.750 sampled /143.667 greedy in21m27.910s. Their combination with B8/T32
has not been evaluated. Neither the earlier result nor the negative context9
forward-only state-recomputation result establishes this combination's outcome.

Change batch/world-microbatch16→8 and sequence/world-backprop16→32. Keep256
sampled positions per update and ratio16. Temporal gradients, sequence diversity,
sampling, warmup and behavior-target rows240→248 change together. Cached recurrent
states already carry history; do not interpret this as added live memory or an
isolated temporal-gradient effect.

Keep centered CDP12M64/Auto/N8/H15/context1, encoder/base4e-5, dynamics4e-4,
behavior8e-5, policy1/imagined-value1/replay-value1, disagreement1, actor-unimix.01,
expert100/unit weights/world gradients and all-state corrective labels. Keep
sampled collection/posteriors, replay100k, the unchanged pixel teacher,20% guided
episodes/cap1920, full18/sticky.25/repeat4/noops0 and native pooledRGB/one GPU64
resize. Native447fe50/SHA35ad0db and runnercf4ea238 remain byte-identical. Use the
qualified runtime at `/x/Temp/kindle-breakout-corrective-20261028/runtime` and an
exact private copy of the retained runner/replay helper. No native build or
production change is required.

Start fresh at learner1009/evaluation4000001009. Transfer no weights, experience,
games, beliefs or RNG state. Train8192 actions /481 updates, first update504 and
residual credit.5, within180s. Then freeze the final checkpoint for1024 greedy
actions within60s, requiring zero updates and357 invariant tensors. This is an
execution check, not a24-episode competence assessment.

Require exact configuration,357 finite initial/final tensors, initial equality
against the retained full18 control,47 shared pre-update semantic events through
action376, finite supervised reports, changed CNN/reward/actor/value parameters,
full training pixel/action/label replay and both host guards. Check each queued
command at the mocked native-construction boundary before launch. Resolve pure
native configuration metadata before replacing both package and native agent
exports, preventing the previously recorded lazy-export preflight failure.
CPU fixtures must reject wrong config/cadence/label bounds and unequal shared
prefixes. No emulator or native agent may be constructed in preflight.

Use logged training time ×200000/8192 ×1.1 as the conservative200k projection;
require at most30min before separately selecting a full learner. Preserve actual
construction and stage times separately. Main service cap600s, training-replay
cap120s and audit cap180s; the independent terminal observer has650s. Preserve
failures, stop on real faults/nonfinite reports, and never restart successful
stages or increase caps. No driver reset, old-life continuation or GPU overlap.

A later full comparison, if separately selected, would use greedy as its fixed
primary mode and sampled as secondary, all four final/actual-initial24-natural-
episode assessments, read-only100k/6219 recovery and200k/12469 endpoint. Retain
the current recipe unless greedy return and initial-subtracted return/min both
rise and the paired lane-bootstrap gain/min interval excludes zero. Reused
seed/cohort and historical control timing are development evidence, not seed
reliability. This bounded selection creates no full learner, confirmation,
extension, context sweep or automatic successor.

Root `/x/Temp/kindle-breakout-value1-context32-20261103`; service
`kindle-breakout-value1-context32-check-20261103`. Check current process and GPU
allocation state before launch. The completed four-action run and its observers
remain closed.

## CPU qualification before launch

The repaired preflight passes21:13:39.520 UTC November3 in3.193s. Both actual
queued commands reach mocked construction boundaries; native-agent and real-
emulator construction attempts arezero,with zeroGPU descriptors. All24 negative
config/cadence/label/prefix/checkpoint fixtures,357 saved tensor shapes,47 shared
prefix events and independently derived8192/481,100k/6219,200k/12469 schedules
pass. All338 retained-control pins verify,with one exact historical replay-source
resolution. Native447fe50 and runnercf4ea238 remain unchanged.

The first CPU preflight fails on a collector fixture that expected a file-hash
map from the older checkpoint audit. Original scripts/stdout/stderr and failure
receipt remain in `preflight-attempt1` and `preflight-failure.json`. The collector
now checks the metadata hash and the three tensor hashes recorded in metadata;
two corruption fixtures exercise that exact helper. Both command-construction
checks passed before the original CPU failure. No service or GPU stage was
launched by either preflight. No full learner is selected.
