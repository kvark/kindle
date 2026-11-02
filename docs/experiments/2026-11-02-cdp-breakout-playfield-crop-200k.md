# Breakout playfield crop: fresh 200k-action comparison

Separately select one full development learner after the
[bounded crop check](../results/2026-11-02-cdp-breakout-playfield-crop.md)
passes geometry, default compatibility, 8,192-action learning/restore and all
374 pins, tensors, guided replay and host guards. Its conservative 200k-action
projection is 21m28.933s, within the existing 30-minute ceiling. No qualification
stage or bounded experience is repeated or transferred.

The larger 25M model was slower and scored worse than the retained 12M control.
This comparison instead tests a different input allocation at the same tensor
and model size: native crop **x=0, y=32, width=160, height=168**, followed by the
existing single RGB64 GPU resize. Vertical playfield scale increases; the HUD's
score/life information is removed. The pixel teacher still receives the original
native pooled image. No additional head, resize or teacher change is introduced.
Better control is a hypothesis, not a consequence established by the cost check.

Keep 12M64/Auto/N8/B16/T16/fullBPTT/microbatch16/H15/R16, encoder/base4e-5,
behavior8e-5, dynamics4e-4, replay-value1 with imagined bootstrap, expert100,
unit label weights, world gradients, all-state corrective labels, 20% episode
guidance/cap1920, actor-unimix.01, action-effects1, replay100k and warmup1000.
Preserve full18/sticky.25/repeat4/noops0. Seed1009 and evaluation4000001009 with
the eight lane offsets are reused development inputs, not fresh confirmation.
Start without transferred weights, replay, beliefs, games, RNG or experience.

Use the qualified native **acb175f**, SHA256
`a97bca1b417f192d78268f62a41f9aadaa81bf56fea436b57a8a1c654dfb68a1`,
and runner SHA256
`046aa5bc6d2f15250192345c4e77a71f85cadbc9d05ffc46113a74ca2c0ea807`.
Runtime: `/x/Temp/kindle-playfield-crop-20261102/repaired/runtime`.
Native/runner differ from retained447fe50/cf4ea238; the completed default frozen
compatibility check does not establish full default learning equivalence.
Keep that limitation and do not claim an isolated crop effect.

Train **200,000 aggregate actions /12,477 updates** under **1,800 seconds**.
First update376, final credit.5. Retain the100k/6,227 recovery checkpoint before
replacement and the final200k checkpoint. Require all357 initial tensors to match
the bounded crop initialization and its47 pre-update transitions. Initial tensors
also match the same-preset full-frame control, but input trajectories and actual
initial frozen scores can differ. Do not require semantic equality to full-frame
control. No intermediate assessment or checkpoint selection.

Assess fixed final and actual-initial sampled/greedy policies, sampled primary,
first three natural episodes in each of eight lanes: **24 per arm**. Each arm has
600k-action/300-second limits, zero updates, guidance, labels and truncations,
and357 invariant tensors. Preserve the original **90%-two-wall** task gate.
Compare with retained full-frame **70.750 sampled /143.667 greedy**, own initial
1.417/2.208, in21m27.910s. Report absolute scores, task successes and
initial-subtracted return per training minute; keep sampling-mode tradeoffs
explicit. A lower training loss or larger image objects alone cannot select
the candidate. A promising reused-seed outcome would still need confirmation.

Require exact crop metadata/architecture, configuration, source/native identity,
counters, finite reports/checkpoints, changed CNN/reward/actor/value, full native
training pixel/action/label replay, four frozen outcome replays/videos, host
guards, curves, actual guidance exposure, comparison and terminal collection.
Ordinary frozen logs lack live pixel hashes; replay cannot prove unavailable
live-image equality. CPU preflight must exercise the actual commands, crop
configuration/argument rejection, startup headers, observers, checkpoint schema,
recovery retention and all input paths before launch.

Keep600s inner/630s outer guided replay,500s per frozen replay and1200s total
CPU audit. Main service cap4500s; postprocessing/progress5100s with4800s waits;
recovery3000s with2700s internal wait; collection5400s with5100s internal wait.
Observers cannot restart or extend training. Record CPU co-host time without
correcting training duration. Declare exact files, schemas and limits before
launch. Pin the protocol snapshot rather than a mutable status document.

Check live state before launch; no overlapping GPU work. Stop on real faults or
nonfinite values without host recovery. Preserve failures and inspect completed
artifacts before a separate CPU repair. No retry, cap increase, extension, crop
sweep, seed confirmation or automatic successor is selected by this protocol.

Root `/x/Temp/kindle-playfield-crop-20261102/full`, trial
`breakout-12m-64-b16-r16-playfieldcrop-expert100-200k-seed1009`.
Main `kindle-breakout-playfield-crop-200k-20261102`, with matching postprocess,
progress, recovery and collection observers. The five-game goal is unfinished.

CPU preparation passes in2.345s (2.451s wrapper): five actual CLI commands,
38 audit/comparison configuration cases, seven prelaunch command cases,
17 startup-header cases, four observer argument cases and13 recovery-body cases.
Recovery metadata, retention and collector checks pass; all374 bounded and338
control pins verify,357 initial tensors match and18 scripts compile. No native
agent or environment is constructed. The selector's actual input list is
resolved separately before launch; no earlier qualification stage repeats.

## Original learner launched

Main starts **12:26:48.481 UTC November2**, source `53a75d7`, PID950033,
invocation `7bc8fe9931bd4f3fbee1f43739efb056`. Declaration SHA256
`57bd2e670bc3120b7e10c0262470018d728af51d4baf77fa6765ecdfd073d189`
pins413 inputs. The selector resolves410 existing files and three declared
outputs before launch. Original postprocessing, progress, recovery and collection
observers are active. Do not restart or overlap this learner.

The startup audit passes **12:28:20.927 UTC**, in1.462s including its wrapper.
All413 pins, exact configuration/cadence and357 initial tensors match. The first
47 transitions match the bounded crop check; full-frame input traces are not
required to match. The complete log prefix has882 finite supervised updates
through14,488 actions. No full frozen result is available.

Separate read-only recovery and success-only terminal readers are prepared at
the package root. Terminal-reader qualification passes five score cases and
seven crop/native/configuration cases in.043s (.062s wrapper), preserving the
original live process identity. Source preparation takes.005s. Recovery reader
SHA prefix `bc1a6f0c`; terminal reader `30763238`. Run recovery confirmation
only after the original100k audit/retention completes, and terminal confirmation
only after all original services and comparison succeed. Neither final reader
has run. Preserve CPU co-host time without correcting training duration.
No preparation failure, learner failure or repeated GPU stage occurs at startup.

## Saved-log coverage follow-up

Prepare one CPU-only reader while the original learner remains active. Run its
final stage only after the original full comparison and terminal confirmation
succeed. Reuse the completed full-frame control analysis from
`/x/Temp/kindle-breakout-score-coverage-20261102/review.json`, `trials.candidate`;
do not recount its trajectories or repeat emulator/native stages.

Keep pre-action score bins0–215,216–431,432–647,648–863 and864+. Reconcile
completed episodes, partial tails, resets, rewards and cumulative emulator-frame
differences. Count teacher overrides and native actions separately. Frozen
summaries use the original first three natural episodes per lane; extra episodes
and tails remain accounting evidence. These score fractions are not verified
wall boundaries, equivalent states or a causal account of learning quality.

Reuse the qualified cumulative-frame parser and its retained five boundary
fixtures. Check the actual crop configuration and200k/12,477 counter assertion
before outcomes, including rejection cases; do not repeat prior trajectory
analysis or fixture execution. Pin the new reader, helper, protocol snapshot,
prior evidence and active declaration. Verify all413 active pins and original
process identity around preparation. Use one low-priority CPU, no native/ALE
imports, no GPU work and no updates; retain co-host duration without correction.
Output `score-coverage-review.json` is written once. This diagnostic selects no
learner, guidance change, extension or automatic successor.

Coverage qualification passes **12:34:24.544 UTC November2**, in2.459s
(2.480s wrapper), with six configuration and three actual counter cases.
All54 diagnostic inputs and413 active pins verify; the original main identity
is preserved. Five retained parser boundary fixtures are reused without repeat.
Reader SHA prefix `2aed450b`, declaration `00d34473`. The final analysis remains
unexecuted and conditional on original full success and terminal confirmation.
No GPU work, trajectory recount, failure or learner change occurs in preparation.

## Halfway checkpoint retained

The original recovery observer succeeds **12:37:54.838 UTC November2**, under
PID950173/invocation `f27736e9ecee44a7b98d91d89c1c82ab`. The100k/6,227 checkpoint
contains357 finite tensors and retains206,067,317 bytes of weights/metadata.
The complete log prefix contains6,380 finite reports through102,456 actions.
All413 declaration inputs remain unchanged.

The prepared read-only recovery confirmation passes once at **12:38:39.553**,
in2.003s (2.023s wrapper), verifying retained hashes, exact metadata/configuration,
the original observer's single audit payload and its terminal success. Main
PID950033/invocation `7bc8fe9931bd4f3fbee1f43739efb056` remains active before and
after. No learner, observer, replay or GPU stage repeats. Retained checkpoints
omit replay, live beliefs, games and RNG; no frozen competence is measured here.
Retain CPU co-host time without correcting training duration.

Recovery review SHA prefix `b9bf4518`, retention `d0036cce`, metadata `10eaab13`;
full hashes are in the trial's `recovery-confirmation.json`. Do not repeat this
reader. Terminal and coverage final stages remain pending original completion;
the fixed200k/30-minute budget and all evaluation gates stay unchanged.
