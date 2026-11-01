# Fresh 200k Qbert comparison with greedy training actions

The [training-history readout](../results/2026-11-01-cdp-qbert-training-readout.md)
finds weaker movement fit on later autonomous states than guided states. Only
159 wholly unguided training actions reach episode age 1920+, while the retained
frozen greedy policy lasts longer and scores 14,028.125 versus 11,522.917 sampled.
Select one fresh learner to test whether greedy collection improves frozen
gameplay at the same budget. This changes exploration and visited states
together; it may also reduce useful diversity. Coverage is a hypothesis.

The [bounded check](../results/2026-11-01-cdp-qbert-greedy-training.md) completes
8192 actions / 489 updates in 66.241s and passes execution, full replay and
frozen-invariance checks. Its conservative 200k projection is 29m39s, below the
35-minute cap. The existing sampled-training control takes 27m41s. Neither
short-check execution nor guided training scores establish gameplay benefit.

Change only live training action selection from sampled to greedy, using
`--training-action-mode greedy`. Keep centered CDP, 12M48, Auto precision,
N8/B16/T16/H15/R16, encoder/base/actor rates 4e-5, dynamics rate 4e-4,
expert scale 100, all action weights 1, world imitation gradients, all-state
corrective labels, 20% episode continuation guidance capped at 1920 actions,
actor-unimix 0, full 18 actions, sticky probability .25 and action repeat 4.
Keep native max-pooled frames and one GPU resize to 48. The categorical posterior
and imagined actor still sample; the new mode controls live action collection.

Use native447fe50 and the qualified runtime
`/x/Temp/kindle-breakout-corrective-20261028/runtime`. The private runner is the
exact bounded-check runner, SHA256
`aba4300d0cd1a7d1a8d2d0c71de4a0cad90ec84b2019060fef4e6ef2db501a13`.
Compared with the retained control runner, it adds only the fresh-training
option, validation, header field and three action-dispatch substitutions. Native
model, losses and teacher are unchanged; no build is needed.

Train 200,000 aggregate actions / 12,477 updates under 2100s. Require first
update 376, residual credit .5 and a verified 100k / 6227 recovery checkpoint.
No prior weights, replay, belief, game state, RNG or bounded-check experience
enters this learner. Seed4013 and evaluation4000004013 are reused development
inputs, not independent confirmation. No checkpoint selection, retry, cap
increase, extension or automatic successor is selected.

Assess fixed final and actual-initial checkpoints with sampled and greedy
actions: first three natural episodes per eight lanes, 24 per arm, guidance
and labels off, zero updates and truncations. Greedy remains primary; retain
sampled separately. Compare against 14,028.125 greedy / 11,522.917 sampled and
the unchanged mean15,000 / 90%-first-pyramid gate. Each arm has 300s and the
existing 600k-action ceiling. Preserve excess episodes and partial tails.
Report both frozen modes, own-initial-subtracted gain per training minute,
first-pyramid counts and total cost. Online curves include teacher actions.

Require 357 control/check-matching initial tensors and 357 invariant finite
tensors per frozen arm. Pre-update training traces must match the greedy bounded
check; record their expected differences from the sampled control. Actual-initial
sampled and greedy frozen traces must match the retained control. Check exact
sources/configuration/counters, finite reports and changed CNN/reward/actor/value,
raw expert world/actor losses, labeled-row weights and actual override exposure.
Complete training pixel/action/label replay, frozen outcome replays/videos,
host guards, curves, coverage, exposure and comparison. Ordinary frozen logs
lack live pixel hashes; their replay cannot establish unavailable image equality.

The CPU preflight takes 1.832s: five actual CLI paths, 24 configuration cases,
eight training-mode audit cases, 11 startup-header cases, ten recovery cases,
four observer-parser cases, 50 paths, 15 script compilations, 357 schema tensors
and all 305 bounded / 276 control pins pass. No native model or emulator is
constructed. Missing, sampled and invalid training modes are rejected in the
candidate checks; frozen commands omit the training option. Source review
corrects a unit-prefix path substitution before this successful preflight.

Training replay cap 1200s, enclosing audit 1800s, whole main service 5400s.
Progress, recovery, postprocessing and terminal-collection observers retain
their bounded deadlines and never start or restart GPU stages. Preserve failures;
CPU repairs must not repeat completed training, evaluation, replay or plot stages.
Check live state before launch; do not schedule another GPU job while active.

Root `/x/Temp/kindle-qbert-greedy-training-20261101/full`, trial
`qbert-12m-48-b16-r16-encoder4e5-greedy-expert100-200k-seed4013`, service
`kindle-qbert-greedy-training-200k-20261101`. This investigation is part of the
existing five-game goal, which remains unfinished.

## Launch and startup

The original learner launches 06:15:51 UTC November1, sourcec659423, with317
pinned inputs. MainPID755504, invocation `5fb0d30ffb5d47aca552009e586bb929`;
progress, recovery, postprocessing and terminal-collection observers are active.
The .948s read-only startup review verifies all357 initial tensors against both
control and bounded check, all47 pre-update semantic events against the greedy
bounded run, and records47 expected differences from sampled collection.
First updates occur at376/392/408/424/440/456;105 reports through2048 actions
are finite and supervised. Configuration, sources and original process identity
match. Keep CPU co-host timing without correcting training wall time.
No full frozen result is available. Inspect this original run; never restart it
or allocate another GPU job while it is active.

## Halfway recovery checkpoint verified

The original recovery observer succeeds06:29:57 UTC November1:100,000 actions /
6,227 updates,357 finite tensors. Its complete-log-prefix check covers6,334 finite
reports through101,712 actions while the original learner continues.

A separate .788s CPU confirmation at06:30:27 verifies all317 pinned inputs,
unchanged originalPID755504/invocation5fb0d30ffb5d47aca552009e586bb929,exact
configuration and greedy-collection header,and all three checkpoint tensor-file
hashes. Exact100k metadata is retained as `recovery-metadata-100k.json`,SHA256
`74bfcfe034b1fb031bea07985e68f3dc1ab9285d8455fec255b9e31a6eb09e48`.
Recovery reviewSHA256`6da37804fe8748a7979b813eae4728ebfc7cb757e89c0f2c2c0c19c537b7f215`.
No GPU work or training restart occurs;do not repeat this confirmation.

The independent `review_terminal.py` preflight passes in.056s using the retained
control's four fixed24-episode evaluation arms and videos,and its five original
service identities. Four journal fixtures cover null messages,duplicate starts,
foreign invocations and failure messages. A prior CPU preflight assertion required
a systemd accounting row from every observer;successful progress/recovery
services have no such row. Preserve the original source and failure;the corrected
check still requires terminal success,unique original invocation and main end
record,with exact collector-child output. No original pipeline stage repeats.
Use the qualified scriptSHA256
`131bee461b853a96936667aa10d60915609178b268343bc5fc3ca70529fc0eba`
only after all original services finish;it has not run final mode yet.

The completed CPU Breakout intervention and these confirmations co-host with
training;retain their measured times without correction. Progress,postprocessing
and collection observers continue. No full frozen result or successor is selected.
