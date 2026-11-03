# Fresh Breakout 200k life-bounded actor comparison

The [qualified prototype](../results/2026-11-03-cdp-life-return-actor.md) passes
native derivatives, legacy compatibility, complete raw-life/pixel replay and
bounded cost checks. Separately select one fresh 200k-action learner to measure
frozen gameplay benefit within the existing five-game goal. The preceding short
run and its repaired CPU collection remain closed; do not repeat them.

Use native `ec12d41e` and the byte-identical qualified runner/runtime under
`/x/Temp/kindle-life-return-actor-20261103`. Keep the bounded candidate's exact
12M64/Auto/N8/B16/T16/full BPTT/micro16/H15/R16/context1 configuration,
encoder4e-5/dynamics4e-4/behavior8e-5, policy1/critic1/replay-value1,
expert100 with world gradients, disagreement1, actor100/observed lower-bound
and auxiliary return boundaries enabled. Keep sampled collection/posteriors,
full18/sticky.25/repeat4/noops0/native pooled RGB/one GPU64 resize,
teacher20%/cap1920, seed1009 and evaluation seed4000001009. Transfer no weights,
replay, live state or experience from any previous learner.

Training stops at 200k actions / 12,477 updates or 30 minutes, whichever comes
first. Save the actual zero-experience initial and a read-only100k/6,227
checkpoint with matching metadata and optimizer moments. These are recovery
artifacts, not exact resume state: replay, games, beliefs and full RNG state are
absent. Require first update376, positive actor targets first at568, all357
initial tensors matching the bounded candidate and retained baseline, and all47
pre-update semantic events matching the bounded candidate, including life fields.

Assess final and actual-initial checkpoints with sampled and greedy actions.
Each arm uses the first three natural episodes in each of eight lanes:24 per
arm, no timeout substitution, zero updates and357 invariant tensors. Keep the
retained600k-action/300-second bounds and100k-emulator-frame episode limit.
Evaluate without guidance or the life-boundary adapter. Preserve actual game
rewards and termination throughout training and evaluation.

Require full training pixel/action/teacher-label replay and independently
reconstructed raw-frame life boundaries; cap this CPU replay at900 seconds.
Preserve full recorded-action frozen replays and videos, each capped at500
seconds. Require finite reports, actor loss/target accounting, source/runtime/
ROM/host guards, learning curves, guidance exposure and score-band coverage.
Report life loss/gain totals and target availability separately from gameplay.
Online aided returns and repeated replay rows are not unaided competence.

Primary comparison remains retained200k sampled-collection greedy143.666667,
own initial2.208333, in1287.910495 seconds. Sampled70.750/own initial1.416667 is
secondary. Reuse its audited arms. Also report a fixed contextual comparison
against the completed whole-game actor: greedy144.750/sample49.166667 in
1306.723854 seconds, with the same own initial means. Reuse those arms too.

For both comparisons, report absolute return and initial-subtracted return per
actual training minute, plus10,000 fixed paired lane-bootstrap resamples with
seed0 and95% intervals. Favor this candidate over the primary retained recipe
only if greedy mean and gain/min both increase and the greedy gain/min interval
excludes zero. The contextual comparison does not replace that decision rule.
Check the unchanged mean864/90%-two-wall task gate directly. Native revisions
differ; exact initial/short compatibility does not establish full equivalence.
One reused development seed/cohort does not establish an isolated mechanism or
training-seed reliability. A confirmation requires a later selection.

Before launch, exercise the actual CPU postprocessing and comparison helpers on
compatible retained fixtures. Check queued commands, exact configuration,
schemas, hashes, caps and collector logic, including the qualified short check's
`complete_with_cpu_collection_repair` closure and list-shaped source manifest.
Pin the original failure and separate repair. Keep the original helpers intact.

Root `/x/Temp/kindle-life-return-actor-20261103/full`; service
`kindle-life-return-actor-200k-20261103`. The main controller validates startup
and100k recovery; separate CPU recovery and terminal observers preserve process
identity and never launch GPU work. Main/terminal caps are7200/7500 seconds.
Retain CPU co-hosting durations without wall-time correction. Check live state
before launch. No overlapping learner, successful-stage repeat, cap increase,
coefficient sweep, extension, second learner or automatic successor is selected.
