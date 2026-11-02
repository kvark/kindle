# Breakout: fresh 200k comparison with greedy training actions

Select one fresh learner within the unfinished five-game goal. The
[bounded check](../results/2026-11-02-cdp-breakout-greedy-training.md) passes
8,192 actions /489 updates in47.418s, with a conservative200k projection of
21m13.430s. All387 pins, control-matching initial tensors, frozen invariance,
full replay and original services pass; its original terminal reader is closed.
Check experience is not reused.

The [retained sampled-collection learner](../results/2026-11-02-cdp-breakout-replay-value1.md)
scores70.750 sampled and143.667 greedy in21m27.910s, with zero of24 two-wall
successes. Its [coverage diagnostic](../results/2026-11-02-cdp-breakout-score-coverage.md)
finds only1,262 native actions at accumulated scores216–431 and zero actions
at432+. These are score ranges, not verified wall boundaries. Better greedy
frozen play motivates testing greedy collection; it may also remove useful
diversity. A result cannot isolate coverage, exploration or architecture as
its cause. Prior Qbert gains do not establish transfer to Breakout.

## Fixed recipe and budget

Only live training action selection changes sampled→greedy. Keep centered CDP,
12M64/Auto/N8/B16/T16/full BPTT/microbatch16/H15/R16, base/encoder4e-5,
behavior8e-5, dynamics4e-4, replay-value1, actor unimix.01, action-effects1,
replay100k, expert100/unit action weights/world gradients/all-state corrective
labels,20% guided episodes/cap1920. Posterior and imagined-action sampling stay
fixed. Preserve full18/sticky.25/repeat4/noops0 and one GPU resize from native
pooled RGB. Native447fe50/SHA35ad0db and the teacher remain unchanged; runner
6e03d739 is byte-identical to the qualified bounded runner. Its difference from
control cf4ea238 is the already-qualified collection CLI/dispatch change.

Start fresh with learner1009 and evaluation4000001009 plus the same eight lane
offsets. These development seeds/cohorts are reused. No weights, replay, beliefs,
games, RNG or prior experience enters the learner. Train exactly200,000 actions /
12,477 updates within1,800s. Keep100k/6,227 recovery weights, metadata and source
hashes. Save actual-initial and final weights. Checkpoints omit replay, live
beliefs, games and RNG; no uninterrupted restart is implied.

## Frozen decision and verification

Run final/actual-initial × sampled/greedy frozen assessments, guidance and labels
off. Select the first three natural episodes per eight lanes:24 per arm,
600k-action/300s caps, zero updates and truncations. Sampled remains primary.
Compare retained70.750 sampled/143.666667 greedy and actual initials1.416667/
2.208333. A useful primary development gain requires both higher sampled mean
and higher actual-initial-subtracted sampled return per training minute;
report greedy tradeoffs separately. Preserve the original90% two-wall gate
(864 points per successful episode), without changing it to a lower-score task.
Development gains do not finish the goal or establish independent reliability.

Require357 initial tensors matching control and bounded initialization. The47
pre-update training transitions must match the bounded greedy check and differ
from sampled control. Both actual-initial frozen trajectories must match the
retained control. Require identical model/loss configuration and the declared
training action mode, finite supervised reports, aligned world/behavior imitation,
changed CNN/reward/actor/critic weights, full training pixel/action/teacher-label
replay, four frozen replays/videos,357 invariant tensors per arm, host guards,
recovery, curves, exposure and comparison. Ordinary frozen logs lack live pixel
hashes; replay verifies recorded outcomes/frames/resets without claiming unavailable
live-image equality. Guided scores and replay CE are not frozen competence.
Retain CPU co-host costs without correcting measured training time.

## Execution package

Root `/x/Temp/kindle-breakout-greedy-training-20261102/full`; trial
`breakout-12m-64-b16-r16-replayvalue1-greedy-expert100-200k-seed1009`; main
`kindle-breakout-greedy-training-200k-20261102`, with original progress,
recovery, postprocessing and terminal-collection observers. Main cap4500s,
audit1200s, post/progress5100s, recovery3000s, collection5400s. Keep the existing
internal deadlines. Pin sources, schemas, fixed protocol, queue, runtime and
reference evidence before launch. No overlap, retry, extension, cap increase,
checkpoint selection, confirmation or automatic successor.

The18 copied/adapted scripts compile before being written. The selector and CPU
preflight share the actual declaration input-path function, including the correct
bounded `preflight.json` name. This addresses the preceding package's historical
filename failure without changing or repeating that completed package. The
bounded CPU wrapper-sentinel failure and its separate qualification remain pinned.
Full CPU qualification must pass before declaration and launch.

## Full CPU qualification

The first full preparation and qualification pass at05:09:23.733 UTC November2,
in2.326s (2.487s wrapper), with no full-package failure or repair. Five actual
commands,32 config cases,16 header cases,12 recovery-body cases, four observer
parsers, four metadata cases, six retention/two collector fixtures and three
mode-audit cases pass. The actual prelaunch function passes its valid command
and rejects changed/missing replay-value, behavior and collection-mode options.
All18 scripts compile; seven global screens,44 path checks and805 dynamically
assembled declaration inputs resolve, allowing only explicitly named future
outputs. All387 bounded and338 control pins stay unchanged. No native module,
agent or emulator is constructed. Preflight SHA256:
`0868dd6def87573ff8a0452838bef1e0a0ff1135900df15e57858f33973a5923`.
