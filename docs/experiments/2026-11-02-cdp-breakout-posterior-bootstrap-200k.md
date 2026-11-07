# Fresh Breakout200k comparison: posterior replay bootstrap

Select one fresh learner after the [bounded check](../results/2026-11-02-cdp-breakout-posterior-bootstrap.md)
closes successfully. It completed8,192 actions/489 updates in46.900s; the
conservative200k projection20m59.510s passes the fixed30-minute cap. The new
runtime's default restore also reproduced128 original events and357 frozen
tensors. Do not repeat either check or reuse their experience.

Keep12M64/Auto/N8/B16/T16/H15/R16, encoder4e-5/behavior8e-5/dynamics4e-4,
replay-value1/expert100, sampled training collection and the retained teacher.
The only intended learner-setting change is `replay_value_bootstrap: posterior`:
real-replay lambda targets bootstrap from observed-state values instead of
imagined returns. Reward/bonus arithmetic, boundaries, discount1−1/333,
lambda.95, slow-value regularization, loss scales and imagined actor/value
target construction remain fixed. Critic and representation training both
consume these targets. This is a development test, not a diagnosed defect.

Use packaged native11373c1/SHAf1cf4317 and runner8952f81f from
`/x/Temp/kindle-replay-bootstrap-20261102/runtime`; do not rebuild or mix overlays.
The retained control uses native447fe50/runnercf4ea238, so no isolated causal
claim follows. Old metadata's missing bootstrap field means `imagined`.
Full18/sticky.25/repeat4/noops0, native pooled RGB and one GPU64 resize remain.
Only native Meganeura+Blade learns; Python supplies adapters and analysis.

Start learner1009 from fresh initialization for **200,000 actions / 12,477
updates**, first update at376 actions and final scheduler credit.5. Cap training
at **1,800 seconds**. Keep complete checkpoints at100k/6,227 and200k/12,477;
preserve the100k weights and metadata before final overwrite. No restored
weights, replay, live belief, games or RNG are reused. Checkpoints remain warm
starts, not uninterrupted snapshots, because they omit those live components.

Freeze the final model and its actual initial model in sampled and greedy
action modes: four arms, first three natural episodes per eight lanes=24,
evaluation seed4000001009 with original offsets. Each arm keeps600k actions /
300 seconds, zero learner updates, zero guidance/labels and zero truncations.
Require357 finite/invariant tensors, full action/reward/frame/reset replays,
videos, guards and the original90%-two-wall gate. Ordinary frozen logs lack
live pixel hashes; do not claim direct image equality. Require both actual-initial
traces and values to match retained controls; do not replace them with an older
evaluation or select a favorable intermediate checkpoint.

Compare against retained sampled **70.750**, greedy **143.667**, own initials
1.417/2.208 and training21m27.910s; all retained arms have zero two-wall successes.
Sampled is primary; report greedy separately, absolute frozen gains, actual
training time and initial-subtracted return per training minute. Keep the
stronger control if the candidate regresses; a favorable development result
does not establish seed reliability or complete the five-game goal.

Before launch, validate all five actual commands, strict config/native/runner
identities, startup gating,100k recovery/retention, service observer arguments,
collector requirements and postprocessing paths. Require the completed bounded
and retained-control pins. Pin the exact protocol, runtime, runner, teacher,
controllers, reference artifacts and source. Check live state again immediately
before launch. Automate progress, recovery, postprocessing and terminal
collection; never overlap a second GPU job with this learner.

Root `/x/Temp/kindle-replay-bootstrap-20261102/full`, trial
`breakout-12m-64-b16-r16-posteriorbootstrap-expert100-200k-seed1009`, main service
`kindle-breakout-posterior-bootstrap-200k-20261102`. Main cap4500s, postprocessing
and progress5100s, recovery3000s and collection5400s preserve the existing
controller limits. Preserve any failure; an observer timeout is not a stopped
learner. No restart, repeated stage, cap increase, extension, sweep, confirmation
or automatic successor. Reused1009/cohort is development evidence. The five-game
goal remains unfinished.

## Prelaunch qualification

CPU preflight completes09:26:27.230 UTC November2 in2.366s, before launch.
Allfive actual commands,18 controllers,34 config cases,15 startup-header cases,
seven prelaunch cases,11 recovery cases,six retention cases,two collector
corruption cases,four metadata cases andfour observer parser cases pass.
All373 bounded/338 control pins remain unchanged;357 initial tensor shapes
match. No native agent or environment is constructed. Source adaptation is
recorded separately and does not alter the qualified native,runner or teacher.
