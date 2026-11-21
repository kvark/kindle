# Frozen forecasts before retained Breakout failures

Within the five-game goal, test the current143.667-point greedy checkpoint on
the saved [teacher-takeover branches](../results/2026-11-04-cdp-breakout-command-takeovers.md).
No learner, objective change or automatic successor is selected. Keep the
original864/90%-two-wall gate and retained70.750 sampled/143.667 greedy results.

Reuse all120 fixed origins eight decisions before recorded life loss, from the
first three natural episodes per eight lanes. Compare the recorded nine-action
sequence with the first nine actions of the existing teacher branch. Every
branch already contains at least nine actions. The teacher avoids92 losses at
that endpoint; among24 recorded game-overs it avoids18. These outcomes and
origins are known before model capture and must not be replaced. Omit the
rejected alternative teacher. Reuse the completed emulator branches; do not
rerun the CPU oracle or prior collectors.

Use native447fe50/SHA35ad0db and the original corrective runtime, restoring the
exact replay-value1 checkpoint. Re-observe all43,424 recorded source actions/
173,660 frames in scalar lane order, forcing recorded actions and retaining
natural resets. Verify every input/output pixel against the saved annotation,
every reward/action/frame/boundary against the original vector log, all120
origin identities and all18-action mappings. The scalar posterior draws differ
from the original vector actor; do not claim exact original beliefs.

At each origin, call the existing read-only prior-behavior rollout for both
nine-action plans. Record rewards, continuations, endpoint values and origin
actor probabilities. Repeat the recorded-plan forecast exactly and verify that
live features, actor probabilities and counters remain unchanged. The diagnostic
uses paired deterministic categorical draws; these are not independent samples.
Require finite outputs, continuations in[0,1], zero learning updates and all357
checkpoint tensors byte-identical after capture. Life counts are diagnostic
labels only and never policy/model inputs. No planning or teacher control is
used on the source trajectory.

The primary question is **game-over discrimination**. The continuation head is
trained on natural game termination, not every lost life. Compare products of
the nine predicted continuations with gamma^9 times actual endpoint survival,
where gamma is the model's F32(1−1/333). Report both arms' squared errors and
means on24 final-life origins, plus paired preference on the18 differing
game-over outcomes; ties within1e-6 receive half credit. Report per-lane counts
and fixed5,000-resample lane intervals, seed20261104, conditional on these states.
Intermediate-life continuations are a separate descriptive stratum.

Also report nine-step external reward forecasts using preceding predicted
continuations, against actual gamma-discounted branch rewards, with the number
of informative reward pairs. Report whether reward plus discounted endpoint
critic value favors the teacher at92 avoided-loss origins. That latter statistic
is only a preference for known short rescues: the critic includes intrinsic
reward while forecast rewards omit it, and complete alternative-policy returns
are unavailable. It is not calibration of the full actor objective, an optimal
action ranking, or evidence that a planner improves gameplay.

This is an outcome-conditioned diagnostic on one reused learner/cohort. Teacher
sequences were generated adaptively in the real branches and are supplied as
fixed plans to the model. Neither survival nor forecast error identifies a
causal learning defect. No loss coefficient or horizon search follows directly.

Before execution, validate joins, terminal/life separation, horizon/discount
alignment, ties, malformed forecasts and the actual capture entrypoint with
CPU fixtures. Pin sources, dependencies, saved branches, checkpoint and protocol.
Run one guarded frozen capture (180-second cap) on an idle host, with a600-second
controller cap. The unchanged qualified native needs no rebuild or repeated
qualification. Stop on a genuine host fault, nonfinite result or failed audit;
preserve failures, with no automatic restart. Perform separate CPU analysis and
terminal verification. Root `/x/Temp/kindle-breakout-rescue-forecasts-20261104`.
