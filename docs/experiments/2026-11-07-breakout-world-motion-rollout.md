# Breakout: motion-weighted direct future-image supervision

The [drift decomposition](../results/2026-11-07-breakout-world-drift-components.md)
favors shared transition/representation investigation before suppressing samples.
Observed motion weighting already recovers108/112 balls,while direct future-image
training previously used uniform MSE/cross-entropy and suppressed the ball. Select
the missing combination: observed plus causal future motion-weighted RGB
cross-entropy. This is a new bounded recipe,not an extension of a closed learner.

Root `/mnt/data/kindle-breakout-world-motion-rollout-20261107`. No actor/critic,
collector,backend optimization,CDP efficiency comparison or capacity increase.

## Fixed objective

Allow explicit `motion_cross_entropy`,prediction_length16. Keep the exact H0
observed reconstruction,one-step KL1/.1,reward/continuation1 and all68 logical
parameters. After observed index15,roll16 prior transitions with real submitted
actions and independent existing categorical RNG. At each arrival,apply the same
RGB cross-entropy as observed states,weighted by
`1+100*abs(real_current-real_previous)`,normalized per image to mean1. The first
suffix previous target is the last observed image. No object/RAM/phase label.

Future images and motion weights enter detached loss targets only,never a state,
prior transition or future action. No future true image is encoded for the prior.
Use the existing suffix mask: reset arrival and all later rows have zero loss;
valid terminal arrivals remain. Invalid motion rows may be uniform,then masked
out. Sum event pixels and average full B×16,including zero-weight rows; multiply
by reconstruction100,as for observed RGB. No extra reward/continuation prediction
loss or moving latent target. Prior-image gradients can reach the observed seed,
encoder/posterior,dynamics and shared decoder,as in the original image rollout.

This deliberately adds future-image work and may interfere with perception or
encourage a blurred conditional mean. Target-dependent weights are not a calibrated
generative likelihood; categorical ST remains biased. No physical benefit follows
from finite losses or shared recurrent-drift statistics.

## Qualification and allocation

CPU checks: independent F64 two-step weighted cross-entropy values/derivatives,
zero/reset masks,stopped current/previous targets,first-suffix motion alignment,
and no future targets/masks in forecast-state ancestry. Preserve legacy H0/MSE/BCE
graphs/metadata and parameter shapes; verify full graph autodiff/compile and
Python11-report/scale/validity contracts. Run affected tests and strict Clippy,
then package exact source/native/runner bytes. Preserve all failures separately.

After pins and current host/process review,declare only one new composed temporal
motion loss native value/derivative fixture(180s) and full-shape preflight(900s).
Require204 exact initial tensors,18 initial arrays,finite11-report update,base
observed component equality,total/validity identities,204 frozen tensors,18 trained
restore arrays and legacy H0 restore. This is bounded integration qualification,
not a repeat of completed primitive fixtures or a backend project.

Only after those pass,separately declare one fresh8,192-update learner(3600s)
and final restore(600s). Keep8,852,675 parameters,RGB64/12M/B8/T32/full BPTT/Auto,
seed41001,rates4e-4,exact coverage corpora and4/2/1/1 source locations. Retain
0/2048/4096/8192 checkpoints,curves,256 fitting/128 development dense exports and
all1/4/8/16/32 horizons. Valid suffix rows975,941,terminal arrivals9,570; other
coverage counts unchanged. Reused development seed/data,not confirmation.

After learner/restore audits,separately declare only the new final checkpoint's
frozen oracle assessment(1800s):history16 primary/32 secondary,eight fixed draws,
84 groups. Reuse retained oracle/controls; no recollection or old-model rerun.

Keep the existing intermediate retention gates: observed ball>=90%,one-step
>=89/109,eight-step>=28/107,primary16/32 ball energy<=.12712666431732942,NOOP8
false balls<=10%,ending32 all-plan Brier<=.10,non-ending group mean ending32<=5%,
ending16 all-plan Brier<=.01 and RIGHT16 mean ending<=.10(oracle0). Report every
category,horizon,action,reward,paddle and brick regression,not just aggregates.
No invalid FIRE90% pair-match gate,unchanged extension,scalar/seed sweep or
automatic successor. Use AMD host guard/Auto;stop on real faults or nonfinite
values,no reset/reload/reboot. No new emulator action/frame.

Passing intermediate gates is not full-goal completion. Precise all-horizon
forecasts,event/reward calibration,brick-removal memory,late-game/tunnel/second-
wall coverage,untouched families and fresh confirmation remain required.
