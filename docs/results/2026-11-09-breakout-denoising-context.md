# Past-motion importance does not repair denoiser fitting

The selected2,048-update check fails. Every draw still gets **0/110** ordinary
16-step ball matches, **0/198** one-step matches from actual history and **0/8**
observed noisy-frame detections. The matched uniform fitting control also scores
zero. All supported new forecast images havezero candidate ball pixels, not an
ambiguous excess of detections. No broad learner or extension is justified.
The full world-model goal stays active and unproven.

[Protocol](../experiments/2026-11-09-breakout-denoising-context.md).
Root `/mnt/data/kindle-breakout-denoising-context-20261109`. Source/runner `879ab527`, native SHA256
`9657af725307ac6ecf5f85456b5cfcbc660940c337cf5133d4107704e83a80de`.
Actor/critic,CDP comparison and the game campaign remain parked.

## Change and comparison

Only image-loss importance changes. Maximum absolute RGB change between the
two latest conditioning images is dilated by a radius4 square, transformed as
1+100*change and normalized to mean1. The same weights apply to each RGB channel;
reset history uses uniform weights. Current targets/noisy targets, object labels,
RAM, colors and positions do not choose training weights. Inference is unchanged.

For positive weights known given conditioning C, the unconstrained residual-MSE
minimizer remains E[T|X,C]. This avoids target-dependent weighting's change to
the pointwise conditional mean. Finite-capacity tradeoffs and optimization still
change; no density/calibration or useful-learning guarantee follows.

Optional `context_motion_scale` defaults to0 and is omitted from v1 serialization.
Positive importance requires v2; disguised v1/positive andv2/zero headers are
rejected before GPU. Existing world models/defaults remain unchanged. Network,
45 logical/1,787,843 scalar weights/135 tensors, four-frame/action conditioning,
three Euler evaluations, rates, precision and event losses stay the same.

Use the exact closed uniform plan/data/counts without a recount:513 calls,
2,048 fresh updates,65,536 exposures over224 targets from eight32-arrival clips.
Each target receives292–296 exposures; B32 contains four noisy copies of eight
trajectories, not32 games. Development seed41001, RGB64, Auto, LR1e-4/LaProp
epsilon1e-8/AGC.3 remain. Two prefix rewards contribute588 positive training
targets; the evaluated suffix hasno reward/ending positives. No new emulator
action/frame, collector or old-model query is used.

| Final fitting task | Uniform control, each draw | Past-motion importance, each draw |
|---|---:|---:|
| Ordinary generated-history1..16 |0/110|0/110|
| Actual-history one-step, source arrivals5..31 |0/198|0/198|
| Observed denoising at sigma.5 |0/8|0/8|

Missing balls fail. Unsupported18 positions per draw remain untested on each
future task. Target4 stays excluded from privileged evaluation by the API's
five-arrival observed-denoising requirement. Actual-history one-step is a
separate diagnostic, not a27-step rollout. All1,376 endpoints and every physical/
event output remain retained. Source arrival indices and forecast horizons are
explicit in both the review and CSV. Five checkpoint curves and16 exact fixed
lane0/draw0 clip frames remain; no best-draw selection or changed detector.

An independent F64 reference of importance on first-transition real histories
finds46 supported targets, with45 rounded ball centers receiving weight>1.
Minimum/median/maximum are1.0/11.768154/22.393171. These are diagnostic-only
coordinates, never training labels. This checks that the selected formula
usually covers the future ball in those contexts; it does not measure actual
ball-specific parameter gradients or isolate why fitting fails. Later training
histories are model-generated and are not included in this reference.

The native trainer still uses some generated conditioning images for three of
four transitions in ordinary eight-arrival calls, beginning before object
reconstruction succeeds. This remains an unisolated difference from fully
observed one-step training; this experiment does not establish its causal role.
The existing five-arrival API can test a single transition from actual history
without changing architecture. No successor allocation is selected here.

## Cost, checks and retained failures

Training-call/loop wall time is137.972865s/146.389261s,
versus uniform call133.853825s (3.077% higher). These are
wrapper measurements, not isolated GPU timings or proof of cost equivalence.
CPU outcome review takes10.184922s. No useful learning-speed gain occurs.

Ten focused CPU tests,351 installed-package checks,three runner checks andstrict
core Clippy pass. Weighted native loss values/derivatives match independent F64
under a separate sealed guard before the learner starts. The execution wrapper
requires this review before constructing a GPU learner. Reuse qualified CNN/
normalization/upsampling primitives; no full-CNN F64 parity claim.

All154 declared/70 CPU inputs pass. All135 initial tensors and20 per-draw
initial forecasts match the cached uniform actual-initial control exactly.
Only that required initial RGB cache is decoded; no old final forecast RGB or
old model is queried. First update changes all45 weight tensors, stays finite,
and retains the control's first reward/ending losses within1e-6 before any
second update. All513 six-component rows, totals/configuration/ages/exposures
agree;135 frozen tensors and20 per-draw restored arrays match exactly.

Original mean-test failure remains: accumulating normalized weights in F32
exceeds its strict tolerance. The separate correction accumulates in F64 with
the same tolerance and changes no production weight. The first CPU pixel-review
serialization fails on a NumPy int64 and leaves its partial file/source/log;
a separate CPU-only repair writes a separate result, without another model query.
Neither original failure is relabeled passed. Root space pressure is relieved
first through archived/released disposable workspace caches and then the user's
[temporary-storage cleanup](2026-11-09-temp-cleanup.md); source/runtime/checkpoints
remain. Qualified source, runtime, wheel, executable, all weights and failures
are retained outside Git. Disposable packaged build caches are released separately.

No real GPU fault, reset/reload or NVML polling occurs. Native qualification,
training and restore have sealed host guards; both services are terminal.
A failed fitting check does not rank all denoisers or budgets. Full physics/
events/calibration, persistent removed-brick memory, late/tunnel/second-wall/
above-top coverage, untouched families and fresh confirmation remain required.
Neither fitting nor a privileged score completes the
[full world-model goal](../kindle_single_life_dreamer_plan.md).

The first closure check runs before cache cleanup finishes and fails on the
not-yet-written release receipt. Its source/log remain; a separate closure
completes after cleanup, without repeating any learner or model query.
