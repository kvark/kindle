# Conditional score-function gradients for the predictive world

The [closed energy comparison](../results/2026-11-07-breakout-world-energy.md)
improves ending but regresses precise ball tracking on fitting and development
windows at86.58% more native learning cost. Its CPU categorical counterexample
proves that coordinate-ST derivatives can miss required variance correction.
This does not attribute the full learner's regression to gradient bias.

Select a bounded comparison of the **conditional categorical gradient estimator**,
not another objective or capacity sweep. The goal remains the complete Breakout
world, including late-game/second-wall coverage, calibrated stochastic futures,
physical action effects and fresh confirmation. This experiment cannot close it
by itself. Actor/critic remain parked. No collector or old-model rerun.
Root `/mnt/data/kindle-breakout-world-score-function-20261107`.

## Exact conditional objective and gradient

Keep old energy-mode sampled worlds, target encodings, past-only RGB metric,
masking, scales, RNG streams, architecture and13 report meanings unchanged.
Add optional `predictive_motion_energy_score_function`; old energy and legacy
objectives remain supported. No parameter,geometry,inference or external label
change. Prior/heads use the same supplied draws and sampling distributions.

Given the teacher-forced past h, each world sample consists of categorical prior
Z, reward bin R and continuation C. Compute
`logp_i=log p(Z_i|h)+log p(R_i|h,Z_i)+log p(C_i|h,Z_i)`.
Use the existing .01-unimixed prior probabilities and current head softmax/
Bernoulli probabilities. Detach sampled categories; differentiate RGB decoder
values and deterministic state paths normally.

For E=.5*(d(X1,Y)+d(X2,Y))-.5*d(X1,X2), add a zero-valued gradient term

```
u = stop(.5*(d(X1,Y)-d(X1,X2)))*logp_1
  + stop(.5*(d(X2,Y)-d(X1,X2)))*logp_2
loss = E + u-stop(u)
```

The other sample's target-distance term is independent of each sample's draw,
so it has zero expected score-function contribution and is omitted as a
baseline. Coefficients/masks/targets/metric weights are detached. This is our
conditional derivation, verified by finite enumeration; the general likelihood-
ratio method follows [Williams1992](https://doi.org/10.1007/BF00992696).

The estimator targets the actual categorical expectation conditional on h,
including latent/heads correlations. It can have greater variance than ST.
Past posterior samples and the unchanged observed ELBO retain their existing ST
path; **this does not make full BPTT globally unbiased**, establish calibration,
or guarantee better learning. Do not double-count coordinate-ST and score-function
gradients through the same current sampled category.

## Qualification before learning

CPU checks must verify exact finite-enumeration gradients for categorical prior,
conditional heads and differentiable RGB values, with .01 prior mixing; the
previous symmetric variance counterexample; independent draws; zero/reset masks;
target/weight detachment; causal prior state; unchanged ordered parameter shapes;
old/new sampled forward scores; graph compilation and objective compatibility.
Use independent F64 references and finite differences. Preserve failures.

After CPU/package checks and exact source/native/runner/data pins, only one new
native derivative fixture (180s) and one full-size preflight (900s) are selected.
Guarded AMD Auto, no driver operations. No repeated old primitive or learner.
Compare204 initial tensors/18 initial arrays with the retained coverage control,
restore legacy motion and old energy metadata, and match all13 first-update
forward reports to the old energy preflight within1e-5 relative tolerance.
Check65 expected first-update changes (three reward hidden tensors initially
receive zero gradient),204 frozen tensors and18 exact age1 restore arrays.
No full learner selected until this review. No automatic successor.

## Separately declared learning after review

If qualification passes, separately declare fresh matched-init8,192 updates on
the exact coverage data/4/2/1/1 sampling plan. RGB64/12M/8,852,675 parameters,
B8/T32/full BPTT/Auto,seed41001,rates4e-4,reconstruction100,KL1/.1,free-nat1,
mixing.01,observed reward/continuation1. No policy/value/imitation/intrinsic/replay-
value losses, no warm extension or new collection. This is reused development.

Pin32,850 available/32,849 sampled arrivals,10,909 terminal,34,440 positive,
10,488 natural-reset,75,808 context-reset and2,021,344 valid predictive rows.
Retain0/2048/4096/8192,unchanged selections/four-draw curves,256 fitting/128
sampled-development exports,204 frozen checks and18-array final restore.
Caps3,600s learning/600s restore. Record native/loop cost,not just finite losses.

After learner review, separately declare only its final checkpoint's frozen
oracle evaluation,history16 primary/32 secondary,eight draws,84 groups,1,800s.
No oracle recollection or old-model rerun. Retain all categories/horizons/reward/
ending/brick scores and raw RGB panels; no favorable-draw or checkpoint selection.

Use the same development gates: observed ball>=90%,NOOP8 false detections<=10%,
ending32 Brier<=.10,every non-ending group's mean ending32 probability<=5%,
and equal-category ball-feature energy over16/32 at least25% below the coverage
control (.1271266643173294). Report individual false-ending probabilities and
precise1/4/8/16/32 errors as well; group means can hide bad cases. Compare against
both retained capabilities without re-evaluating either old model. FIRE90% pair
matching remains invalid. A passing aggregate is insufficient for the full goal.
The oracle's33 distinct contexts, limited detector support and absent above-top-
brick/late-game/second-wall cases remain explicit gaps.


## CPU preparation complete

48 CPU world tests pass(22 GPU tests ignored),51 Python checks pass. Exact test
selection: `test_breakout_world_corpus.py`, `test_breakout_world_coverage.py`,
`test_breakout_world.py`, `test_breakout_world_branches.py`, `test_world_probe.py`.
Strict workspace/all-targets and bindings Clippy pass. No GPU ran during repairs.
Preserve the first fixture's unnecessary scalar/1x1 unused-term shape failure
and the Clippy Vec/array failure; separate repairs pass without tolerance changes.

The previous energy run's published41/41 Python counts were wrong: logs show
27/27. Keep the original receipt/protocol/publication and separate CPU count audit;
this run's51 checks do not retroactively complete that claim. See the prior
[result correction](../results/2026-11-07-breakout-world-energy.md#python-verification-count-correction).
Next package/pin/qualify only; no full learner selected.


## Qualification reviewed; fixed learner declared

Native/source0ab7fea7,SHA0ebed9cd,matching root/runtime.48 CPU world and51 affected/
51 packaged Python checks,strict workspace/bindings Clippy pass. Both new guarded
jobs pass,sealed audits pass,83 input pins verify. Native exact-enumeration
conditional derivatives match the independent F64 reference. Full-size first
update matches all13 old energy forward reports exactly;204 initial/frozen
tensors,18 initial/restore arrays,65 expected changed parameters and legacy
motion/energy restores pass. No GPU/numerical fault or host recovery.

`qualification-review.json` reviewed. `learning-protocol.json`/queue separately
declare only the fresh fixed8,192-update learner and restore,95 input pins,
all previously specified settings/data/counts unchanged. Training is active;
frozen oracle evaluation is not selected yet. No collector or old-model rerun.


## Closed negative comparison

The [complete result](../results/2026-11-07-breakout-world-score-function.md)
records all stages closed. Native990.401s,loop1,240.170s; observed109/112 stays
accurate but one-step71/109 regresses81 ST/94 coverage. Primary ball energy.393950,
NOOP8 false balls45.31%,ending RIGHT8.591% and ending Brier.433024 fail important
world gates. No extension,confirmation campaign or replacement is selected.
All83/95/858 phase pins,five guards/sealed audits,204 frozen tensors and18 restore
arrays pass. Mathematical/native verification is distinct from useful learning.
The separate [frozen-readout diagnosis](../results/2026-11-07-breakout-world-readout-diagnosis.md)
finds accessible short-term motion but no simple linear forecast-head rescue.
No new GPU job/learner is active. Full world goal remains open; actor/critic parked.
