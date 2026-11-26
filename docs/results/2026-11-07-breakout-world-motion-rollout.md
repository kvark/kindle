# Motion-weighted forecasts improve short tracking but fail long-horizon retention

The [motion-weighted rollout experiment](../experiments/2026-11-07-breakout-world-motion-rollout.md)
is **closed and fails retention**. Recorded development eight-step ball accuracy
improves 14→25/107, sixteen-step 3→10/114 and thirty-two-step 0→4/95. Perception,
one-step forecasts and bricks regress. On the retained fixed-action oracle,
primary long-horizon ball-feature error worsens **84.05%**, with new false endings.
Keep the coverage control; no extension or replacement is selected.

Direct future-image training improves some short tracking measurements while
long-horizon distributions, object presence and event calibration remain weak.
Whether feedback-control information, coverage, uncertainty, decoder averaging
or other optimization effects cause the remaining errors is unresolved. The full
world-model goal remains active and unproven. Actor/critic stays parked.

The later [matched-metric action audit](2026-11-07-breakout-world-action-gap.md)
clarifies that some fixed-action point tracking also improves at8/16. The failed
primary energy mixes different horizons,missing detections and event weights;
it did not establish that all short-term gains failed to transfer. Thirty-two-step
tracking and unwanted action sensitivity remain clear failures. Original scores,
gates and failed retention status are unchanged.

## Fixed intervention and identity

Keep observed motion-weighted RGB cross-entropy, one-step KL 1/.1 and observed
reward/continuation 1. Enable `motion_cross_entropy` with prediction length 16.
After observed index 15, run 16 causal prior transitions under submitted actions.
Use the existing decoder and categorical samples to predict each future RGB.
Weight cross-entropy by `1+100*abs(real_current-real_previous)`, normalized per
image to mean 1. The first suffix previous image is the final observed image.
Sum event pixels, average full B×16 including zero-weight rows, and apply the
existing reconstruction scale 100. Reset arrival and later suffix losses are
zero; genuine terminal arrivals remain valid.

Future images and motion weights are detached loss targets only. No future
image is encoded into a prior transition. Unlike the closed recurrent-energy
experiment, there is no moving latent target or sample-repulsion term. Gradients
can reach the observed seed, encoder/posterior, dynamics and shared decoder.
Target-dependent attention is not a calibrated generative likelihood; intermediate
categorical ST gradients remain biased. Pointwise image pressure can blur uncertain
futures. No object/RAM/phase labels, new parameter, decoder or behavior head.

This tests the missing combination identified by code review: earlier direct
future-image losses were uniform MSE/cross-entropy, while successful observed
motion weighting explicitly rejected H>0. The separate
[CPU drift decomposition](2026-11-07-breakout-world-drift-components.md) favors
investigating shared learned drift before sampling suppression; it does not
prove this objective would help or describe this new checkpoint.

Root `/mnt/data/kindle-breakout-world-motion-rollout-20261107`. Implementation
`85dbe150aed1e87a42a3089ee612fc6ef2d88f86`, native SHA 256
`5b1128667e6436b9865c31dedcb50e22318a43250e884046b155e51dc955ad82`,
matching root/runtime and pinned runner. Defaults and H0 inference stay unchanged.

Fresh 8,192 updates retain 8,852,675 parameters, RGB 64/12M/B8/T32/full BPTT/Auto,
seed 41001, rates 4e-4 and exact coverage corpora/4-2-1-1 source locations. Reused
development initialization/data are not fresh confirmation. Replay 2,097,152
arrivals, with 975,941 valid suffix rows and 9,570 suffix terminal targets. Other
exposures match:10,909 observed terminals,34,440 positive rewards,10,488 natural
resets,75,808 context resets and 32,849 sampled unique arrivals of32,850 available.
No new environment action or emulator frame; no collector or old worker repeats.

Native calls 765.720s versus 529.515s(+44.61%); complete training/evaluation/export
loop 1,105.997s versus 796.900s(+38.79%). Timers exclude construction, include host
interaction and are not isolated GPU-kernel or broad speed measurements.

## Fitting versus recorded development

All source images, masks, labels and locations match the control in both dense
splits. Ball counts mean a supported target within one resized pixel under the
first fixed draw, not independent episodes or stochastic calibration.

|Measure|Coverage fitting|New fitting|Coverage development|New development|
|---|---:|---:|---:|---:|
|Observed ball|159/163|154/163|108/112|99/112|
|One-step ball|153/177|130/177|94/109|84/109|
|Four-step ball|73/180|86/180|33/109|47/109|
|Eight-step ball|34/176|61/176|14/107|25/107|
|Sixteen-step ball|14/171|21/171|3/114|10/114|
|Thirty-two-step ball|5/179|19/179|0/95|4/95|
|Observed brick balanced error|.008525|.098104|.223954|.278206|

The declared observed>=90%, one-step>=89/109 and eight-step>=28/107 gates all
fail. Fitting improvements show this is not exclusively a development-generalization
problem, but fitting forecasts are still far from accurate. No favorable draw or
checkpoint replaces the final 8,192 endpoint. Retain 0/2048/4096/8192 checkpoints,
all periodic four-draw curves,256 fitting/128 development dense exports and all
horizons. `development-curves.png` includes the closed reference learners.

Missing or diffuse predicted balls remain substantial. At8, detection falls
99→77/107 despite correct positions rising 14→25. At32, detection falls 77→27/95.
The preselected visual examples show diffuse balls/paddles in some new forecasts;
this is consistent with image averaging/suppression, not an isolated causal proof.

A **post-hoc CPU-only** diagnostic reuses the earlier threshold-free red-excess
peak/local 3×3 centroid without changing the scored instrument. On visible true
development balls, the new conditional counts become 27/107 at8 and 7/95 at32,
versus 25/107 and 4/95 scored. Mean conditional position errors remain 4.103/11.274
pixels. A simple brightness-threshold relaxation does not rescue these forecasts.
Argmax always supplies a coordinate even for noise; this is not presence accuracy,
a replacement gate, a trained head or proof that latent information is absent.
`faint-diagnosis.json` separately pins 13 inputs and the exact copied method;
the already-pinned main closure remains unchanged. No native query repeats.

## Frozen fixed-action assessment

Only the new final checkpoint runs:84 groups, history 16 primary/32 secondary,
eight fixed draws,284.869s. Reuse the original sticky .25/repeat 4 oracle,33 distinct
contexts across seven categories, without recollection or old-model assessment.
All oracle arrays match. Repetitions/draws are not independent; the runner's
historical fresh-development string is not an untouched-test claim.

|History 16 measure|Coverage|Motion rollout|
|---|---:|---:|
|Primary 16/32 ball-feature energy|.169502219|.311976907|
|NOOP 8 false-ball draws|0/64|0/64|
|RIGHT 16 ending probability, oracle 0%|11.049%|8.200%|
|Ending 16 Brier, all four plans|.003345|.002422|
|RIGHT 32 ending probability, oracle 100%|50.211%|50.754%|
|Ending 32 Brier, all four plans|.144566|.189225|
|Non-ending origin/plans above 5% ending probability at32|0/304|23/304|
|Maximum non-ending origin/plan probability at32|1.907%|43.840%|

Primary error rises 84.055%, failing the 25% reduction cutoff. History 32 secondary
energy is .320062338. NOOP and both ending 16 timing gates pass; ending 32<=.10 and
non-ending group means<=5% fail. Largest non-ending group mean is8.262%. The finite
oracle entropy reference remains .004812590, not a certified floor or a calibration
bound. Early-timing improvement must not hide poor later termination detection.

There are local endpoint gains: free-flight 32 ball energy .30596→.15218, brick 32
.13811→.02392 and wall 32 .29655→.04481. Intermediate dynamics regress: free-flight 8
.02059→.14555, contact 8 .09393→.34585 and life-loss 8 .00453→.66527. Ending 32 ball
energy worsens .15896→.69862. At8, paddle energy jumps from approximately .008–.014
to .233–.291 in the non-serving categories. Some later paddle endpoints recover;
this is not a uniformly better physical model.

Reward and brick regressions persist. Ending 32 reward-event Brier .000173→.428074,
and alive-only brick Brier .003655→.009259. Other reward slices improve, including
free-flight 32 .26657→.00342. `branch-comparison.json` retains every category,
horizon, action plan, history, reward, paddle and brick result, alongside latent,
recurrent and ST controls. The invalid FIRE 90% pair-match gate is not reused.
`branch-forecasts.png` shows the preselected six cases at8/32, draw 0 and survival
weights, without best-draw matching or a competence claim from visual similarity.

## Validation and remaining work

62 CPU world tests(27 GPU tests ignored),55 selected/55 packaged Python checks,
strict workspace/all-target and binding Clippy, and release packaging pass. New
composed temporal motion loss F64 values/derivatives, stopped targets/masks,
first-suffix alignment, causal ancestry and parameter-shape checks pass. New
native value/derivative fixture and full-shape preflight pass, with 204 exact
initial tensors,18 initial arrays and legacy H0 restore. Expected 65 parameters
change at the first update; all 68 change during the full learner.

All 8,192 finite 11-report rows, objective/validity identities, exact source
plan/exposures,204 initial/frozen tensors and 18 final restore arrays pass.
Qualification 86, learning 95 and branch 1,280 input pins revalidate at closure.
Five host guards/45 sealed pins pass; 204 final tensors equal both branch snapshots.
No GPU fault, recovery, worker retry or active Kindle job remains. Execution
checks do not establish learning quality. `closure-review.json` records failures
of the declared quality gates separately from successful execution.

Preserve `python-cpu-first.log`:54 checks pass and one mocked probe fixture errors
because the selected source package lacks its native Agent export. Correct package
selection passes 55/55, and the new packaged wheel separately passes 55/55. No GPU
or model stage repeats for that correction. The initial CPU run passed; unrelated
formatter changes were removed and final source rerun passes 62 tests with 27 GPU tests ignored. Release the
1,275,600,936-byte disposable binding cache after packaging, retaining wheel,
runtime and separate cache receipt without changing the pinned build receipt.

Next investigate the recorded-control versus fixed-action gap and how conditional
forecasts preserve small objects, before selecting another loss or extension.
Neither action-feedback confounding nor a decoder-only cause is established here.
No new learner/collector is selected. Precise all-horizon trajectories, calibrated
reward/events, brick-removal memory, late-game/tunnel/second-wall coverage,
untouched families and fresh confirmation remain required. CDP efficiency and
actor/critic integration stay parked.
