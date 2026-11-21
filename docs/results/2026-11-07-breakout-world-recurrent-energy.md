# Recurrent-state supervision improves some endpoints, but fails tracking and timing

The bounded recurrent-state energy experiment is **closed and fails retention**.
Primary long-horizon ball-feature energy improves 22.11%, short of the declared 25%
gate. Observed ball reconstruction stays 108/112, but one-step tracking falls 94→88/109
and eight-step 14→13/107. Game-over predictions arrive too early. Keep the
[coverage trajectory control](2026-11-07-breakout-world-coverage.md); do not
replace it, extend this recipe or claim a proved Breakout world model.

## Intervention and cost

The [protocol](../experiments/2026-11-07-breakout-world-recurrent-energy.md)
tests the hypothesis suggested by the
[state substitutions](2026-11-07-breakout-world-state-swap.md). An image-informed
future recurrent state repairs ball readouts; learning to forecast that state
does not automatically follow from the diagnostic.

Select optional `motion_recurrent_energy`, H16, default off. Keep observed motion
RGB cross-entropy, one-step KL1/.1 and observed reward/continuation 1. From the same
detached posterior h/z at index 15, run two causal prior futures under real actions,
with independent intermediate categorical samples. At every suffix step, score
against the stopped image-informed recurrent state y:

`d(a,b)=sqrt(mean_components((a-b)^2)+1e-6)`

`E=.5(d(h1,y)+d(h2,y))-.5*d(h1,h2)`

Average over the full B×16, including zero-weight rows, and add at reconstruction
scale 100. Mask from the first reset arrival onward; keep genuine terminal
arrivals. No future image/state enters a prior transition. Added gradients reach
dynamics only; intermediate straight-through gradients remain biased. The target
representation is jointly learned, not a fixed physical observable. This is not a
globally unbiased estimator or a demonstration of calibrated observable futures.

No new decoder, trainable parameter, inference input, actor or critic. Fresh 8,192
updates reuse 8,852,675 parameters, RGB 64/12M/B8/T32/full BPTT/Auto, seed 41001,
rates 4e-4 and the exact coverage 4/2/1/1 sampling plan. This is a reused development
initialization/corpus, not independent confirmation. Replay 2,097,152 arrivals,
975,941 valid suffix rows,9,570 suffix terminal targets,10,909 observed terminals,
34,440 positive rewards,10,488 natural resets and 75,808 context resets. Sample
32,849 unique arrivals of32,850 available. No new environment action/frame.

Native learner calls 604.731s versus 529.515s(+14.20%). Training/evaluation/export
loop 837.601s versus 796.900s(+5.11%). These elapsed costs exclude construction
and do not isolate GPU kernels. No backend or precision optimization is tested.

Root `/mnt/data/kindle-breakout-world-recurrent-energy-20261107`, source
`ab68b0f74f9b421296eaf77332210e92bc14a70d`, native SHA 256
`2d02f98163d3aa628eb9e8be411359ea08d1e915c60d2200d2a9d964e2c3681e`.
Use its matching root/runtime and runner; no overlay mixing.

## Matched dense development

All source/label/location/mask arrays match the retained control. Supported ball
targets within one resized pixel, one fixed draw; these are not independent
episodes or population calibration evidence.

|Measure|Coverage|Recurrent energy|
|---|---:|---:|
|Observed ball|108/112|108/112|
|One-step ball|94/109|88/109|
|Four-step ball|33/109|33/109|
|Eight-step ball|14/107|13/107|
|Sixteen-step ball|3/114|6/114|
|Thirty-two-step ball|0/95|0/95|
|Observed brick balanced error|.223954|.216280|

Observed>=90% passes; one-step>=89/109 and eight-step>=28/107 fail. Sixteen-step
counts improve but remain poor. Observed bricks do not establish removal memory;
the observed dense slice contains no newly removed-brick targets. All 256 fitting/
128 development final exports, intermediate four-draw curves and 0/2048/4096/8192
checkpoints remain retained. The fixed endpoint is not replaced by a better draw.

Across 128 observed origins, recurrent h RMS norm is .37885 versus .37484 control,
and across-origin RMS standard deviation .36575 versus .36157. Supported 32-step
prior h dispersion is .36114 versus .35852. These do not show an obvious constant
state, but neither prove semantic preservation nor measure stochastic forecast
spread: only the first retained feature draw is available. Cross-model hidden
units are not an invariant quality metric. `recurrent-state-spread.json` retains
all horizons, participation ratios, source pins and limits. The first/last 256-update
mean energy is .02851/.07594; targets/states change during learning, so this is not
a fixed-target convergence measurement.

## Frozen action branches

Only the new checkpoint is assessed: history 16 primary/32 secondary, eight fixed
draws,84 groups,241.801s. Reuse the original sticky .25/repeat 4 oracle,33 distinct
contexts across seven categories, without recollection or old-model reruns.
All oracle arrays match exactly. Repeated origins/draws are not independent;
the runner's historical fresh-development string is not an untouched-test claim.

|History 16 measure|Coverage|Recurrent energy|
|---|---:|---:|
|Primary 16/32 ball-feature energy|.169502219|.132026495|
|NOOP 8 false-ball draws|0/64|0/64|
|RIGHT 16 ending probability, oracle 0%|11.049%|55.427%|
|Ending 16 Brier, all four plans|.003345|.229969|
|RIGHT 32 ending probability, oracle 100%|50.211%|63.914%|
|Ending 32 Brier, all four plans|.144566|.052126|
|Non-ending origin/plans above 5% ending probability at32|0/304|0/304|
|Maximum non-ending origin/plan probability at32|1.907%|4.742%|

Primary energy improves 22.109%, failing the 25% cutoff. History 32 secondary energy
is .142286092. The empirical oracle entropy reference remains .004812590: a finite
descriptive reference, not a certified floor or a reason to dismiss model error.
NOOP waiting, ending 32 Brier<=.10 and non-ending group means<=5% pass. Largest
non-ending group mean is .979%; all 304 individual origin/plans also stay below 5%.
Both new timing checks fail: ending 16 Brier<=.01 and RIGHT 16 ending<=10%.

Some long endpoints improve while intermediate motion regresses. Free-flight
ball energy at8 worsens .02059→.05348 but at32 improves .30596→.05423; contact 8
worsens .09393→.24175. Wall 32 improves .29655→.08493. Ending 8 worsens .07104→.50347
and ending 16.00921→.24050, despite ending 32 improving .15896→.06829. Endpoint
success cannot hide event timing. Ending 32 reward-event Brier worsens .000173→
.419860 and alive-only brick Brier .003655→.008816. Other reward/brick groups
improve; the full comparison retains every category, horizon, history and plan.

`development-curves.png` compares all retained learning curves.
`branch-forecasts.png` shows the six preselected cases at8/32, oracle/model draw 0
and survival weights. These examples are not optimal-draw matching or acceptance
criteria. `branch-comparison.json` retains all metrics, including the closed
latent/ST controls; the invalid FIRE 90% pair-match gate is not reused.

## Execution evidence and remaining scope

60 CPU world tests(26 GPU tests ignored),54 selected/54 packaged Python checks,
strict workspace/all-target and binding Clippy pass. New checks cover F64 energy
values/derivatives, finite differences, stopped targets/masks, causal ancestry and
dynamics-only auxiliary gradients. Native values/derivatives and full-shape
preflight pass, including legacy coverage/latent restores. All 204 initial tensors
and 18 initial arrays match; the first update changes the expected 65 parameters.

All 8,192 finite 13-report rows, energy/total/validity identities, exact plan/counts,
204 initial/frozen tensors and 18 final restore arrays pass. All 68 logical
parameters change by the endpoint. Qualification 90, learning 99 and branch 1,068
input pins revalidate at closure. Five native host guards and 45 sealed-file pins
pass; 204 final tensors match both frozen branch snapshots. No fault, recovery,
active Kindle worker, unchanged extension or successor remains selected.

Preserve the initial CPU declaration failure: a copied helper expected
`cpu-final.log` instead of the successful `cpu-initial.log`. Its subsequently
launched controller cannot open the not-yet-packaged queue script and exits 2,
before native construction. `declaration-first.json`, original helper/log and
`controller-first-service.json` retain that failure. Before any native worker,
review also corrects a copied old protocol-prose path; original declaration,
prose and helper remain separately hashed. These are not GPU failures or passed
original attempts. Corrected qualification runs only previously unstarted work.
`closure-review.json` explicitly distinguishes the failed initial controller.

Release 808,782,247 bytes of disposable binding build cache after packaging;
retain wheel/runtime and a separate `cache-release.json`, without changing the
originally pinned build receipt. No learner or assessment repeats for repairs.

The full goal remains active and unproven. Good observed readouts plus direct
latent-state supervision have not produced reliable causal trajectories or event
timing. This tested objective does not solve the recurrent drift identified by
the substitutions; it does not rule out every form of recurrent supervision.
Precise all-horizon forecasts, reward/event calibration, removed-brick memory,
late-game/tunnel/second-wall coverage, untouched families and fresh confirmation
remain required. CDP efficiency and actor/critic integration stay parked.
