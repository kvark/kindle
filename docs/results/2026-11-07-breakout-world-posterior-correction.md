# Correcting the posterior at forecast h gives partial repair, not a reliable target

The [declared readout diagnostic](../experiments/2026-11-07-breakout-world-posterior-correction.md)
is closed. Recomputing q(z | h_prior, true future image) repairs more ball
readouts than swapping a posterior sample tied to teacher-history h, but still
fails the predeclared natural8>=80% / natural32>=70% target-availability triage.
Do not launch the proposed future-latent loss on this justification. This does
not disprove all posterior targets or identify a causal fraction of model error.
The world-model goal remains unproven; no learner runs in this diagnostic.

## Fixed comparison

Use only the retained coverage H0 checkpoint (8,192 updates), all cached four
natural batches (32 origins), two ending-at17 batches (10 real origins) and four
original draws. Padding, reset crossings and unavailable tails remain unscored;
terminal arrivals remain valid. These are reused development data, not independent
episodes or untouched tests. Read cached h/z/logits; never execute prior or teacher
transitions. Native Meganeura/Blade encodes the true current future image, computes
its posterior at cached h_prior, and decodes that h with a fresh posterior sample.
Use identical per-step random numbers; no correction feeds another transition.

All corrected readouts are privileged future-image diagnostics, not causal
forecasts. Cached ordinary prior, teacher-z swap and full teacher are controls.
First-step logits/readouts agree within1e-5+1e-5*abs(reference); categorical samples
match exactly, as all methods share the same h at that step.

|Decisions ahead|Cached prior|Cached teacher-z swap|New posterior at prior h|Cached full teacher|
|---|---:|---:|---:|---:|
|1|91/108|104/108|104/108|104/108|
|4|31/108|34/108|71/108|105/108|
|8|15/108|24/108|45/108|104/108|
|16|6/112|9/112|15/112|102/112|
|32|1/108|0/108|4/108|101/108|

Counts pool four draws per supported natural origin, with the original within
one resized pixel instrument and missed detections retained in the denominator.
At8, the correction rescues 30 prior misses and harms 0 successes; at 32 it rescues 3
and harms 0. It renders a supported ball in 95/108 and 68/108 images at 8/32; merely
supplying the correct future image does not make this frozen visual path recover
accurate long-rollout readouts at the drifted recurrent state. The old teacher-z
swap understated the available short correction, but its poor long result was
not solely that mismatched conditioning.

Ending-at17 origins separately improve1→12/36 at 8, versus 24/36 full teacher.
There is no visible-ball support at 16 in that slice, or any supported 32-step
teacher target. No reward/continuation head is queried, so this diagnostic makes
no new event-calibration claim. Full paddle/brick metrics, paired rescues/harms
and per-horizon categorical KL remain in `analysis-review.json`; latent KL is an
internal checkpoint-specific statistic, not a physical quality score.

## Implication

The current visual readout relies on a recurrent state whose history has drifted.
Partial repair with a newly conditioned posterior is useful evidence, but does
not establish a strong target for another frozen-posterior alignment loss. A
concrete architectural question is whether an observation-only visual code and
readout can preserve current image information independently of h, while h
models temporal dynamics. That change still requires native implementation,
bounded learning and the unchanged full-world gates; no benefit is established.
Training on deliberately drifted states is another possibility, not a proved fix.

Keep the coverage control. Do not repeat this completed diagnostic, extend a
closed learner or run an automatic successor. Precise causal1/4/8/16/32 futures,
reward/event calibration, brick memory, late/tunnel/second-wall coverage,
untouched families and fresh confirmation remain required. Actor/critic and CDP
comparisons remain parked.

## Execution and identity

Test-only source `32b182a165b3246c8c3d985f012db172e1773f28`, executable SHA256
`40e79a338cc81134cc4f2cee430437752bb7ebc72bc4e8af32932a84bba625a0`.
No production graph, API or installed runtime changes. Two focused CPU checks
and strict Kindle lib/tests Clippy pass. Data preparation verifies 43 pins, and
the native declaration verifies 55. Preserve the initial preparation copy failure:
it requested nonexistent `gpu_driver_guard.py`, before a declaration/queue or
GPU worker existed. The separately recorded repair copies the actual
`gpu_guard.py` dependency and executes only unstarted work.

All 24 traces/768 correction steps complete first native attempt in 22.751s.
All 768 regenerated cached prior samples match exactly; all 24 first-step teacher
controls pass. All 36 visual parameters match source before/after, with no
parameter gradients or optimizer configured and optimizer ages remaining 0.
The original 204-tensor checkpoint bytes remain unchanged. AMD/Auto host guard
and nine sealed-file audit pass, with no recovery/reset/NVML/unfinished child.
CPU scoring takes 2.267s; no native/emulator query is repeated by scoring or display.

Root `/mnt/data/kindle-breakout-world-posterior-correction-20261107` retains exact
source/test executable,config,inputs,cached-state identities,new logits/samples/
readouts,protocols,guard and CPU evidence. `posterior-correction.png` displays the
fixed first natural origin/draw, all five endpoints, and explicitly labels future
image access. It is not a best-match selection or an acceptance metric.

The separately selected [observation-only code experiment](../experiments/2026-11-07-breakout-world-observation-code.md)
remains to implement and qualify. Its training allocation is not declared here.
The disposable2.14GB build cache is released after retaining the exact test
executable and source archive; no model data is removed.
