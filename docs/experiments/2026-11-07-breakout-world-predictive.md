# Direct one-step supervision of the predicted world

The [closed coverage comparison](../results/2026-11-07-breakout-world-coverage.md)
teaches waiting and improves several contacts with unchanged architecture/losses,
but long forecasts, ending and brick memory remain inadequate. The retained
model's complete-case decomposition also finds systematic drift beyond oracle
randomness. Select an explicit **one-step predictive-world objective**, retaining
the coverage data/sampling and observed losses. This tests useful training, not
an isolated image-versus-head cause or a noise/precision optimization.

Root `/mnt/data/kindle-breakout-world-predictive-20261107`.

## Exact change

Add optional `pixel_loss="predictive_motion_cross_entropy"` to the standalone
native world learner. Keep the existing observed motion-weighted RGB loss, KL,
reward and continuation. At every arrival t, construct a separate categorical
prior sample from `p(z_t | h_t)`, where h_t uses past posterior state and incoming
action a_t, before posterior_t sees image_t. Predict current RGB, external reward
and natural continuation from this prior. Teacher forcing supplies past observations;
no current/future image, target, phase, RAM or object label enters the prior state.
Sampling uses an independent deterministic RNG stream; posterior RNG is unchanged.

Add the following losses with the same configured scales100/1/1:

- RGB soft cross-entropy with detached motion weights
  `1+100*abs(image_t-image_t-1)`, normalized per image to mean1.
- Two-hot reward cross-entropy against the same observed external reward.
- Continuation logits BCE against0 at natural terminal arrivals, otherwise gamma.

Mask all three added terms when incoming action is-1, including clipped starts
and genuine resets. Include valid terminal arrivals; a lost life is not a terminal
or penalty. Normalize by complete batch/time rows, not the number of valid rows;
report that fraction. Targets/weights/masks are detached. Observed loss remains
unchanged at starts/resets. Full BPTT lets the added targets train previous
representation/dynamics, decoder and world heads, without actor/critic dependencies.

This is one-step supervision throughout the sequence, not an open-loop suffix
loss. `prediction_length=0` retains its meaning of no extra suffix rollout.
Thirteen reports retain the original nine indices, followed by predictive RGB,
reward,continuation and valid fraction. The total includes all three new terms;
valid fraction is a metric only. No trainable parameter, input geometry, latent
width, inference path or policy changes. Additional decoding/head work costs time;
measure native and whole-loop costs rather than claiming a speedup.

Default MSE/plain BCE/observed-motion paths remain available and unchanged.
Checkpoint formatv2 already requires an explicit objective: the new enum name
rejects in older readers instead of silently changing training semantics. New
readers retain v1/v2 compatibility. Use exact new native/runner identities;
do not overlay the new objective onto an old binary.

## Bounded qualification before learning

The first GPU allocation is only one full-size preflight worker after source,
package, native, data, runner and executable bytes are pinned. Auto on the current
AMD host, existing host guard,900s cap. No backend port or repeated primitive
qualification: motion weights/CE kernels are unchanged from the qualified runtime.

CPU checks cover independent F64 image/head losses and derivatives, all-zero/
reset masks, detached targets, causal prior-state dependencies, gradient reach,
unchanged ordered parameter shapes, graph compilation and objective compatibility.
42 CPU world tests pass (20 GPU tests ignored),41 affected Python tests pass,
strict workspace/bindings Clippy pass; package checks remain before GPU.
Preserve the first literal compile failure,incorrect prior parameter-prefix
fixture and F64 fixture's unit-target-sum assumption. Repairs keep1e-10 tolerance
and account for actual F32 target values; no GPU stage has run during preparation.

The full-size preflight must:

1. Match all204 initial parameter/moment tensors and18 initial-development arrays
   to the retained coverage control. Restore its old observed-motion checkpoint
   under the new runtime and reproduce those arrays.
2. Use the already declared coverage plan's first batch, containing waiting and
   a valid natural terminal. Match its observed loss components to the retained
   first report at1e-5 relative tolerance. Execute exactly one finite13-report
   update, verify weighted total and exact valid fraction.
3. Verify predictive/dynamics/decoder/head gradients are connected in CPU audits,
   parameters change, and all204 tensors remain exact through frozen evaluation.
   Save the explicit new objective,restore it at age1,and reproduce18 arrays.

Stop on real faults/nonfinite values. No NVML polling, reset/reload/reboot,
automatic retry or repeated completed worker. Qualification does not prove world
competence. No full learner is selected until its result review.

## Fixed learning after qualification review

After review,separately declare one fresh matched-init8,192-update learner and
final restore. Keep coverage's8,852,675 parameters,RGB64/12M/B8/T32/full BPTT/Auto,
seed41001,4e-4 rates,reconstruction100,KL1/.1,free-nat1,mixing.01,world reward/
continuation1,and no actor/critic/imitation/replay-value/intrinsic objectives.
This reuses a development initialization, not a fresh confirmation seed.

Reuse the exact16-stream fitting corpora and coverage4/2/1/1 plan; no collector.
There are32,850 available arrivals and32,849 sampled,not equal counts. Preserve
the unsampled arrival. Keep10,909 terminal,34,440 positive and10,488 natural-reset
rows;2,097,152 replayed arrivals,2,021,344 valid one-step rows. Pin the complete
plan and verify it before GPU construction; do not repeat the old count mistake.
Keep native max pooling/one GPU resize and the original fitting/development split.

Retain0/2048/4096/8192 checkpoints,ordinary four-draw natural/contact/ending curves,
final256 fitting/128 development exports,18-array restore and204 frozen checks.
All input arrays/locations must match the coverage comparison. Final checkpoint
is fixed; no best-draw/early checkpoint selection. Declare learning/restore
timeouts3,600/600s after qualification. No automatic successor or unchanged extension.

Then separately declare only this new checkpoint's frozen evaluation on the
retained oracle,history16 primary/32 secondary,eight fixed draws,84 groups,1,800s.
Do not recollect or rerun old checkpoints. Report all groups, not favorable clips.
Primary scores remain finite-mixture observable energy, event Brier, false alarms,
supported errors, brick memory and full RGB panels. Respect detector-region
limits and the oracle's finite-sample/duplicate-origin uncertainty.

Intermediate gates retain observed ball reconstruction at least90%,NOOP8
false-ball detections at most10%,last-life32 ending Brier at most.10,and mean32
ending probability at most5% in every non-ending group. Do not reuse the invalid
90% FIRE pair-match gate: report proper distribution scores and oracle spread.
Require a25% reduction in equal-category mean ball-feature energy over16/32
decisions versus coverage,while reporting every category/regression separately.
These are development diagnostics,not a full-world completion criterion.
Untouched tests,calibrated stochastic forecasts,late-game/second-wall coverage,
removed-brick memory and fresh learner confirmation remain unfinished.

## Closed prototype before full learning

Native source498b94e5/SHA5fabcd51 completed the preflight's204 initial/frozen
checks,18 initial/restore arrays and one13-report update. The original worker
still **failed** its final `changed==68` assertion:65 changed, with exactly the
three hidden reward tensors unchanged because the reward output weight starts
at zero. A separate CPU-only70-pin review verifies this against the earlier
motion preflight and preserves the failed guard, broad kernel snapshot limit,
and all original logs. No GPU rerun or full learner occurred.

The pointwise objective is parked because expected per-sample error against one
observed future can favor averaging stochastic futures. The selected replacement
is the separately declared [joint energy experiment](2026-11-07-breakout-world-energy.md).
Archived matching source/runtime and `prototype-closure.json` remain in this
root; do not relabel the failed protocol as passed.
