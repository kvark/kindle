# Optional pixel cross-entropy for the isolated Breakout world

The [completed squared-error run](../results/2026-11-07-breakout-world-model-corpus.md)
omits every supported ball in256 fitting/128 development observed reconstructions.
It predicts nearly black at true ball pixels, where sigmoid-MSE output gradients
are about2,500× weaker than logits-based cross-entropy. This motivates one
bounded objective comparison, not a capacity increase, a dataset recollection
or an actor/critic experiment. The world-model goal remains active and unproven.

## Implementation and scope

Add `pixel_loss="cross_entropy"` to the standalone native `WorldModelLearner`.
Default is `squared_error`. Store the choice in checkpoint metadata; absence
means squared error for old checkpoints. Invalid names must fail, and RGB
target standardization is incompatible with cross-entropy. The general agent
keeps its existing squared-error path.

Use elementwise `softplus(logit) - (centered_rgb + .5) * logit`, summed over RGB
pixels and averaged over batch/time. Targets stay soft RGB intensities, not
binary/object labels. Apply it to observed reconstruction and the same shared
decoder's16-step causal prior loss. Preserve loss-only reset masks, terminal
arrivals, RNG streams, image preprocessing, parameter shapes and inference.
No RAM, object labels, emulator state or future boundary labels enter model
input. Future pixels are detached targets only.

This changes objective units, gradient magnitudes and weighting against KL.
Keeping the coefficient100 does not match gradient scale or isolate saturation
as the only mechanism. Compare frozen physical/RGB metrics, not raw training
loss numbers across objectives.

## Qualification before allocation

CPU graph/reference checks must cover logits-80…80, soft RGB targets, complete
pixel sums, partial masks and derivatives. Check that posterior/prior paths
retain parameter names/shapes, no behavior targets, causal prior states and
parameter-gradient connectivity. Run the affected world tests, strict workspace
and bindings Clippy, then package and test the exact runtime.

Preserve the initial CPU bias-broadcast failure and the separate F32 reciprocal
gradient-tolerance failure, with their corrected checks. The initial installed
Python import failure is preserved; packaged tests must use the retained runtime.
These preparation failures do not become successful runs by renaming logs.

The first GPU queue has only three selected qualification phases:

1. Copied Rust test executable: native loss and derivatives against independent
   F64 arithmetic (value tolerance2e-5, derivative2e-6).
2. Cross-entropy tinyB2/T4/H2 mask fixture:12 synthetic updates, finite weighted
   loss, all parameter families change, frozen invariance and exact restore;
   altered masked targets leave prior loss unchanged, all-zero masks give0.
3. RGB64/12M/B8/T32/H16 initial check: all204 initial tensors and18 forecast/
   feature arrays match the retained squared-error control exactly; old metadata
   restores as squared error with matching forecasts. One full-shape update must
   be finite with correct loss arithmetic. This checkpoint is discarded as a
   learning start; the comparison starts fresh from the matched initialization.

Do not launch a learner until these results have been reviewed. A failed phase
stops its queue; preserve outputs and use a separate declaration for any repair.
Check live service/process identity; an observer timeout is not a stopped job.
Use the existing AMD host guard, Auto precision and matching runtime. No driver
reset/reload/reboot or automatic restart.

## Fixed comparison after successful qualification

Declare one8,192-update learner, followed by its frozen restore check. No
automatic extension or successor. Reuse the exact fitting/development corpus,
config, seed41001, B8/T32, RGB64,12M preset, rates4e-4, reconstruction coefficient100,
KL1/.1/free-nat1,16 observed+16 prior steps and uniform source-sampling plan
from `/mnt/data/kindle-breakout-world-corpus-20261107`.

Fit only on16,384 actions/65,503 emulator frames; development8,192 actions/
32,755 frames remains separate. No collection or coupled learner repeats.
All18 action counts are nonzero, but state/action and late-game coverage remain
limited. Assert exact initial tensors, exact sampler plan and evaluation origins
against the control before the first update.

Keep0/1024/.../8192 frozen natural/contact/ending exports, four fixed draws,
1/4/8/16/32 decisions and final256 fitting/128 development origins. Retain all
204 final parameter/moment tensors unchanged through frozen evaluation and
18-array exact restore. Audit all finite updates, exposure counters and pins.
Report native training and total loop wall-time separately; replayed arrivals
are not new actions or emulator frames.

Primary comparison is observed-ball detection and within1-pixel accuracy on the
same development origins, against control0/112. A promising reconstruction
recipe should recover at least90% of supported observed balls within1 pixel;
this is an intermediate diagnostic gate, not world-model qualification. Report
fitting results, paddle/brick retention, all draws and misses. Forecast errors,
reward/endings and frozen fitting-only latent readouts remain required evidence.
Do not select a favorable draw or score a blurred mean as a physical trajectory.

The comparison uses reused development data/initialization; it is not an
independent reliability test. Even passing reconstruction does not qualify
prospective action effects, calibrated stochastic outcomes, untouched tests,
late-game/tunnel/second-wall dynamics or fresh confirmation. Keep actor/critic
parked until world competence is established.

Root `/mnt/data/kindle-breakout-world-bce-20261107`. Native build source535de479,
SHA256 `adfc2f9ec8e9174b56628ab8c390516ee6acee31dc4151418208bcfdf7f59c54`.
All three native workers exit0: loss/derivatives, mask fixture and matched-init
preflight. The qualification queue **fails its declaration**, expecting17
compared arrays when the runner reports16. Preserve that original failure;
no GPU phase repeats. A separate CPU review verifies all304 pins and all18
saved arrays, including horizons/locations, against both the original control
and legacy restore. Initial204 tensors are exact; the full-size update is finite.
This completes missing comparison evidence without relabeling the failed queue.
The runner now explicitly checks/reports all18 arrays for future restores.

The fixed8,192-update comparison is selected after that CPU review. Pin the
current execution runner/source separately from native build535de479, whose
Rust/Python-library code is unchanged. Reuse the original data/config. No
automatic successor is selected. The two-stage learner/restore queue is now
active under execution878f66be, with306 pinned inputs and unchanged native
build535de479. Initial tensors, sampler and origins match the control; first128
updates are finite. Retain actual-initial/sampling assertions and stop after the
fixed budget. A qualification failure is not a learning win.
