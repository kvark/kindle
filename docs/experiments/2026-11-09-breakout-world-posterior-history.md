# Breakout: posterior-conditioned generated image history

The [one-time refresh diagnostic](../results/2026-11-09-breakout-world-refresh.md)
finds temporary recovery from correcting image history, without comparable
recovery from a latent-only correction. The [local H0/RGB1 learner](../results/2026-11-09-breakout-world-local-h0.md)
learns observed/one-step images but fails longer fitting and development
forecasts. Select a distinct training objective to expose the decoder to its
own image history while retaining outcome-conditioned posterior reconstruction.
Root `/mnt/data/kindle-breakout-world-posterior-history-20261109`.

Add explicit `motion_posterior_history`, checkpoint v21, only for the local-action
CNN/categorical RSSM with a positive suffix shorter than the sequence. The
ordinary observed RGB/KL/reward/continuation losses remain. The added suffix
reconstruction uses the posterior state associated with each real arrival, but
its two past images come from previous decoded predictions, seeded once by the
real prefix. It does not draw independent prior futures to match a different
realized image. This removes that particular independent-sample averaging
incentive; it does not guarantee coherent samples or a useful prior.

Keep full derivatives through generated image history, the existing motion
weights and rollout support masks. Clear history at actual sequence resets.
Real images inform posterior states and stopped targets during training;
inference remains unchanged, using only prior samples, past images and actions.
No current target image is directly inserted into generated suffix history.
No additional imagined-event loss, weight, pretrained encoder, object/RAM target,
backend primitive, actor or critic. Existing v18/v20 and defaults stay unchanged;
restore preserves v21 objective/geometry and rejects disguised legacy headers.

CPU checks cover v21/legacy/scope validation before GPU construction, identical
parameter geometry, posterior versus sampled-prior ancestry, action/target time
causality, generated-image recurrence and gradient connectivity. Reuse qualified
native decoder/history/motion-CE primitives and their value/derivative receipts;
no new full-backend qualification matrix or independent full-CNN F64 claim.
Package the matching runtime/runner separately and pin source/config/data.

Select one bounded fresh2,048-update development learner: size1M/RGB64/B8/T32,
full BPTT32, suffix16, Auto, seed41001, rates4e-4, RGB1, KL1/.1/free-nat1,
observed reward/continuation1, mixing.01. Same80 logical/603,915 weights/240
saved tensors and inference. The extra reconstruction also changes total loss
balance; no isolated-cause claim. No warm start, extension, sweep, best-checkpoint
selection or automatic successor.

Use only the first2,048 rows of the retained4,096-row broad corrective plan,
covering the existing496 natural/launch/contact/life-loss/ending streams. Count
this new prefix's exposures and suffix support once, without recounting an old
full plan. No new gameplay. Keep16 initial natural source streams for dense
fitting evaluation and the original development streams. All sample selection
and data are reused development material, not untouched confirmation.

The first actual update counts toward2,048. Before a second update, require11
finite scalar reports, correct combined total and support fraction, correct
initial240 tensors/18 inference arrays against the retained actual-initial
control, and agreement of unweighted ordinary losses with the old first update
within1e-6. Save/check all240 finite tensors and required changed families at
age1. Stop on failure. These state checks are included in call timing, not an
isolated GPU benchmark. The new temporal objective's usefulness remains untested.

Use one bounded host-guarded train/restore queue. Save initial/512/1024/final,
fixed curves every512, final256 fitting/128 development natural origins, frozen
240 tensors and18-array final restore. Keep all finite loss rows, new plan counts,
source/native identity and terminal service/guard evidence. Initial dense/curve
controls may be reused only after own initial240 weights and18 arrays match;
never query an old model. Original preparation/qualification failures remain
retained with separate repairs rather than relabeling them as passed.

After CPU learning review, separately declare a final-only84-group chosen-action
assessment with the original oracle, histories16/32, eight draws and five
endpoints. Reuse completed control scores without old model queries or repeated
completed scalar/point tests. Preserve all physics/event gates, every action/
horizon regression, individual alarm and fixed first-draw clips. A2,048-update
budget does not relax those gates or complete the full goal.

The original full-game requirements remain: precise physics/events, natural-
frequency calibration, persistent removed-brick memory, late/tunnel/second-wall/
above-top coverage, untouched families and fresh learner confirmation. Existing
oracle has no above-top-brick endpoint. No actor/critic restart follows from a
lower reconstruction loss, sharper sample or easier fitting subset.

Preparation results:the original CPU compile stops on a JSON test-literal syntax
error. After that fixture repair,the broad run completes131 passed/3 failed/39
ignored. It finishes before a proposed stop;no process is signaled. The failures
catch a missing v21 restore allowlist,an incorrect test reward-head size,and a
last display sigmoid incorrectly required as a loss ancestor. Original logs and
source remain. Focused corrections pass all three new checks and13 affected
legacy header checks;strict core Clippy and326 installed-package CPU checks
pass. Graph-output checks include the existing padded report slots andtwo
posterior-state traces,not just the11 scalar API reports. New CPU graph geometry
uses a shorter sequence with the same80/603,915 parameters to avoid repeating
large legacy graph work.

The first unqualified wheel is overwritten before retention;its original source
and build log remain. An intermediate objective-rejecting wheel is retained in
a correctly labeled directory. Only the separately built candidate is installed
for this experiment. No original failed comparison is relabeled. New prefix
counts:2,575 observed terminal rows,2,394 supported suffix terminal rows,5,552
positive targets,18,636 context resets,2,294 source resets,56,189 unique arrivals.
The legacy counter name prior_terminal_rows denotes posterior suffix support
here;no prior future is sampled for this objective. Checkpoints at512/1024 are
saved inside the learner wrapper,so call timing includes those saves as well as
the first-state check.
