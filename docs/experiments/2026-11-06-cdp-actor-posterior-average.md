# Actor posterior averaging: frozen development protocol

Select a default-off native inference option that averages the actor's action
probabilities over eight independent draws from the current posterior. The live
RSSM retains its original single sampled state and RNG stream. Only the actor's
decision marginalizes the additional draws; no critic, world loss, training
recipe, checkpoint parameter or image transform changes.

The [modal-posterior assessment](../results/2026-11-04-cdp-breakout-value1-modal.md)
raised the retained greedy mean 143.667→175.708, but its paired interval crossed
zero and its recurrent state also changed. The [64×8 grouping experiment](../results/2026-11-06-cdp-latent-shape.md)
failed primary retention. Neither identifies posterior noise as the cause.
This is a distinct decision-time intervention, not a modal retry, another latent
grouping learner, or a claim of a training-efficiency improvement.

Use the existing sampled posterior logits and deterministic state. Draw eight
complete categorical states from their mixed distribution (latent unimix .01),
run the same native actor on all eight, and average softmax probabilities. Return
log probabilities to the existing action selector; clamp the mean to at least
1e-30 before log for numerical safety. Apply existing actor unimix once, after
averaging. Greedy selection takes the largest averaged probability. A separate
per-lane RNG, seeded by config seed plus lane and XOR 0x4d415247494e414c, supplies
only these additional draws. Its state is transient and not checkpointed.
Original posterior/policy RNG streams and actual-command feedback stay intact.
This averages conditional actor outputs, not value estimates, independent
recurrent belief histories, or the actor applied to an averaged latent vector.

Expose a frozen-only, pre-observation API and `--actor-posterior-samples` CLI
argument, supporting bounded counts 2–16 but selecting only eight here. Reject
training, missing restore, modal posterior and action-temperature combinations.
Both native learning entry points and the Python entry point must reject the
option before updates. Log it as evaluation metadata; checkpoint architecture
and configuration stay unchanged. Default inference graphs remain unchanged.

First implement and check CPU graph shapes/parameter identity, argument guards,
installed-runner lifecycle and independent numerical reference logic. Prepare a
tiny guarded native fixture checking fixed-draw sampling, arithmetic averaging
against separately evaluated actor states, shared parameters, masks, frozen
weights/moments/counters, unmodified original RNG streams and recurrent state
under identical forced actions. Use nonuniform actor outputs so averaging cannot
pass through a uniform-policy fixture. No derivative or optimizer test is needed
for this inference-only path. A separate native declaration must pin packaged
bytes, device/guard/caps and the exact fixture before GPU execution.

If qualified, separately declare one bounded compatibility/cost check and then
a trained-checkpoint greedy comparison against ordinary sampled-state inference.
Reuse the retained 200k checkpoint; no learner or initial-model arm is selected
here. Preserve natural-episode evaluation, fixed cohorts, tensor invariance,
full replays/videos, inference cost and original 864/90%-two-wall gate. The
evaluation declaration must fix its improvement rule before scores are observed.
No sample-count sweep, extension, repeated completed stage or automatic successor.

Root `/x/Temp/kindle-actor-posterior-average-20261106`. This declaration selects
implementation and CPU qualification only; no GPU allocation is active or selected.
Retain 143.667 greedy/70.750 sampled and the five-game goal unchanged.
