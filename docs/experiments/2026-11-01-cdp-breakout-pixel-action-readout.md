# Fixed CPU readout: learned Breakout features plus current pixel positions

The retained encoder4e-5 features decode ball position more precisely than the
older encoder, but remain much less precise than the saved RGB64 pixel reference.
Existing learned-feature action readouts trail its actor. Before considering a
native spatial-input change, select one CPU screen of whether current pixel
positions add useful teacher-action information to those retained features.

Reuse the original6e-6 control trajectories and the completed4e-5 feature capture,
actor probabilities and action readout predictions. Keep all2,064 rows and the
original lane split:954 training rows on0–3,509 validation on4–5,601 test on6–7,
including136 near-descending test rows. Reuse the saved RGB64 pixel predictions;
no new image processing, emulator replay, native forward pass or learner occurs.

Add exactly five current-observation columns: ball visibility,x/159,y/209,
paddle visibility andpaddle x/159. Missing positions are zero with visibility
off. Use saved pixel predictions, never reference target positions. Do not add
motion, previous commands, teacher landing target, teacher action, phase or
future observations as inputs. Keep all missing/extra detector cases. Validate
unique lane/index identities, splits, phases and original pixel/capture hashes.

The primary readout averages two RBF kernels with equal fixed weight: one over
the3,584 concatenated CNN/recurrent features and one over the five pixel columns.
Normalize each block with its own training-only means/standard deviations and
width scaling. Each bandwidth is the median positive off-diagonal training
squared distance. Center each kernel using training statistics. Use the existing
seven ridge penalties1e-4 through100, chosen independently per one-hot action
target by validation MSE. Validation never refits; test labels never select.
Fit a pixel-only kernel as a separately reported diagnostic under the same rule.
Do not refit either original learned-feature baseline or native actor.

Require augmented teacher agreement at least **five percentage points above
both the retained combined kernel and actor**, overall and near descending,
and higher than both in each test lane. Report both new readouts regardless of
screen outcome, all class/phase/lane confusion counts and exact selections.
No coefficient, input-set, bandwidth or split retry follows a negative result.

Qualify exact pixel joining, missing inputs, rejection cases, independence from
reference targets, training-only statistics, one-block equivalence with the
qualified kernel fitter, and a direct scalar/matrix ridge comparison. Pin all
consumed data, source, protocol and output receipts before the first real fit.
Keep analysis60s/controller90s, one low-priority CPU core and BLAS thread.
Block native/emulator imports and preserve active Qbert seed12011's562 pins,
PID842010/invocationd37840f258104f9caf8b6bc5c0316899 before/after. Retain CPU
co-host timing without correcting the learner's measured training time.

This is a reused development corpus with game-specific color/crop priors and
teacher labels sharing related information. The additive kernel changes readout
geometry as well as supplied information. A positive result would support
considering a spatial-input prototype; it would not prove missing feature
information, a causal limitation, native-path correctness or gameplay benefit.
No head replacement, architecture, loss, teacher or learner is selected by this
screen alone. The original Breakout two-wall gate and five-game goal remain.

Root `/x/Temp/kindle-breakout-pixel-action-readout-20261101`. Qbert confirmation
continues under its original observers; do not restart or overlap its GPU job.
