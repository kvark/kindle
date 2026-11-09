# Breakout: posterior-only local H0 improves one step, fails sustained forecasts

The fixed16,384-update world-only learner completes its training, frozen
assessment and CPU reviews, but fails retention. Observed and one-step ball
precision improve; longer forecasts regress on fitting and development data.
The chosen-action primary ball distribution score is.429470, versus the
retained global H0 reference.169502 and required maximum.127127. Five of ten
intermediate gates pass. The full world-model goal remains unproven.

[Learning protocol](../experiments/2026-11-08-breakout-world-local-h0-learning.md),
[fixed final assessment](../experiments/2026-11-08-breakout-world-local-h0-assessment.md).
Root `/mnt/data/kindle-breakout-world-local-h0-20261108`; detailed comparisons,
all regressions, configurations, failures, sources, checkpoints and clips remain
there. No extension, sweep or automatic successor is selected. Actor/critic and
CDP comparison remain parked.

## What was trained

One fresh size1M/RGB64/v20 model,603,915 weights/80 logical parameters/240 saved
tensors; native/source/runner5bcaaebd, native SHA256
`6930e7b8ddfd8c6595266033afdee814b1697e12ee767af58ca825e345997846`.
B8/T32/full BPTT32, Auto, seed41001, rates4e-4, observed RGB1, KL1/.1,
free-nat1, observed reward/continuation1, mixing.01. Keep the local-action
CNN/categorical RSSM/two-image decoder history and inference. H0 removes the
added sampled-prior RGB and imagined-event objectives. No actor, critic, CDP,
pretrained encoder, object/RAM target or new primitive.

The retained corrective4,096-row plan repeats four times continuously:
496 streams/62,666 source arrivals,57,215 unique arrivals,20,604 terminal rows,
44,564 positive targets,18,268 source resets and148,988 context resets. Reuse
sealed counts scaled fourfold, without repeating an old generic recount.
No new emulator actions or frames during training or assessment.

Compared with small64, the objectives, RGB scale and data/sampler change together.
Compared with the warm corrective model, initialization/age also differ. These
comparisons neither isolate the removed loss nor establish a general H0 failure.
The earlier loss review remains an argument about the added term's averaging
incentive; it did not guarantee that removing it would repair the coupled model.

## State reading succeeds more than rollout

Dense natural forecasts use the same256 fitting/128 development origins and
supported truth counts. Fractions below count visible truth balls within one
RGB64 pixel; missing predictions fail. Fixed first draw, no best-draw selection.
Point matching alone is limited when several futures are valid.

| Scope/model | Observed | One step | Four | Eight | Sixteen | Thirty-two |
|---|---:|---:|---:|---:|---:|---:|
| Development, new local H0/RGB1 |109/112|107/109|60/109|15/107|3/114|0/95|
| Development, retained small64 H16/RGB100 |103/112|102/109|69/109|53/107|22/114|8/95|
| Fitting, new local H0/RGB1 |155/163|164/177|90/180|36/176|4/171|2/179|
| Fitting, retained small64 H16/RGB100 |146/163|154/177|107/180|84/176|37/171|20/179|

The actual initial control has zero observed/future ball matches. Its240 weights
and18 forecast arrays match the qualified initial exactly; dense/curve controls
are reused without querying an old model. The new model demonstrably learns
observed reconstruction and immediate prediction. The fitting rollout gap shows
that failure is not only held-out generalization. It does not by itself separate
latent-prior error, generated-history drift, loss balance or optimization.

## Chosen-action distributions and events also fail

The final-only84 groups use42 plans, histories16/32, eight draws and endpoints
1/4/8/16/32. Physical truth joins match210 arrays; new image-derived ball/paddle
features match168 exports. Five sealed controls are reused without old model
queries, old predicted RGB reads or repeated completed scalar/point tests.
All action/horizon ball/paddle/reward/ending/brick regressions and both histories
remain in `supplementary-metrics.json`, including every individual alarm.

| Model | Balanced survival-weighted ball energy at16/32; lower is better |
|---|---:|
| Global H0 reference |.169502|
| Retained1M small64 H16 |.172366|
| Warm launch-corrective H16 |.229385|
|12M local-action H16 |.240566|
|12M action/transport H16 |.387593|
| New1M local H0/RGB1 |.429470|

New category energies: serve.334534, freeflight.531412, contact.276698,
brick.463521, wall.281147, life-loss.642945, ending.476035. Versus small64,
only serve improves on this aggregate; survival weighting prevents interpreting
that alone as better image physics. Matched-point accuracy over all supported
model/oracle pairs is13.051%/0.780%/0.096% at8/16/32; these are related pairs,
not independent trials or universal stochastic pass criteria.

Freeflight eight-step ball presence is8/64 under NOOP,48/64 LEFT,56/64 RIGHT
and48/64 reversal. All four fail the90% gate. Serve NOOP has no false ball at8.
Ending16 Brier.002660 and RIGHT16 risk.068670 pass their false-alarm checks;
these early selected outcomes are all alive, so they do not prove genuine ending
recognition. Ending32 Brier.268326 fails. Nonending alarms fall to68/304, but
maximum risk.496106 and category means still fail the nonending gate.

Passing gates: observed ball, one-step ball, serve NOOP absence, ending16 Brier,
RIGHT16 risk. Failing: eight-step ball, ending32 Brier, nonending category risk,
primary ball energy and per-command freeflight presence.

Fixed first-draw/lane0 causal clips retain all96 lossless frames for serve
LEFTFIRE, freeflight NOOP and contact LEFT. Their decision8 posters show diffuse
or missing moving detail while the static wall remains recognizable. A detector
centroid is not proof of a coherent sampled ball. These three fixed cases are
illustrations, not a measured population-wide causal diagnosis.

## Execution, pause and verification

The user pause froze the live worker after7,852 logged updates. The same native
PID1929598/start ticks113967517 resumed; no checkpoint restore, RNG reset or
extra update. Buffered logging is not an exact live optimizer-age observation:
the paused native-call gap ends at update7,874. The expired original raw wall
clock is explicitly amended by pinned supplemental supervision; the original
controller/guard stop until native exit, then finish their declared restore.
CPU identity/termination/exit checks, kernel-gap evidence and all three
train/restore/continuation guards pass. No original raw-wall-cap pass is claimed.

Raw native-call/total times are13,850.512s/13,976.831s. Exact systemd journal
freeze/thaw duration is12,767.850s. Subtracting only that interval yields
1,082.662s/1,208.981s (about18m03s/20m09s). The native-call timer includes the
first-state check and synchronous wrapper work. These are not isolated GPU
kernel times or a clean matched throughput comparison; raw clocks remain intact.
Learning curves use updates. Final frozen assessment takes122.866s; CPU score
review14.250s.

All697 learner input pins,216 CPU-audit pins and16,384 finite nine-component
rows pass; all80 logical weights change. First actual RGB1 update agrees with
qualified unweighted forward components and correct weighted total; all240
saved tensors are finite and ages/configuration agree. Initial240 tensors and
18 arrays match the qualifier; final/frozen240 tensors are exact, and final
restore matches18 arrays. All51 natural forecast exports and84 full causal
exports are finite. Assessment1,574 input/236 CPU-audit pins, one successful
nine-file guard and terminal unit; artifact36-input review verifies96 exact
clip frames. Earlier qualification graph-count and unused-counter byte-identity
failures remain with their separate repairs; no original failure is relabeled.

Original learning invocation `d69b4c399521422087535f14af4797ac`, supplemental
`18f1aae6ca364bea8a4104e98715f86d`, assessment
`b4095694bf8045f3bdc9b22ebfe33ff5`. No host recovery or driver change. Runtime,
wheels and CPU executable remain; disposable3,360,843,323-byte build cache was
released before the pause. Detailed closure receipt records terminal services,
four learning/assessment/supplemental guard audits and all artifact pins.

## Implication for the world-only goal

Keep pixel supervision as the inspectable reference and retain the existing
baselines. This run is a negative combined recipe, not proof that CDP or another
pixel generator will solve the problem. Before another full learner, a separately
justified test should distinguish one-step state accuracy from accumulated
latent and generated-history errors on fitting sequences; previous state/history
swap and contact-fitting results already constrain that investigation and must
not be repeated unchanged.

The full goal still requires precise physics and events, natural-frequency
calibration, persistent removed-brick memory, late/tunnel/second-wall/above-top
coverage, untouched families and fresh learner confirmation. Existing related
development histories have no above-top-brick endpoint. Neither passing short
forecasts nor a sharper image would discharge those requirements.
