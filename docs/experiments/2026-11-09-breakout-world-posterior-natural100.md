# Posterior-history learning on the retained natural RGB100 recipe

The [local forward comparison](../results/2026-11-09-breakout-world-local-forward-contract.md)
finds exact forced-sample recurrent states and generated-history suffix pixels,
with retained small observed/sampling failures. It does not explain the poor
free rollouts. The [real-history probe](../results/2026-11-09-breakout-world-oracle-history.md)
finds recovery when actual past observations/history replace predictions, but
changes both posterior and local history. Investigate training the generated
image recurrence with outcome-conditioned posterior states, using the existing
MotionPosteriorHistory implementation. No implementation change is selected.

The [earlier posterior-history run](../results/2026-11-09-breakout-world-posterior-history.md)
fails at2,048 updates/RGB1 on broad corrective data. Do not extend or repeat it.
Select a fresh, distinct recipe at RGB100 on the retained natural two-cycle
plan, with16,384 updates. These changes to data, loss balance and budget are
bundled; no isolated cause or broad reliability claim is allowed. Root
`/x/Temp/kindle-world-posterior-natural100-20261109`.

## Recipe and fixed comparison

Reuse the qualified posterior-history64f5489f runtime/runner, native SHA256
aa12d0507897b09d69d8d41cd7a1cd010f183ee47ffb99d1217b10a901956c16, from
`/mnt/data/kindle-breakout-world-posterior-history-20261109`. No rebuild or
overlay.1M/RGB64/B8/T32/full BPTT32/H16/Auto/41001,80 weights/603,915 scalars/
240 saved arrays. RGB100, rates4e-4, KL1/.1/freeNat1, mixing.01, observed events1.
One-hot actions; no physical-feature table, late conditioning or auxiliary head.

Use exactly the retained small64 configuration and16,384-row two-cycle plan.
Compare final-only frozen forecasts to cached small64/early-feature/late-command
results without querying an old model. Relative to small64, the changed objective
uses posterior states for generated-history reconstruction and removes imagined
reward/continuation losses; ordinary observed RGB/KL/events and inference remain.
Source revision and wrappers also differ. This is a recipe comparison, not an
isolated test of posterior conditioning. The v21 format remains explicit.

The suffix uses image-informed posterior states during training, but previous
decoded predictions as image history, seeded once from the real prefix. Keep
derivatives through generated history and existing reset/support masks. Future
actual images inform stopped targets and training posterior inference; they
cannot correct an evaluation trajectory. Inference uses only prior samples,
past images and submitted future commands. No object/RAM training targets,
pretrained encoder, actor, critic, new collection or emulator action/frame.

## Bounded execution and decision

One fresh learner, no warm start, extension, sweep, checkpoint selection or
automatic successor. The first actual update is inside16,384. Before update2,
require own initial240 tensors and18 cached-control inference arrays exact,
11 finite reports, ordinary unweighted components1..8 matching the retained
small64 first update at1e-6 tolerance, correct combined loss/support fraction,
240 finite age1 tensors and changes in every required encoder/posterior/dynamics/
decoder/reward/continuation family. Stop on failure; no automatic retry.

Reuse the existing source-count receipts without recounting completed exposure
plans:32,850 source arrivals/32,849 unique,4,194,304 replay rows,1,951,882 valid
suffix targets,19,140 supported suffix terminals,32,016 positive suffix targets.
The legacy prior_terminal_rows counter denotes posterior suffix support here.
Pin actual config, exact source/native/runner, data, plan, controls and scripts.
Run a meaningful CPU driver check with model/emulator construction trapped.

Declare a serial host-guarded train/restore queue on the healthy AMD/Auto host:
train cap4,200s, restore600s. Save initial/age1/2048/4096/final, fixed curves
every2048, actual-initial and final dense256 fitting/128 development origins,
all finite reports and240 final/frozen tensors. Require exact18-array restore.
No old queue, NVML poll, driver recovery or learner restart. Check authoritative
live service/process identity; observer timeout alone does not stop a job.

Declare CPU analysis against terminal closed inputs, excluding open logs and
outputs from their own manifests. Automate guard audit, sampler/loss/age/tensor/
forecast checks and fixed final comparisons; preserve every regression and
original failure with any separate CPU repair. Intermediate gates stay observed
>=.90, one-step>=89/109 and eight-step>=28/107 within one RGB64 pixel, first fixed
draw, missing/diffuse cases failing. Do not infer mastery from passing them.
Retain all draws, paddle/brick/event scores, curves and fixed figures. A broader
frozen physics assessment requires a separate declaration after review; it is
not allocated here. Update PR31/current work and keep large artifacts outside Git.

Precise action-conditioned physics, natural event calibration, persistent removed-
brick memory, late/tunnel/second-wall/above-top coverage, untouched families and
fresh learner confirmation remain required. Actor/critic, CDP and five-game work
stay parked. The full world-model goal remains active and unproven.
