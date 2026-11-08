# Breakout: posterior-trained local-action world

Select implementation and bounded qualification of the existing local-action
decoder with H0 observed-posterior training. The [loss review](../results/2026-11-08-breakout-world-prior-pixel-loss-review.md)
identifies an averaging incentive in the added sampled-prior RGB term; the
[fitting result](../results/2026-11-08-breakout-world-launch-fitting.md) supplies
compatible visual evidence. Neither proves a sufficient repair.
Root `/mnt/data/kindle-breakout-world-local-h0-20261108`.

No full learner is selected yet. Keep unchanged CNN/RSSM/decoder/inference,
size1M/RGB64,603,915 weights/80 logical/240 tensors,B8/T32/full BPTT32,Auto,
rates4e-4,RGB100,KL1/.1/free-nat1,observed reward/continuation1,mixing.01.
Allow explicit `motion_cross_entropy` with prediction_length0 for local-action
worlds,without transport/residual/feedback/Gaussian/previous-frame/latent-only
options. Preserve all old H16/v18 and transport/v19 restrictions and semantics.
Use a separate v20 checkpoint identity for the new objective/geometry combination;
ordinary restore must not change objectives or accept disguised older headers.
No new trainable weights,primitive,coordinate input,object/RAM target,encoder,
actor/critic,CDP objective or game action mapping. No JAX/backend optimization.

CPU checks must cover valid/invalid format/objective flags,pre-GPU rejection,
unchanged parameter names/shapes/counts and causal inference inputs,9 observed
reports without suffix-loss/noise inputs,all required gradient connections,
legacy/header compatibility and normal restore validation. Run relevant existing
CPU/core-binding checks and strict Clippy once after changes;reuse existing
Conv5/RMS/upsample/categorical primitive qualifications,not a new backend project.
Build/package with portable locked dependencies into this root and retain exact
source/native/runner identities;do not overlay a retained runtime.

After CPU success,declare a guarded single-update first-shape qualification and
initial/trained restore check on the actual B8/T32/H0 data. Require finite9-report
components,weighted-total/mask identities,required changed parameter families,
all240 finite/frozen tensors/ages and18 exact restore arrays. Verify bounded
initialization/source bytes,current healthy host/driver/live process identity and
existing host guard. No automatic resets/reloads/reboots or stale queue restart.
No repeated completed derivative/qualification stage;preserve any new failure.

Prospective bounded learner,still unselected: fresh16,384 updates on the retained
corrective4,096-row plan repeatedfour times,with its exact new data and sampler.
Reuse sealed counts scaledfourfold;do not recount the completed old plan ledger.
H0 has no imagined losses/terminal mask targets. Data,objective,budget/initialization
and removal of imagined-event loss change together;no isolated causal or seed
reliability claim. The old H0 model has a different global decoder.

A separate learning declaration follows successful qualification review. It must
pin all inputs/plan/counts and actual-initial controls,save useful milestones,
retain curves and finite reports,do final dense/frozen/restore checks,then
separately declare final-only chosen-action evaluation. Reuse cached controls
and original oracle outcomes;all original physical/event gates,regressions and
individual alarms stand. Distribution scores and coherent sampled images matter;
raw pairwise point matching is not a universal stochastic success gate.

Do not restart completed models/collectors/diagnostics,extend an unchanged learner
or append automatic successors. Full precise physics,event timing/calibration,
persistent brick memory,late/tunnel/second-wall/above-top coverage,untouched
families and fresh confirmation remain required before the world goal can close.
