# Frozen-encoder RSSM target-standardization ablation

Authorized by the user's “proceed” following the fixed-latent report. That
diagnostic establishes readable/predictable state, not useful online dynamics.
This test changes one factor in the production Dreamer learner: the units of
its future-feature target. No new gameplay, Tiny backward or pretraining.

## Declared comparison

Six fresh RSSMs: raw versus standardized targets, paired seeds1009/2017/3019,
each using its own corresponding trained-Tiny corpus from
`runs/fixed-tiny-latents-20261003.5uHqSE`. Both arms ingest the identical raw
features/actions/rewards; initialization and RNG seeds match. Standardization
is per-coordinate `(target - train_mean) / train_std`, F64 training-split
statistics cast to F32, std≤1e−6 replaced by1. No test/validation statistics.
The head predicts standardized absolute features, **not residuals**. Forecasts
are converted back to raw units before scoring. Encoder/RSSM inputs stay raw.

Other settings match the earlier direct-policy Size1M/B8/T16/H15/microbatch1
recipe:4e−5 learning rate,1,000-update warmup, AGC.3, future loss.25, unchanged
KL/reward/continuation/replay-value/actor/value settings and ac_grads=true.
There is no Tiny graph: the frozen features are already recorded. Replay capacity
16,384 keeps every training arrival; the scheduler is off and2,048 explicit
production `DreamerCore::learn` updates are performed. Raw and standardized
arms differ only in the optional target statistics. Save exact initial/final
weights and audit initial parameter equality across each pair.

The three whole training traces contain12,288 recorded actions plus separate
reset arrivals. Insert all before learning; no online interaction accounting
claim. Training curves every128 updates; final-budget selection only, no
validation tuning or best-seed selection. All encoder weights remain untouched.

## Frozen diagnostics

After training, no further updates. Evaluate all three test traces with origins
every4 actions, horizons1/15, retaining terminal targets without crossing resets.
Validation uses both complete traces at stride16; training-fit diagnostics use
the first512 actions of each training trace at stride16. These choices are
fixed before execution. Save before/after checkpoints and compare all tensor
keys/shapes/dtypes/bytes, not file serialization hashes.

Controls: latent persistence, the constant training-feature mean, and unrelated
uniform actions with identical prior categorical draws. A standardized head
starts near the training mean; beating the raw arm alone is not dynamic skill.
Report raw and training-scale-normalized errors, by trajectory and encoder chunk
crossing, direct reward forecasts versus zero/training-mean reward, event counts
and decoded state/event forecasts using the already fitted frozen GPU readouts.
Privileged RAM labels never enter RSSM training or replay. Target statistics
use only features, not these labels. Resample whole trajectories and paired
learner seeds, not correlated frames. No gameplay competence claim.

## Qualification and finite budget

First qualify independent full-width F64 target/output transforms and raw
gradients; default-off node identity; three-update identity-statistics parity;
nonidentity full updates, exact checkpoint restore and frozen forecasts.
One guarded production Size1M plumbing smoke uses2 updates and at most512
recorded training transitions plus128 recorded evaluation transitions. Its
outputs are excluded from the six-run comparison; no gameplay is added.

Each numerical test has a120-second deadline. The plumbing smoke has a5-minute
deadline; each complete paired arm has a20-minute deadline. Serialize all GPU
work, ordinary host guards, expected RTX5080/580.178.04 and≥2GiB sampled Vulkan
estimated headroom. Stop/review every failure; no automatic retry, NVML polling
or driver recovery. Existing exact reviewed startup-warning cursors remain
historical; any new allocation warning is fatal. Meganeura main was rechecked
October3 at23:24:6268ea5, unchanged;13b19d33 includes the qualified dV fix.

This ablation can identify a benefit of fixed target conditioning on a frozen
corpus. It cannot establish stable joint representation learning, a JEPA
efficiency advantage, or that every important visual detail is represented.
Centering changes initial raw forecasts; per-coordinate scaling also changes
error weighting relative to other losses. It is one defined target transform,
not a mathematically identical objective or an isolation of centering from
variance scaling. The constant-mean control helps distinguish these effects
from learned dynamics; neither arm's training loss is in comparable raw units.
No larger gameplay queue, Phase3 or swarm starts automatically.
