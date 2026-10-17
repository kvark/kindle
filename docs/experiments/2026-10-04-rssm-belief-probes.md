# Locate readable state in the frozen RSSM

Authorized by “proceed” after the target-standardization report. Reuse all
three standardized final RSSMs (seeds1009/2017/3019) from
`runs/rssm-target-standardization-20261003.9j2EYE` and their corresponding saved
Tiny trajectories. No new gameplay, actor/world/encoder updates or pretraining.
Restore fresh belief/RNG as documented; do not claim uninterrupted training or
bitwise reproduction of the previous diagnostic's random samples.

## One frozen diagnostic

Replay all eight whole trajectories in the existing train/validation/test split,
4,096 recorded actions each. Record the trainable adapter output and full
posterior feature (`deter`, sampled categorical state) at every arrival,
including reset and terminal arrivals. RAM labels never enter the actor or
replay; they are readout targets only. Preserve actual resets and separate
artificial recording cuts. Save before/after and compare all252 saved tensors.

At every fourth action, collect h1/h15 prior states and predicted Tiny features,
retaining terminal targets but never crossing an episode reset. Proposed and
unrelated uniform actions use identical prior categorical draws. The original
prior path is shared; diagnostic reads must not change live belief or RNG.
The h1 deterministic prior must match the deterministic part after the actual
observation arrives: that observation may change the posterior categorical
state, but must not leak into the prior. Full belief is what the policy sees;
the existing feature-prediction head consumes only its deterministic part.

Fit eight small native-GPU readouts per RSSM (24 total): adapter and full
posterior state to current outcome labels, plus deterministic prior, full prior
and predicted features to future labels separately at h1/h15. Use the already
qualified128-hidden/B64 MLP,2,048 updates, fixed head seed20261003, Adam1e−3,
the existing masked-target/regularization recipe, training-only normalization
and validation-only selection every128 updates. No test tuning/refitting.
One head seed per stage does not establish probe initialization robustness.

Reuse the existing Tiny-feature readout. Evaluate both stage-specific fitted
readouts and transfers: posterior-trained readout on prior states, and
Tiny-trained readout on predicted features. Transfers alone confound information
loss with distribution shift. Compare identical forecast origins, actual future
posterior/Tiny readouts, persistence, training-label means and unrelated-action
controls. Report position errors/R², reward/terminal event counts and AUC,
training/validation/test fit, per-trajectory/chunk slices and crossed paired
seed/whole-trajectory bootstrap uncertainty. No posterior score is a forecast.
No RAM-assisted policy, pixel reconstruction or competence claim.

## Budget and interpretation

One120-second guarded native canary checks read-only/legacy-output parity,
one-step prior alignment and categorical-state layout. One5-minute pipeline
smoke replays at most128 actions in each of one train/validation/test trace,
with32 updates/readout on the two player coordinates (short prefixes need not
contain all enemy lanes); its heads are excluded. Then three individually guarded
15-minute extraction-plus-fitting jobs, serialized, no automatic retry.
Expected RTX5080/580.178.04, ≥2GiB sampled Vulkan estimated headroom, ordinary
host-only guards; only the exact reviewed historical warning cursors and VUID
remain allowed. Any new warning/fault stops work for review. No NVML polling or
host recovery. Meganeura main6268ea5 rechecked October4 at03:57, unchanged;
the qualified13b19d33 attention-gradient fix remains pinned.

This can localize poor *readability under this probe recipe*: adapter, posterior
compression, prior transition or feature head/alignment. A failed readout does
not prove information-theoretic absence; a successful one is not usable reward
learning, a good policy or online JEPA efficiency. The representation covers
coarse state, not bullet labels. No larger learning campaign starts here.
