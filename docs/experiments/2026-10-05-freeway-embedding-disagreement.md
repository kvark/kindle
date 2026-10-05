# Freeway: predict the existing CNN embedding for exploration

The [soft-target screen](../results/2026-10-05-freeway-soft-learning.md) removes
sampling-loss variance without improving bonus, coverage or reward discovery.
Saved-feature probes find player position more readable from the existing CNN
than either sampled stochastic features or reconstructed posterior probabilities.
Test the existing detached arrival encoding as the ensemble target. This is
Plan2Explore's [embedding-target option](https://github.com/danijar/dreamerv2/blob/main/dreamerv2/expl.py),
not a new pixel encoder or an exact reproduction of that system.

Only the target changes: next CNN encoding instead of posterior probabilities.
With Size1M RGB this is256 outputs/head rather than128, adding33,280 ensemble
parameters. Preserve its existing RMSNorm/SiLU feature scale; no added
normalization or bonus coefficient change. The changed information, output width
and natural scale are inseparable in this screen and must be disclosed. Inputs
remain previous RSSM feature plus actual action. Detached inputs/targets prevent
the ensemble from shaping the encoder/RSSM. The CDP world loss is unchanged.

Do not keep a production target-selection matrix or migrate old ensemble
checkpoints. The previous runs retain their original native binary identity.
Fresh training only; extrinsic-only checkpoint shapes remain unchanged.

## Qualification and finite budget

CPU graph test verifies the one-transition loss reaches the arrival CNN encoding
but no current posterior parameters or categorical draw. The full target graph
must give gradients only to ensemble parameters. Run that guarded GPU check plus
the existing independent F64 bonus/raw-derivative, repeat-versus-novel/detachment,
transition alignment and zero-scale equivalence checks: five120-second jobs.
Local CPU tests, formatting/Clippy, then one excluded2,048-action/451-update
production smoke (seed9001) and1,024 frozen actions with all saved tensors equal.
Review before learning; these are not competence or throughput comparisons.

Three fresh seeds1009/2017/3019, each32,768 actual aggregate actions/8,131 updates.
Compare with the retained three soft-target and three extrinsic-only controls.
No additional unchanged control training. Freeway/full18/sticky.25/repeat4/no
reset no-ops, one GPU RGB64 resize, Size1M/N8/B8/T16/H15/R32/microbatch8/replay100000.
CDP cosine500, encoder6e-6/dynamics4e-4/base4e-5, warmup1000, AGC.3, ac_grads=false.
Disagreement coefficient1; no action aid, video prior or external reward shaping.

Each run has a30-minute deadline and is audited before the next. Native jobs
are serialized under the host guard on RTX5080 with>=2GiB sampled Vulkan budget
headroom. Retain standalone allocation warnings; failures stop for review.
No NVML polling, recovery, blind retry or automatic budget extension.

Retain every episode/tail and audit rewards, counters, finite checkpoints and
guards. Measure real reward discovery, online score/action/time curves, bonus,
advantages, entropy, update cost and CPU-replayed height/action coverage. Labels
are diagnostics only; streams/episodes are not independent learner replicates.

If all seeds remain at zero, skip frozen evaluation and review action contrast/
coverage before another change. If any discovers reward, frozen-evaluate all
three candidates and three retained extrinsic-only controls: first three natural
episodes/stream (24 total), cap200k actions/30min, sampled policy, zero updates,
exact tensor equality and full stream0 videos. Reward discovery alone does not
finish the goal: verify repeated unassisted crossings and frozen improvement
over untrained controls before a competence claim.

Qualified runtime remains Meganeura592a2f5a/Bladee349cddf; no precision workaround.
Raw artifacts: `runs/freeway-embedding-disagreement-20261005.Vml8ghWH`.

Retained qualification failure: the first full-graph detachment test assumed
every dead parameter lacked a gradient buffer. Meganeura emits scalar-zero
sentinels; an unused scalar continuation bias can therefore have a zero buffer.
Check the autodiff zero sentinel for every non-ensemble parameter and read back
any scalar gradient as exactly zero. The target/model code is unchanged by this
test correction. No training or host recovery preceded review.
