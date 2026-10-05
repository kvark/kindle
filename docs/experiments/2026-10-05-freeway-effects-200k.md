# Freeway CDP: fixed200k action-effects follow-up

The [32k frozen comparison](../results/2026-10-05-freeway-action-effects-frozen.md)
finds14/72 crossings versus1/72 extrinsic-trained and0/72 untrained. All three
candidate models retain some improvement, but most episodes score zero and
seed3019 regressed during training. This supports one longer test, not mastery.

## One change: interaction budget

Three **fresh** learners1009/2017/3019,200,000 actual aggregate actions and49,939
updates each:600,000 new training actions/149,817 updates total. Do not resume
32k checkpoints without their replay/RNG/live belief. No automatic extension.

Keep the qualified action-effects bonus and native `f4b6a5c7` unchanged:
Freeway/full18/sticky.25/repeat4/no reset no-ops, Size1M/N8/B8/T16/H15/R32,
microbatch8/replay100000, native-detail input with one GPU RGB64 resize.
CDP cosine500, encoder6e-6/dynamics4e-4/base4e-5, warmup1000, AGC.3,
ac_grads=false, disagreement coefficient1. No normalization, new representation,
action override, game label, reward shaping or pretraining. CPU114/Python1,161,
strict Clippy, independent GPU qualification and CI303/304 cover unchanged code;
do not rebuild or repeat qualification solely for the budget change.

Serialize guarded native processes in one persistent service, Restart=no,
KillMode=control-group, four-hour service deadline,45min per training seed.
Expected training~85min plus evaluation/replay audits. Audit each seed before
the next. Require the expected RTX5080 and>=2GiB sampled Vulkan estimated
headroom. Retain standalone allocation warnings; API/numerical/hard faults and
deadlines stop the job for review. No NVML polling, recovery or blind retry.

## Evaluation and decision

Any training reward: evaluate **all three** final policies and actual
initial-weight controls, including any unsuccessful seed. Reuse saved zero-action,
zero-update initial checkpoints with the identical configuration, but perform
new evaluations. New held-out base2,000,000,000 plus learner seed and per-stream
offset1,000,003; do not reuse the32k frozen environment seeds. Sampled policy,
first3 natural episodes/stream (24/model), cap200k actions/30min each, zero updates,
exact saved tensor equality, full CPU replay and unfiltered stream0 videos.
Retain every completed/excess episode and unfinished tail. An incomplete cohort
is incomplete, not a zero or success. All-zero training skips frozen evaluation
and stops for diagnosis.

Report full extrinsic-return versus actions/time curves, first reward, seed
variation, entropy, intrinsic advantages, update cost and actual emulator-frame
throughput. Three learner seeds are replicates, not streams/episodes.
The retained32k extrinsic cohort is **not a200k matched control**; this follow-up
tests whether the selected mechanism can develop useful play, not its matched-
budget superiority. Do not rerun the stopped representation matrix.

Review whether improvement grows and persists before choosing further work.
The original mastery gate remains mean>=25, >=90% rounds reaching25 crossings,
>=20 natural rounds and no cutoffs. Passing plumbing, finding a few rewards or
beating untrained controls does not close the goal.

Raw work: `runs/freeway-effects-200k-20261005.cK3FhKKq`.
