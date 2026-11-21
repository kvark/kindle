# Breakout retained-recipe confirmation on one fresh learner seed

The strongest retained 200k recipe was selected on learner1009 and repeatedly
assessed on cohort4000001009. Its later 400k point has an identical training
prefix, so it is not independent confirmation. Many subsequent interventions
regress against this selected baseline. Before changing another loss or teacher,
test whether its useful learning survives one new initialization and experience
stream. This is one fixed confirmation, not a seed search or a new recipe.

Select fresh learner **22003**, 200,000 aggregate actions /12,477 updates, and
previously unused evaluation cohort **4000022003**. Change only the seed in the
effective learning config. Training environment seeds, native sampling and
guidance selection consequently change together. Transfer no checkpoint, replay,
belief, game or RNG state. No old learner or completed evaluation is restarted.

Reuse exact retained native447fe5013b6442c477fc7bfef7eab4122082bc54/SHA256
`35ad0db42fc7ed5dbaf6dacddff613fdc04d83ae6e0f38548532f6ab2c7812a2`,
runnercf4ea238/SHA256
`cf4ea2389092d4ccd7be55a76d0f24d333796a81e154e34a95829c7e61d44ab3`,
and `/x/Temp/kindle-breakout-corrective-20261028/runtime`. No build, overlay,
new native feature or repeated bounded training check is required.

Keep centered CDP500/12M64/Auto/N8/B16/T16/full BPTT/micro16/context1/H15/R16,
replay100k, base/encoder4e-5, dynamics4e-4, behavior8e-5, policy1/critic1/replay-
value1, expert100/unit weights/world gradients/all-state labels, disagreement1,
actor/latent mixing.01, sampled collection/posteriors and20% episode guidance
capped1920 actions. Full18/sticky.25/repeat4/noops0, native max-pooled RGB and
one GPU64 resize. All newer optional objectives and architecture changes remain
off. The original teacher and reward are unchanged.

Training cap1800s. Save actual-initial, read-only100k/6,227 recovery and final
200k/12,477 checkpoints. Require357 finite arrays, exact config/schema/counters,
first update376, both weighted losses and changed CNN/reward/continuation/actor/
critic parameters. New seed initialization must differ in random encoder/actor
weights; equality to learner1009 is neither required nor claimed. Full training
pixel/action/label/override/reward/frame/boundary replay and host guards must pass.

Evaluate final and actual-initial checkpoints in sampled and greedy modes on
cohort4000022003: first three natural episodes per eight lanes,24 per arm,
guidance/labels/learning off,357 invariant tensors, zero selected truncations.
Per-arm caps600k aggregate actions,100k frames/episode and300s wall. Save complete
recorded-action/outcome/frame/boundary replays and videos. Frozen logs omit live
pixel hashes; do not claim unrecorded image equality.

Also evaluate the retained learner1009 final and actual-initial checkpoints in
both modes on this same new cohort, four additional frozen arms with identical
caps. The original training is reused from disk, with its measured1287.910s
cost. This cross-cohort control separates evaluation-cohort movement from the
combined new training-seed effect. Stochastic policy draws differ between
learner seeds. No equality of new and old trajectories is assumed.

Greedy remains primary. Report own-initial gains and gain per training minute
for both learners, raw final scores, paired differences and10,000-resample
eight-lane95% intervals with seed0. Report the original1009/cohort4000001009
point as historical context. Positive fresh own-initial intervals are evidence
of learning on that seed, not task completion or broad reliability. Do not
change settings, choose intermediate checkpoints, enlarge caps or launch another
seed based on observed outcomes. There is no automatic promotion of a new best
seed. The original task gate remains mean864 and90% two-wall; five-game scope
and all other unfinished requirements remain intact.

Root `/x/Temp/kindle-breakout-seed22003-20261106/full`; service
`kindle-breakout-seed22003-200k-20261106`. Before launch qualify exact commands
with all native/emulator construction mocked, pin sources/native/runner/teacher/
ALE/ROM/reference files, verify current host identity and no conflicting GPU
work, and check disk space. Use the existing AMD host guard; stop on actual
faults or nonfinite values. No driver reset/reload/reboot or automatic retry.

One learner and one GPU phase at a time. Main/terminal caps7200/7500s; training
replay900s and frozen replay500s. Independently verify100k recovery during
training and original process identities/artifact hashes at closure. Include
CPU observer cost in measured training wall time. Preserve every failure; a
separate CPU repair cannot relabel a failed original stage. Retain curves,
exposure, source identities, all useful checkpoints and compact reports; large
artifacts stay outside Git. No extension, second learner, broad seed campaign or
automatic successor is selected.
