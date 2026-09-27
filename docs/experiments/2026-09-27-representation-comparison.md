# Phase 2: representation comparison (in progress)

The objective remains all of strategy Phase 2: offline probes plus a matched
three-seed learning comparison, then an architecture decision. A passing probe
or implemented harness alone does not close this phase.

## Offline protocol

Pong, Breakout and **Seaquest, absent from Tiny's pretraining corpus**. Record
eight independent random-action trajectories per game: four training seeds,
two validation, two test. Each contributes 256 complete 16-arrival clips.
Keep whole trajectories in one split; never split adjacent frames randomly.
Use full18 actions, repeat4, sticky .25, no reset no-ops or reward/action aid.
The native 210x160 RGB is max-pooled over the final two emulator frames;
RAM is an offline label source, never a policy/replay input. Track discarded
terminal tails and actual emulator frames. Labels describe the final arrival;
velocities are backward differences per executed emulator frame, not forecasts.
Absent objects and respawn jumps have explicit per-target masks and counts.

RAM coordinate maps follow [OCAtari's pinned source](https://github.com/k4ntz/OC_Atari/tree/99c874675df6b76a33a80b57776c123fbcd051af/ocatari/ram).
Check their alignment independently against sprite colors before interpreting
probe scores. Seaquest uses fixed enemy-lane slots, not a changing nearest enemy.
Constant targets have undefined R², not fabricated zero/perfect scores.
The [completed corpus audit](../results/2026-09-27-probe-corpus.md) finds
unmatched Seaquest sprites. Inspecting one training trajectory shows most
player misses have no expected-color pixels at all and death/blink RAM105
values 15–22. Do not silently redefine or discard those targets. Report both
all-valid-RAM and secondary visibly aligned R²; a velocity is visibly aligned
only if both endpoint sprites pass. Visibility never selects training examples,
regularization, stopping or a model. Preserve the original dataset unchanged.

Compare released Large, pretrained Tiny `7fe9b252`, its own initial Tiny
`7bc344f3`, and a small reconstruction-trained CNN. For every frozen encoder:

- Native input versus explicitly labelled RGB64-upscaled ablation. The latter
  is diagnostic only; no return to that path in the production actor.
- JL64 + 2x2 mean versus JL16 + space-to-depth, both exactly 7x7x64. Reuse the
  first 16 random directions with JL16 scaling, rather than change both draws
  and packing. Repeat the contrast with 64/16 PCA components.
- Fit PCA **once per encoder** from <=32,768 native phase-15 TRAINING tokens,
  balanced across the three games. No labels, validation or test data in PCA.
  Freeze and record its actual mean/axes for all input/phase/pooling variants.
- Compare phase15 after all sixteen arrivals with phase0 of the **same final
  image**, reset before that image. This measures useful history plus chunk
  phase; it is not a claim of pure positional-phase invariance.

All diagnostic encoding runs batched on native Meganeura/Blade. Readbacks and
CPU projection/fitting are offline analysis, not a new acting pipeline.
Check exported token order against the production JL/mean-pooled result.
Fit linear ridge and a small MLP; choose regularization/stopping only on
validation trajectories. Report every target's held-out R², error and valid
count, with position and motion separate, plus raw-pixel/constant controls.
Do not use test scores to tune the model or select a favorable subset.
Ridge penalties are 1e-4 through 100 in powers of ten, independently selected
per target by validation MSE. The MLP is 128 ReLU units, batch64, 512 Adam
updates (lr1e-3, .9/.999/1e-8, weight penalty1e-4/N), with head seeds
1009/2017/3019. Select among every32-update checkpoints by validation
target-standardized MSE. Both fit only training trajectories, with training-only
feature/target normalization; neither refits on validation or selects on test.
Head seeds describe probe-fit variability, not independent RL experiments.

The reconstruction control is a **172,864-parameter stateless patch CNN**:
stride16/kernel16 RGB stem (64 channels), two padded 3x3 convolutions at 14x14,
and a patchwise linear RGB decoder. Native GPU letterboxing preserves the
input detail; this is not the exact DreamerV3 encoder. Train on arrivals
3/7/11/15 of TRAIN clips (12,288 frames across all three games), with no RAM,
actions or rewards. Fixed 16,384 updates, batch16, Adam lr3e-4/.9/.999/1e-8;
select by validation reconstruction MSE every512 updates, including step0.
Its extra in-corpus Seaquest reconstruction experience and lack of temporal
history must remain explicit. Phase0/15 features are identical by construction.

Raw RGB56 one-/two-frame probes are label-decodability controls, not production
inputs or size-matched encoders: they expose 9,408/18,816 values versus 3,136.
The two-frame input orders previous then current arrivals; it receives neither
actions nor RAM. Constant predictors use training-target means.

Tiny's prior experience remains 250k random-play RGB64 frames from Boxing,
Pong, Freeway, Breakout and Qbert (45k train +5k validation each). Large's
[model card](https://huggingface.co/galilai-group/LeVJEPA-VideoMix-Large/blob/e831a0347737fcaa660b39c57d41c109de399845/README.md)
reports 1,806,869 VideoMix clips from Kinetics-710, Something-Something v2,
Walking Tours and PE-Video. Large versus Tiny changes capacity and corpus;
only trained versus initial Tiny isolates that pretraining intervention.

## Learning comparison — still required

Pong, Breakout and Seaquest; learner seeds **1009/2017/3019** per variant. Include upstream
DreamerV3 12M, Kindle Large, pretrained Tiny and initial Tiny, plus a jointly
learned CNN if the offline comparison supports it. Fix the budget before the
first RL launch: **200,004 actual actions** per run (the nearest full N6 batch
above the 200k reference), B16/T64/context1/full BPTT/H15/R256, F32, lr4e-5,
warmup1000, AGC .3. Six independent environments use seeds
`(learner_seed + stream*1,000,003) mod 2^32`, full18/repeat4/sticky .25,
no reset no-ops or action/reward aids, 100,000-frame artificial cutoffs that
bootstrap. Resets add replay context but earn neither action budget nor update
credit. Discard prefill debt; first eligible replay batch earns one update,
then one update per four actual actions. Record the actual warmup boundary and
fractional debt; episode-dependent reset counts can move that boundary.

The Phase 2 upstream mode is `run_upstream_control.py --matched-actions`.
It retains the pinned agent, losses, optimizer and native policy-sync delay;
only collection/accounting uses the shared Kindle Gym/ALE wrapper. Both
interpreters must agree on RGB/reward/RAM/reset/cutoff traces before a GPU
smoke. Upstream uses its learned RGB64 encoder/decoder; Kindle uses native
RGB -> frozen features and its prediction-only latent objective (.25 scale).
These are whole packages, not a claim to isolate the world-loss choice.

Replay retains 100,000 arrivals: upstream's sequence-start capacity is
`100000 - 6*64 = 99616`, since each stream also retains 64 context/tail rows.
The capacity and warmup match; sampling RNG, chunk storage and eviction
implementations do not become identical. A stock-upstream sanity run is not
this comparison. Keep input, corpus and architecture differences explicit.
Report every completed online episode and unfinished tail against actions and
elapsed time, with final checkpoints and three-seed uncertainty. No development
mastery gates or favorable-episode filtering. A bounded shared-protocol smoke
precedes the campaign, but is not one of its learner seeds.

The decision rule is unchanged: frozen LeVJEPA must beat the random/learned
baseline on both probes and learning curves to justify its 2D cost. Otherwise
remove it as the 2D default and retain it as a 3D hypothesis. Any useful
pooling/projection/input change is adopted separately after a learning check.

Raw preparation: `runs/representation-probes-20260927.POCnif`. Completed results
will be committed as self-contained JSON + Markdown under `docs/results/`.
No learning or representation advantage has yet been measured in this phase.
