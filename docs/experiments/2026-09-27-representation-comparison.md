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

Tiny's prior experience remains 250k random-play RGB64 frames from Boxing,
Pong, Freeway, Breakout and Qbert (45k train +5k validation each). Large's
[model card](https://huggingface.co/galilai-group/LeVJEPA-VideoMix-Large/blob/e831a0347737fcaa660b39c57d41c109de399845/README.md)
reports 1,806,869 VideoMix clips from Kinetics-710, Something-Something v2,
Walking Tours and PE-Video. Large versus Tiny changes capacity and corpus;
only trained versus initial Tiny isolates that pretraining intervention.

## Learning comparison — still required

Pong, Breakout and Seaquest; three learner seeds per variant. Include upstream
DreamerV3 12M, Kindle Large, pretrained Tiny and initial Tiny, plus a jointly
learned CNN if the offline comparison supports it. Set the equal actual-action
budget before launching the campaign (200k is the strategy's reference).
First align reset/action/update accounting, replay capacity, sticky actions,
time-limit targets and score/time curves. A stock-upstream sanity run is not
this comparison. Keep architecture/corpus/input differences explicit; do not
attribute a whole-package difference solely to JEPA.

The decision rule is unchanged: frozen LeVJEPA must beat the random/learned
baseline on both probes and learning curves to justify its 2D cost. Otherwise
remove it as the 2D default and retain it as a 3D hypothesis. Any useful
pooling/projection/input change is adopted separately after a learning check.

Raw preparation: `runs/representation-probes-20260927.POCnif`. Completed results
will be committed as self-contained JSON + Markdown under `docs/results/`.
No learning or representation advantage has yet been measured in this phase.
