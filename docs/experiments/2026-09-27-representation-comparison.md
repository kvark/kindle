# Phase 2: original representation comparison (large matrix stopped)

**September 30 user-approved change:** finish active upstream Seaquest seed2017
and cancel the 21 unstarted 12M entries. Preserve this protocol and every result
as a partial study, not a completed benchmark. The dispatcher is held; the
independent guard finishes its worker before the drain service stops the queue.
See `runs/representation-learning-20260928.kjidlR/queue-cancellation.json`.

Phase 2 remains open, but now follows the
[replication-first small-screen plan](../strategy_reset_plan.md#2b-replication-first-then-qualify-the-cheap-learning-screen).
Offline probes below are complete. The learning section records the **historical
recipe**, not authorization to resume the cancelled queue. Numerical/learning
replication, a cheap JEPA comparison and an architecture decision are still required.

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

The [complete offline results](../results/2026-09-27-representation-probes.md) cover
ridge and corrected MLP fits for all five controls, every target/variant and both held-out trajectories. Trained
versus initial Tiny remains mixed. Large improves primary position/motion
decoding substantially; the 173k CNN roughly matches Tiny's position decoding
at much lower capacity. Gameplay remains required. Large extraction
took 4,976.55s versus 678.06/682.03s for trained/initial Tiny; this includes
offline projection/compression, not isolated actor inference. CNN reconstruction
completed 16,384 updates in 47.17s, validation MSE 1.68956 -> .00614619,
selecting the final checkpoint `ff05e413`. Different training corpora and
temporal inputs still prevent attributing all differences to architecture.

The fixed ridge contrasts also argue against changing several factors at once:
Large's history increases mean motion R² by .364 on Pong and .430 on Breakout;
trained Tiny's corresponding changes are -.042 and +.073. These are backward
velocity labels, not successful future forecasts. JL16 space-to-depth reduces
trained Tiny's mean position R² by .119/.226/.318 on Pong/Breakout/Seaquest;
PCA has no consistent advantage. Native versus RGB64-upscaled inputs is mixed,
not proof of a universal resolution benefit. Tiny's prior RGB64 corpus is a
distribution difference, not a reason to restore destructive preprocessing.
Keep native/JL64/mean unchanged in the declared learning matrix. Seaquest
motion remains weak across all controls, with the retained visibility limits.

The first MLP sweep stopped after three recorded variants when another device
construction returned `no supported device found`. No kernel GPU fault was
recorded; the cause is not established. Preserve `offline-queue/tiny-mlp` and
its partial `tiny-mlp/progress.jsonl`. The continuation uses one device per
sweep, with fresh sessions/weights/optimizer state for each head. The native
independent value/gradient test plus 33 resets (including shape changes) passes
on RTX5080, preserving initial weights and exact first-update moments/weights;
897 Python tests, 97 Rust CPU tests, formatting and release Clippy pass.
`offline-queue-v2` contains only uncompleted probes/smokes and a fresh
`tiny-mlp-v2` output, not a restart of the stopped writer. This is GPU device
reuse, not a CPU learner, changed fit budget or established historical root cause.

All five initial MLP sweeps then complete, but their normalization is superseded:
F32 training-statistic accumulation reports false nonzero variance for 7,498 /
3,136 / 4,104 truly constant raw-feature columns in Pong / Breakout / Seaquest.
This amplifies held-out variation and invalidates the raw-pixel positive control.
Use F64 training-only means/standard deviations for **every** MLP, then cast
normalized inputs to F32 for native learning. This is a numerical correction,
not a test-selected scale floor, changed fit budget or CPU learner. Preserve the
original fits; the report reader refuses their missing corrected-normalization
marker. Ridge already used F64 statistics and is unaffected. Corrected MLP
outputs use fresh `*-mlp-f64` directories; all five now complete. Even corrected,
the fixed-budget MLP generalizes poorly, especially on raw pixels. The ridge
positive control decodes positions from the same inputs; MLP failure is not
evidence that pixels lack the target information. Do not tune again on test scores.
The [normalization report](../results/2026-09-27-probe-normalization.md) records
the training-only diagnosis and all five superseded result identities.

The first matched upstream smoke stops before any action because the collector
indexed the outer carry tuple as streams. The pinned JAX wrapper instead returns
dict/tuple structure with per-stream **list leaves**. Gather/commit those leaves
without reading device arrays; a CPU fixture checks reordered partial resets and
untouched streams. Preserve the failed `upstream-smoke`; `upstream-smoke-v2`
is a fresh corrected invocation. No model, replay budget or optimizer changes.
The corrected carry/normalization/queue checks pass in the full 901-test Python
suite. Corrected MLP work waits for the smoke controller's positive completion,
not just a native exit, and remains serialized with all learning work.

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

## Historical 12M learning protocol — do not resume

Pong, Breakout and Seaquest; learner seeds **1009/2017/3019** per variant. Include upstream
DreamerV3 12M, Kindle Large, pretrained Tiny and initial Tiny, plus a jointly
learned CNN: its competitive position probes justify this conditional arm.
The online CNN starts fresh, not from the offline reconstruction checkpoint.
It uses one explicitly declared native-to-RGB64 GPU resize, with no upscale
back to 224. This is a learned-RGB research control, not a downgrade of the
native-detail JEPA path. Replay must store pixels and re-encode them with current
weights; stale frozen feature replay is not joint learning. Record the final CNN
recipe and pass its integration smoke before launching that arm.

The implemented native RGB control uses a kernel8/stride8 RGB stem, two padded
3x3 spatial convolutions at 8x8, group normalization (eight groups, epsilon1e-4)
and SiLU after each convolution, then 2x2 max pooling. Width is four times the
preset vision depth: 64 for 12M, yielding 4x4x64 posterior input. The encoder
has 86,400 parameters. The decoder is a normalized 256-unit SiLU hidden layer
and a dense 12,288-value RGB output. Pixel reconstruction uses summed squared
error, scale1; the frozen-feature future-prediction head is disabled. RSSM,
actor/critic, replay ratio, update scheduling and optimizer stay unchanged.
The resize is bilinear, no antialiasing, CHW /255-0.5; upstream uses Pillow's
resize and its own CNN encoder/decoder. This is a whole-package native learned
baseline, not an exact upstream encoder/decoder port or a pretraining ablation
of the offline 173k reconstruction CNN. The common 12M RSSM preset does not
equalize total trainable capacity: upstream has 10,498,772 parameters, native
frozen-feature arms 10,281,233 plus their frozen encoder, and joint RGB CNN
12,906,641. Include this difference when interpreting learning and cost.
GPU tests cover independent scalar
values/gradients, CPU/resident preprocessing, unchanged pixel replay across
encoder updates, per-stream resets and optimizer-preserving checkpoint restore.
Those tests and the new arm's integration smoke now pass. The first replay
fixture compared different windows because fresh-arrival samples take priority
over seeded RNG draws; draining that queue fixes the test without changing the
learner. Preserve the original failed fixture. All nine CNN first-moment
tensors are nonzero after the smoke; checkpoints are finite and restore exactly.
Fix the budget before the
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

Use the same 3,072-action reporting interval in both runners, plus the exact
final budget. Scores are the last 50 completed episodes per learner seed (all
if fewer), including artificial cutoffs; every raw episode remains in the
published JSON. Aggregate with equal learner-seed weight and 10,000 percentile
bootstrap resamples (RNG seed0). Interpolate wall-time curves only inside common
measured support; do not extrapolate slow runs or fill absent early episodes
with zero. Initial policy/encoder compilation and final checkpoint writes count
in run time; report agent construction separately. The shared reader is
`python/examples/summarize_representation_learning.py`.
The first full declaration, upstream Pong seed1009, stopped **before spawning
its GPU worker** in `/mnt/data/kindle-representation-learning-20260927.I1MQPA`.
Its NAS-backed kernel-log capture timed out and was not reaped within the
guard's deadline; that capture PID is now absent. No native failure or new
kernel GPU fault was recorded. Preserve the failed attempt. Use local storage
for live guards/logs, and archive completed artifacts separately; no guard
timeout or acceptance threshold is relaxed. The local matrix lives in
`runs/representation-learning-20260927.bTE8iK`. All five Pong seed1009 arms
complete 200,004 actual actions / 49,651 updates, with zero debt, finite
checkpoints and passing/reaped guards. The
[rolling learning report](../results/2026-09-27-representation-learning.md)
holds scores, timings, memory audits, all episodes and unfinished tails.
Upstream leads this first seed; Large is the strongest native arm but slower,
and Tiny shows no pretraining benefit. This is not frozen competence or a
multi-seed frontend decision. Native/upstream architectural differences prevent
attributing the gap solely to the frozen encoder or reconstruction choice.

`learning-remainder.json` declares the other 44 entries behind positive
completion of both the first native guard and its controller. The complete
matrix is 45 runs / 9,000,180 actions, with unchanged budgets. Within each
learner seed, game order is Pong/Seaquest/Breakout; method order rotates by
game and seed. Each direct worker has an eight-hour limit, with no overlap,
retry or successor after failure. This is a multi-day comparison, not a fast
screen. All native full runs use the same RGB-capable package (native SHA256
`da7e9cd03d9cda5a1d1d2569d4745ee4ea965d792a540d8822024984ece0b7d4`);
adding the RGB arm does not change frozen-feature learning arithmetic.
Keep runners/package/configurations fixed through the matrix; do not compile
locally during timing. Guarded short-test memory headroom does not guarantee
headroom at full replay capacity, so every run retains its native memory checks.

The [September 28 external interruption](../results/2026-09-28-learning-interruption.md)
preserves eight completed results. A new-boot declaration restarts interrupted
Large Seaquest seed1009 from scratch, then continues the same 36 unstarted
entries. The package, protocol and budgets stay fixed; only output paths and
boot identity change. Preserve the incomplete attempt and disclose its extra
compute instead of counting it as a completed seed or an equivalent resume.

The decision rule is unchanged: frozen LeVJEPA must beat the random/learned
baseline on both probes and learning curves to justify its 2D cost. Otherwise
remove it as the 2D default and retain it as a 3D hypothesis. Any useful
pooling/projection/input change is adopted separately after a learning check.

Raw preparation: `runs/representation-probes-20260927.POCnif`. Completed results
will be committed as self-contained JSON + Markdown under `docs/results/`.
No multi-seed gameplay representation advantage has yet been established.

The [five integration smokes](../results/2026-09-27-matched-integration.md)
now pass: each executes 6,144 actions / 1,186 updates, starts updates at 1,404,
and finishes with zero debt. Final native checkpoints are finite with nonzero
world/behavior moments; upstream RSSM/encoder/actor parameters change. Tiny,
initial Tiny, Large, upstream and the joint CNN run at 25.64 / 25.60 / 16.94 /
25.01 / 23.17 actions/s
over these short, warmup-containing windows. These are not steady-state rates
or learning results. All direct-child guards pass; no kernel GPU fault. Keep
the earlier failed carry-layout invocation separate from the passing correction.
