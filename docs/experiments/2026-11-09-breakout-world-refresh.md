# Breakout: one-time image-history and latent refresh

The [closed local H0/RGB1 trial](../results/2026-11-09-breakout-world-local-h0.md)
reads observed/one-step balls accurately but fails sustained fitting and
held-out forecasts. The earlier [state/history diagnostic](../results/2026-11-08-breakout-world-state-history.md)
substituted privileged blocks at readout only, without feeding a hybrid forward.
It cannot measure whether a one-time correction persists in a resumed forecast.
Select that distinct frozen intervention before another full learner.
Root `/mnt/data/kindle-breakout-world-refresh-20261109`.

Use only local H0/RGB1 final16,384, v20, size1M/RGB64/B8,
603,915 weights/80 logical/240 tensors, Auto. Production runtime/inference remain
unchanged. Add test-only native orchestration using existing production
observe/transition/decoder/head graphs, sampling and history helpers. Reuse
unchanged math qualification; no wheel, backend primitive, optimizer, actor,
critic, new loss, emulator action or collection.

Take first16 dense fitting and first16 dense development origins, in their
original B8 order, four seeds83001/83003/83009/83023. Join all source images,
actions, masks and metadata to retained exports. Reused development families,
not untouched examples or independent confirmation. Keep original native pooled
RGB64 from one resize. RAM/positions/lives/outcomes remain CPU metadata only.

Observe the original16-image prefix. Immediately before future action4, branch
into four trajectories: unchanged prior, image-history refresh, latent h/z
refresh, both. A refresh may use true arrivals only through future action3.
The history refresh supplies those two latest true images; the latent refresh
supplies an independently image-informed posterior through that same cutoff.
No current/future image at action4 or later enters a branch. Then run all branches
causally through32 with no further correction, using the same per-step random
numbers and submitted actions. Hybrid outputs feed their own image histories;
latent refresh also feeds later recurrent state. This differs from the completed
readout-only swaps and uses neither old H16/H32 model.

Retain every step's RGB, latent features and reward/continuation probabilities.
Require canonical baseline exact against production prediction at all32 steps;
first-draw endpoint/control arrays match retained dense forecasts. All four
branches match before injection. Image-only refresh must preserve all latent
states and heads, since this model has no image-to-latent feedback. Verify prefix
and refresh state/history cutoffs, finite outputs,240 unchanged weights/moments,
optimizer/model age16,384, and no training. Save outputs before derived assertions
when feasible; preserve original failure logs and never relabel a failed run.

Before GPU execution, pin source/executable/config/data and review CPU checks of
refresh timing and factors. Separately declare one bounded900s host-guarded
native run. Check current boot/driver/process identity; stop on real faults or
nonfinite states, with no reset/reload/recovery/NVML. No full learner is selected.

CPU analysis must keep fitting/development separate and all four modes/draws.
Report ball/paddle/brick readout metrics at every supported step, first failure
and recovery persistence after the refresh, plus every paired harm/rescue and
reward/ending trace. Natural reset/terminal support masks remain unchanged;
padding or unknown post-terminal arrivals are unscored. No best-origin, draw,
mode or horizon selection. Inspect fixed first cases as illustrations only.

All corrections use privileged future information relative to the original
forecast; hybrid states can be off-manifold. Recovery is a diagnostic, not a
trainable repair, an optimal ceiling, a causal percentage, calibrated event
prediction or a replacement for any original physics/event gate. Full-game
physics, persistent brick memory, late/tunnel/second-wall/above-top coverage,
untouched families and fresh confirmation remain unproven. No unchanged
extension, completed-stage repeat or automatic learner successor.

Preparation:two new CPU checks pass with one native check ignored;strict core
Clippy passes. Original Clippy failure on a fixture Vec allocation remains;the
separate array repair and final CPU/Clippy logs are retained. Original cached
input preparation completed32 source joins and12 input/target/reference files,
then failed on a nonexistent flat reconstruction_scale config key. Original
source/log/outputs remain. Separate metadata-only finish checks the saved
loss_scales.reconstruction key and writes settings/receipt without repeating
source joins. Neither preparation constructs a model or emulator. A broad
formatter also touched five unrelated files;all five were restored byte-exact
to the base before packaging. Only a cfg(test) child-module registration and
the new test-only orchestrator change Rust source.
