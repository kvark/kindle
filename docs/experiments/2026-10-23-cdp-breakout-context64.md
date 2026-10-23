# Breakout: longer training sequences at fixed sampled-position budget

**Prepared, not running.** Finish the active Freeway replication first. Run a
disposable construction/train/frozen-restore smoke, review it, then separately
launch the learning comparison. Backend optimization remains parked.

## Learning hypothesis

The2M CDP actor scores98.21; another6M warm-start actions produce85.17. Its
encoder and recurrent states retain ball/paddle information, but15-step ball
forecasts remain imprecise and do not improve with the extension. Test whether
longer gradient paths through the recurrent world model improve learning.

Change the existing12M recipe from **B8/T16 to B2/T64**, with full-sequence
backpropagation and full-batch world updates. Both have128 sampled positions
per update, replay ratio32 and the same post-warmup update frequency. Eight
live environments and batched action inference remain unchanged. Keep all
losses, precision, optimizer rates, imagination length15, model capacity,
reward/exploration settings, replay capacity100k and seed1009 fixed.

This is not a pure BPTT-only ablation: longer sequences also reduce independent
sequences per batch and change temporal correlation of centered CDP targets.
Stored recurrent state already carries history atT16; this does not suddenly
give the actor memory. It lengthens training gradient paths and sampled chunks.
No decoder, larger encoder, privileged labels, game hints or new kernels.

## Fixed budgets and comparison

1. Disposable smoke:4096 actions, expected867 updates, then1024 frozen actions.
   Compare all346 initial tensors with the retained B8/T16 initial checkpoint;
   verify finite reports, independent action/credit accounting, final checkpoint
   and all346 frozen tensors. Per-training cap900s. No speedup acceptance gate.
2. After reviewing smoke: fresh2M-action life, not smoke continuation. Save fixed
   500k/final/actual-initial checkpoints and evaluate24 natural rounds each with
   the existing sampled policy and evaluation seed4000001009. Compare retained
   B8/T16 scores58.83/98.21/1.42. No best-checkpoint or greedy-mode selection.
3. Report scores, two-wall864-point successes, online curve, actor updates and
   wall time together. Published381.81 and reliability remain required. One
   seed is hypothesis screening, not supremacy or a matched-time advantage.

With eight streams,128 eligible context-plus64-step sequences first become
available at632 actions, versus248 in the control. Expected500k/2M update
counts are124,843/499,843:96 fewer than the control due only to warmup. Preserve
this difference rather than silently adding actions or optimizer steps.

Use the unchanged qualified c8aeba28 native binary and runner. Existing CLI
sets backpropagation length equal to batch length and supports both shapes.
CPU forwarding tests cover B8/T16 and B2/T64, including centered CDP and the
intrinsic channel. Smoke tests execution/finiteness/restore, not a new
independent full-gradient numerical oracle.

Training cap8h, each frozen arm600k actions/1800s, CPU audit3600s. No automatic
extension. GPU service12GiB/no swap, CPU audits2GiB/one CPU. Host-only monitoring,
known allocation warnings recorded; real faults and numerical failures stop.
The scheduler and original control determine counters, not successful scores.
Retain failed captures and observer failures; CPU repairs repeat no GPU cohort.

Artifacts: `runs/cdp-breakout-context64-20261023.5OA15Iew`.
Controller: `run.py smoke` followed, only after review, by `run.py train`.
The controller validates exact config differences and initial tensors, frozen
exports, complete replay/video and fixed evaluation cohorts. No worker from
this declaration has started yet; Freeway owns the GPU.
