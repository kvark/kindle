# Separate Qbert replay-value1 learner and evaluation-cohort differences

Prepare four frozen crossed assessments within the existing five-game goal.
Development14009 scores16525.000 greedy on cohort4000014009; fresh16001 scores
14913.542 on4000016001. Both clear24/24 first pyramids, but the latter misses
the fixed15k gate. The [saved-log phase diagnosis](../results/2026-11-02-cdp-qbert-replay-value1-phases.md)
places the entire score gap after an identical550-point prefix and finds
substantial later-play training exposure in both learners. Learner and evaluation
cohort still change together. Cross the retained checkpoints before attributing
this latest recipe's difference solely to learning.

This is **unselected CPU preparation**. No executable queue, GPU allocation or
service is created. The original Breakout replay-value1 learner and its four
observers must complete, all original source pins must still match, and the
host must be idle before separate selection. Do not overlap, interrupt or
restart Breakout or repeat any completed Qbert assessment.

| New greedy arm | Checkpoint | Evaluation seed |
| --- | --- | --- |
| Learner14009 | Fixed200k final |4000016001|
| Learner14009 | Actual initial |4000016001|
| Learner16001 | Fixed200k final |4000014009|
| Learner16001 | Actual initial |4000014009|

Reuse the four completed diagonal greedy arms without repeating GPU work.
Report complete2×2 final and actual-initial matrices, initial-subtracted means
and per-lane means. Keep15k/90%-first-pyramid gates visible for every final cell.
The original fresh confirmation remains a failed gate. No favorable-cell
selection, relabeling or independent learner confirmation follows from a
different evaluation cohort.

Keep each checkpoint's own config, sampled categorical posteriors, centeredCDP/
12M64/Auto/N8/B16/T16/fullBPTT/H5/R24/base-encoder-behavior4e-5/dynamics1e-4/
replay-value1/actor-unimix0/expert100. Training used greedy collection and stays
at200k actions/18715 updates,25m36.935s and24m45.754s. CLI seed changes environment
and lane seeds on restore; this does not isolate weights from checkpoint RNG
config or establish broad reliability.

Reuse native447fe50/SHA35ad0db, exact runnere304af89 and matching
`/x/Temp/kindle-qbert-color-masks-20261101/runtime`. Guidance and labels are off.
Full18/sticky.25/repeat4/noops0/native pooled RGB/one GPU resize remain fixed.
No training, model/loss change, native build or backend work occurs.

Each new arm selects the firstthree natural episodes per eight lanes=24, under
600k actions/300s. Requirezero updates/truncations,357 finite invariant tensors,
source checkpoint hashes, full recorded-outcome replays/videos and host guards.
Retain excess episodes and partial tails. Ordinary frozen logs lack live pixel
hashes; replay cannot establish unavailable live-image equality. Replay cap180s
per arm; main2400s, collection2700s. No cap increase or repeated stage.

CPU qualification exercises allfour actual CLI paths and header guards, checks
allfour source checkpoints against the qualified RGB64 schema, reproduces the
four retained diagonal audits/scores and tests original-invocation checks.
No native agent or emulator is constructed. Preserve all338 active Breakout pins
and originalPID887827/invocation4b67c053d030409f8734e6bfbdab6707 before/after.
Record CPU co-host times without correcting training time. Pin all controller,
runtime, runner, reference and protocol sources before guarded selection.

Root `/x/Temp/kindle-qbert-replay-value1-cross-cohort-20261102`. No learner,
extension, retry, mode sweep, intermediate-checkpoint selection or automatic
successor. Earlier RGB48/RGB64 crossed assessments and their failures/results
remain closed. The five-game goal remains unfinished.
