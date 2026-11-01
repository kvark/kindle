# Qbert RGB64: separate learner variation from evaluation-cohort variation

**Completed:**10007 stays ahead on both cohorts and passes15k in both;
12011 misses15k in both, all24/24 first pyramids. Original services and all611
inputs pass, withzero updates and no stage repeats. See the
[accepted result](../results/2026-11-01-cdp-qbert-rgb64-cross-cohort.md).

Prepare four frozen assessments within the existing five-game goal. The fixed
RGB64 recipe scores15,710.417 on learner10007/cohort4000010007 and12,678.125
on learner12011/cohort4000012011. Both clear24/24 first pyramids, but only the
first mean passes15,000. Learner and evaluation seeds differ together. Assess
both retained agents on the other cohort before deciding whether more learning
is warranted. The original negative confirmation remains a negative result.

This preparation is CPU-only and unselected. The original Breakout dynamics1e4
learner and its four observers must complete successfully, all source hashes
must still match, and the host must be idle before a separate selection.
No executable queue, GPU allocation or service is created during preparation.

| New greedy arm | Retained checkpoint | Environment seed |
| --- | --- | --- |
| Learner10007 | Fixed200k final | 4000012011 |
| Learner10007 | Actual initial | 4000012011 |
| Learner12011 | Fixed200k final | 4000010007 |
| Learner12011 | Actual initial | 4000010007 |

Reuse the four completed diagonal greedy arms without repeating them. Report
the complete2-by-2 final and actual-initial matrices, initial-subtracted means,
and per-lane means. Keep the unchanged15,000/90%-first-pyramid gate visible for
every final cell. Do not choose whichever cohort passes or count these reused
checkpoints as independent learner confirmation. A passing alternate cohort
does not erase the original confirmation failure.

Keep each checkpoint's own config and sampled categorical posteriors; the CLI
seed changes environment/lane seeds on restore. Thus this compares frozen
agents and environment cohorts without isolating weights from checkpoint RNG
configuration or establishing broad reliability. Both agents use centeredCDP/
12M/RGB64/Auto/N8/B16/T16/fullBPTT/H5/R24/baseencoder4e-5/dynamics4e-4/
behavior unset/actor-unimix0/expert100 and were trained with greedy collection.
Original training times25m09.797s/25m27.366s remain unchanged.

Reuse native447fe50/SHA35ad0db, runneraba4300d and matching runtime
`/x/Temp/kindle-qbert-color-masks-20261101/runtime`. Guidance and labels are
off, full18/sticky.25/repeat4/native pooled RGB/one GPU resize remain fixed.
There is no training, model/loss change, native build or backend work.

Each new arm takes the firstthree natural episodes per eight lanes=24, under
600k aggregate actions/300s. Requirezero updates/truncations,357 finite invariant
tensors, original checkpoint/source hashes, full recorded-outcome replays and
videos, and host guards. Retain excess episodes and partial tails. Ordinary
frozen logs lack live pixel hashes; replay cannot establish unavailable image
equality. Replay cap180s per arm; main service2400s, collection2700s.

CPU qualification must exercise allfour actual CLI commands and header guards,
check allfour source checkpoints against the RGB64 schema, reproduce retained
diagonal audits/scores, and test original-invocation checks. Native agents and
emulators must not be constructed. Preserve all272 active Breakout input hashes
and original PID847728/invocation72a7b4600a4641bb8c93174668127f3f. Record CPU
co-host time without correcting the learner's measured wall time.

Root `/x/Temp/kindle-qbert-rgb64-cross-cohort-20261101`. No learner, extension,
cap increase, retry, action-mode sweep, checkpoint selection or automatic
successor is selected. Do not repeat completed diagonal assessments or the
earlier RGB48 cross-cohort experiment. The five-game goal remains unfinished.

## CPU preparation complete; GPU assessment unselected

Command qualification completes21:23:00.983 UTC November1/source8439f27 in
6.390s (6.421s wrapper):four actual CLI paths,32 header cases,357 finite tensors
per source checkpoint,593 unchanged preparation inputs and272 active Breakout
inputs. Controller qualification completes21:23:12.986 in6.405s (7.115s wrapper):
four retained diagonal audits reproduce15,710.417/1.042 and12,678.125/168.750,
357 invariant tensors per arm, full stored replay/video/guard checks andfour
original-journal fixtures. Both checks pass on their first execution.

Original BreakoutPID847728/invocation72a7b4600a4641bb8c93174668127f3f stays
active andunchanged. Retain both CPU co-host timings without wall-time
correction. Native agent/emulator construction attempts remainzero. No native
build, GPU stage, training update, executable queue, declaration or service is
created. The prepared sources reuse the completed RGB48 cross-cohort workflow;
its old failures/results and all completed GPU stages remain untouched.

Do not rerun successful preparation unless a concrete change requires a
separately recorded check. Separate selection after all five original Breakout
services and terminal collection succeed, clean committed sources, unchanged
pins andidle host remains required. No cross-cohort score is available yet.

| Prepared evidence | SHA256 |
| --- | --- |
| `preflight.json` | `2fa08e6ea5e3314e968376aa5adc3d204467810014f71c8ea2aa3b2be01d7e96` |
| `controller-preflight.json` | `3f124764b31e72d83a2dc137f664ed87e1415c13b1c2ff239fa057dd5c30a344` |

## Preceding learner complete; ready for guarded selection

The original Breakout comparison now completes allfive services and original
collection at21:43:05 UTC November1. Its50.625 sampled/138.750 greedy result
is mixed; the original primary decision andzero two-wall successes remain.
The corrected read-only terminal summary confirms all272 inputs, original
identities, retained recovery and completion hash unchanged, with no original
stage repeated. The host is idle at this closure.

Proceed with the prepared four-arm assessment only after the existing selector
rechecks those original terminal services/completion, all593 preparation inputs,
qualified controller sources, clean committed worktree and same idle host.
Do not rerun CPU qualification or any retained diagonal assessment. All fixed
arms, caps, gates and interpretation limits above remain unchanged. Selection
and launch will retain their own declaration and original process identities.


## Original selection and launch

Selector passes611 pinned inputs and writes declaration SHA256
`b3daecda9ced211e70ee1ca60cf2f3bf0eacb8f4de3ba4e8a86ddb1f740f2b75`.
Original main launches21:47:20.921 UTC November1, source63dd44a,
PID850983/invocationef4177111f444dea8f2d905f7f800544. Original collection
observerPID850986/invocation053db596d5a44ccab149a1d1f3063bb8 launches alongside.
The exact prepared runtime, runner, checkpoint sources, four arms and caps stay
unchanged. No CPU qualification, diagonal GPU assessment or learner repeats.
Preserve these original services; do not overlap another GPU allocation.


## Accepted terminal result

The completed matrix is10007:15,710.417/16,218.750 versus12011:11,982.292/
12,678.125, all24/24 first pyramids. Initial-subtracted gaps remain3,883.333/
3,705.208. Cohort choice alone does not explain the learner difference;
original12011 failure remains and no independent learner is added.

All611 pins,357 invariant tensors per new arm, full outcome replays/videos,
host guards andoriginal terminal collection pass. Four new arms total140,488
actions/561,791 frames/117.899s,zero updates/truncations. Main/collector
complete21:51:01/21:51:22 UTC November1/source63dd44a. Both original services
succeed; no stage fails or repeats. No learner, build, extension, retry or
automatic successor is selected. No GPU allocation remains at closure; do not
rerun completed work. The five-game goal remains unfinished.
